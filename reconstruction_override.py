from __future__ import annotations

import math
from xml.sax.saxutils import escape

import cv2
import numpy as np
from scipy.signal import savgol_filter

import app as _app

QUALITY = {
    # simplify + smoothing + sampling are intentionally all different
    1: {"spacing": 4.2, "window": 5, "epsilon": 2.2, "min_len": 8.0},
    2: {"spacing": 3.0, "window": 7, "epsilon": 1.5, "min_len": 6.0},
    3: {"spacing": 2.2, "window": 7, "epsilon": 0.9, "min_len": 5.0},
    4: {"spacing": 1.6, "window": 9, "epsilon": 0.55, "min_len": 4.0},
    5: {"spacing": 1.15, "window": 9, "epsilon": 0.28, "min_len": 3.0},
}


def _resample(points, spacing):
    arr = np.asarray(points, dtype=np.float32)
    if len(arr) < 2:
        return arr
    seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(cumulative[-1])
    if total <= 1e-6:
        return arr[:1]
    count = max(2, int(math.ceil(total / spacing)) + 1)
    samples = np.linspace(0.0, total, count, dtype=np.float32)
    x = np.interp(samples, cumulative, arr[:, 0])
    y = np.interp(samples, cumulative, arr[:, 1])
    return np.column_stack((x, y)).astype(np.float32)


def _smooth(points, quality):
    cfg = QUALITY[quality]
    arr = _resample(points, cfg["spacing"])
    if len(arr) < 7:
        return arr
    window = min(cfg["window"], len(arr) if len(arr) % 2 else len(arr) - 1)
    if window < 5:
        return arr
    try:
        x = savgol_filter(arr[:, 0], window, 3, mode="interp")
        y = savgol_filter(arr[:, 1], window, 3, mode="interp")
        result = np.column_stack((x, y)).astype(np.float32)
        result[0], result[-1] = arr[0], arr[-1]
        return result
    except ValueError:
        return arr


def _rdp(points, epsilon):
    arr = np.asarray(points, dtype=np.float32)
    if len(arr) <= 2:
        return arr
    out = cv2.approxPolyDP(arr.reshape(-1, 1, 2), epsilon, False)
    return out.reshape(-1, 2).astype(np.float32)


def _bbox(path):
    a = np.asarray(path, dtype=np.float32)
    return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())


def _orient(path):
    if len(path) < 2:
        return path
    a = np.asarray(path[0], dtype=np.float32)
    b = np.asarray(path[-1], dtype=np.float32)
    d = b - a
    if abs(float(d[0])) >= abs(float(d[1])):
        return path if a[0] <= b[0] else list(reversed(path))
    return path if a[1] <= b[1] else list(reversed(path))


def _direction(path, from_start=True):
    p = np.asarray(path if from_start else list(reversed(path)), dtype=np.float32)
    if len(p) < 2:
        return np.zeros(2, dtype=np.float32)
    span = min(8, len(p) - 1)
    d = p[span] - p[0]
    n = np.linalg.norm(d)
    return d / n if n > 1e-6 else np.zeros(2, dtype=np.float32)


def _order_probable_strokes(paths):
    """Infer pen-lift order from geometry; never claim exact historical order."""
    paths = [_orient(list(p)) for p in paths if len(p) >= 2]
    if not paths:
        return []

    # Main strokes first; detached dots/very small marks follow the nearest
    # preceding stroke. This avoids the common 'dot drawn first' plotter result.
    paths.sort(key=lambda p: (_bbox(p)[1], _bbox(p)[0], -_app._path_length(p)))
    ordered = [paths.pop(0)]

    while paths:
        prev = np.asarray(ordered[-1][-1], dtype=np.float32)
        prev_dir = _direction(ordered[-1], False)
        best_i = 0
        best_score = float("inf")
        for i, candidate in enumerate(paths):
            start = np.asarray(candidate[0], dtype=np.float32)
            travel = float(np.linalg.norm(start - prev))
            candidate_dir = _direction(candidate, True)
            # Small travel is good; continuing in a similar direction is also good.
            turn = 1.0 - float(np.clip(np.dot(prev_dir, candidate_dir), -1.0, 1.0))
            x0, y0, x1, y1 = _bbox(candidate)
            reading_penalty = 0.015 * abs(x0 - prev[0]) + 0.006 * abs(y0 - prev[1])
            # Very small detached marks are slightly deferred.
            area = max(1.0, (x1 - x0 + 1.0) * (y1 - y0 + 1.0))
            small_penalty = 2.0 if area < 35.0 else 0.0
            score = travel + 1.8 * turn + reading_penalty + small_penalty
            if score < best_score:
                best_score = score
                best_i = i
        ordered.append(paths.pop(best_i))
    return ordered


def _svg_path(points):
    p = np.asarray(points, dtype=np.float32)
    if len(p) < 2:
        return ""
    if len(p) == 2:
        return f"M{p[0,0]:.2f},{p[0,1]:.2f} L{p[1,0]:.2f},{p[1,1]:.2f}"
    commands = [f"M{p[0,0]:.2f},{p[0,1]:.2f}"]
    for i in range(len(p) - 1):
        p0 = p[max(0, i - 1)]
        p1 = p[i]
        p2 = p[i + 1]
        p3 = p[min(len(p) - 1, i + 2)]
        c1 = p1 + (p2 - p0) * (0.48 / 6.0)
        c2 = p2 - (p3 - p1) * (0.48 / 6.0)
        lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
        c1, c2 = np.clip(c1, lo, hi), np.clip(c2, lo, hi)
        commands.append(f"C{c1[0]:.2f},{c1[1]:.2f} {c2[0]:.2f},{c2[1]:.2f} {p2[0]:.2f},{p2[1]:.2f}")
    return " ".join(commands)


def handwriting_to_svg(processed, quality=3):
    """Reconstruct centerline geometry and infer a probable pen-motion order."""
    if processed is None or processed.ndim != 2:
        raise ValueError("A feldolgozott kép érvénytelen.")
    q = max(1, min(5, int(quality)))
    cfg = QUALITY[q]
    ink = (processed < 128).astype(np.uint8)
    if not np.any(ink):
        raise ValueError("Nem található vektorozható kézírás.")

    # Keep the existing conservative UUNA-TEK-style centerline topology code.
    skeleton = _app.skeletonize(ink > 0)
    side = min(processed.shape)
    skeleton = _app._prune_spurs(skeleton, max(4, int(round(side / 300))))
    if not np.any(skeleton):
        raise ValueError("A középvonal-képzés után nem maradt rajzolható vonal.")

    paths = _app.trace_strokes(skeleton)
    paths = _app._merge_near_strokes(paths, gap=max(2.0, min(5.0, side / 450.0)), alignment=0.92)
    if not paths:
        raise ValueError("A vonalpálya nem állítható elő.")

    prepared = []
    for path in paths:
        if _app._path_length(path) < cfg["min_len"]:
            continue
        smooth = _smooth(path, q)
        simple = _rdp(smooth, cfg["epsilon"])
        if len(simple) < 2 or _app._path_length(simple) < cfg["min_len"]:
            continue
        # trace_strokes returns x,y coordinates already.
        prepared.append(simple)

    if not prepared:
        raise ValueError("A vonalpálya üres lett a zajszűrés után.")

    ordered = _order_probable_strokes(prepared)
    body = []
    total_points = 0
    for path in ordered:
        d = _svg_path(path)
        if not d:
            continue
        body.append(
            f'<path d="{escape(d)}" fill="none" stroke="#000000" stroke-width="1" '
            'stroke-linecap="round" stroke-linejoin="round"/>'
        )
        total_points += len(path)

    if not body:
        raise ValueError("Nem sikerült rajzolható vonalpályát készíteni.")

    h, w = processed.shape
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="210mm" height="297mm" viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet">\n'
        '  <g id="idraw-handwriting" fill="none" stroke="#000000" stroke-linecap="round" stroke-linejoin="round">\n'
        + "\n  ".join(body)
        + '\n  </g>\n</svg>\n'
    )
    return svg, {"paths": len(body), "points": total_points, "quality": q, "order": "geometric-probable"}
