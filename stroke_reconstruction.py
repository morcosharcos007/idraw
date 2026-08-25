
from __future__ import annotations

import math
from xml.sax.saxutils import escape

import cv2
import numpy as np
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize

QUALITY = {
    1: {"spacing": 4.2, "window": 5, "epsilon": 2.2, "min_len": 8.0},
    2: {"spacing": 3.0, "window": 7, "epsilon": 1.5, "min_len": 6.0},
    3: {"spacing": 2.2, "window": 7, "epsilon": 0.9, "min_len": 5.0},
    4: {"spacing": 1.6, "window": 9, "epsilon": 0.55, "min_len": 4.0},
    5: {"spacing": 1.15, "window": 9, "epsilon": 0.28, "min_len": 3.0},
}


def _path_length(points):
    if len(points) < 2:
        return 0.0
    a = np.asarray(points, dtype=np.float32)
    return float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum())


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


def _neighbors(point, point_set):
    y, x = point
    return [
        (y + dy, x + dx)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dx or dy) and (y + dy, x + dx) in point_set
    ]


def _edge_key(a, b):
    return (a, b) if a < b else (b, a)


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.zeros(2, dtype=np.float32)


def _endpoint_tangent(edge, reverse=False, look=8):
    seq = list(reversed(edge)) if reverse else edge
    if len(seq) < 2:
        return np.zeros(2, dtype=np.float32)
    span = min(look, len(seq) - 1)
    return _unit(np.asarray(seq[span]) - np.asarray(seq[0]))


def _prune_spurs(skeleton, max_branch_length):
    skel = skeleton.astype(bool).copy()
    for _ in range(3):
        points = [tuple(p) for p in np.argwhere(skel)]
        point_set = set(points)
        if not point_set:
            break
        nb = {p: _neighbors(p, point_set) for p in points}
        degree = {p: len(v) for p, v in nb.items()}
        remove = set()
        for start in (p for p, d in degree.items() if d == 1):
            if start in remove:
                continue
            path = [start]
            prev = None
            cur = start
            while len(path) <= max_branch_length + 1:
                choices = [q for q in nb.get(cur, []) if q != prev]
                if not choices:
                    break
                nxt = choices[0]
                path.append(nxt)
                if degree.get(nxt, 0) != 2:
                    break
                prev, cur = cur, nxt
            if len(path) <= max_branch_length:
                remove.update(path[:-1])
        if not remove:
            break
        for y, x in remove:
            skel[y, x] = False
    return skel


def _extract_edges(skeleton):
    points = [tuple(p) for p in np.argwhere(skeleton)]
    if not points:
        return [], {}
    point_set = set(points)
    nb = {p: _neighbors(p, point_set) for p in points}
    degree = {p: len(v) for p, v in nb.items()}
    nodes = {p for p, d in degree.items() if d != 2}
    visited = set()
    edges = []

    def walk(start, nxt):
        path = [start, nxt]
        visited.add(_edge_key(start, nxt))
        prev, cur = start, nxt
        while cur not in nodes:
            choices = [
                q for q in nb[cur]
                if q != prev and _edge_key(cur, q) not in visited
            ]
            if not choices:
                break
            incoming = _unit(np.asarray(cur) - np.asarray(prev))
            nxt2 = max(
                choices,
                key=lambda q: float(
                    np.dot(incoming, _unit(np.asarray(q) - np.asarray(cur)))
                ),
            )
            visited.add(_edge_key(cur, nxt2))
            prev, cur = cur, nxt2
            path.append(cur)
        return path

    for node in nodes:
        for nxt in nb[node]:
            if _edge_key(node, nxt) not in visited:
                edges.append(walk(node, nxt))

    if not nodes and points:
        start = points[0]
        if nb[start]:
            edges.append(walk(start, nb[start][0]))

    incident = {}
    for i, edge in enumerate(edges):
        incident.setdefault(edge[0], []).append((i, False))
        incident.setdefault(edge[-1], []).append((i, True))
    return edges, incident


def _best_pairing(items, edges):
    if len(items) < 2:
        return []

    def cost(a, b):
        ia, ra = a
        ib, rb = b
        ta = _endpoint_tangent(edges[ia], ra)
        tb = _endpoint_tangent(edges[ib], rb)
        # Opposing endpoint tangents form a visually continuous stroke.
        return (1.0 + float(np.dot(ta, tb))) / 2.0

    memo = {}

    def solve(rest):
        key = tuple(rest)
        if key in memo:
            return memo[key]
        if len(rest) < 2:
            return 0.0, []
        first = rest[0]
        best = (float("inf"), [])
        for pos in range(1, len(rest)):
            second = rest[pos]
            remaining = rest[1:pos] + rest[pos + 1:]
            score, pairs = solve(remaining)
            score += cost(first, second)
            if score < best[0]:
                best = (score, [(first, second)] + pairs)
        memo[key] = best
        return best

    return solve(tuple(items))[1]


def trace_strokes(skeleton):
    edges, incident = _extract_edges(skeleton)
    if not edges:
        return []

    paired = {}
    for items in incident.values():
        for a, b in _best_pairing(items, edges):
            paired[a] = b
            paired[b] = a

    used = set()
    paths = []

    def extend(start_key):
        edge_index, reverse = start_key
        edge = edges[edge_index]
        coords = list(reversed(edge)) if reverse else list(edge)
        used.add(edge_index)
        current = start_key
        while current in paired:
            nxt = paired[current]
            if nxt[0] in used:
                break
            edge2 = edges[nxt[0]]
            seq = list(reversed(edge2)) if nxt[1] else list(edge2)
            coords.extend(seq[1:] if coords[-1] == seq[0] else seq)
            used.add(nxt[0])
            current = nxt
        return [(x, y) for y, x in coords]

    for i, edge in enumerate(edges):
        if i in used:
            continue
        if len(incident.get(edge[0], [])) == 1:
            paths.append(extend((i, False)))

    for i, edge in enumerate(edges):
        if i not in used:
            paths.append(extend((i, False)))

    return [p for p in paths if len(p) >= 2]


def _merge_near_strokes(paths, gap, alignment=0.90):
    work = [list(p) for p in paths if len(p) >= 2]
    changed = True
    while changed and len(work) > 1:
        changed = False
        best = None
        for i in range(len(work)):
            a = work[i]
            for j in range(i + 1, len(work)):
                b = work[j]
                candidates = []
                for aa in (a, list(reversed(a))):
                    ta = _endpoint_tangent(aa)
                    for bb in (b, list(reversed(b))):
                        tb = _endpoint_tangent(bb)
                        d = float(np.linalg.norm(np.asarray(aa[-1]) - np.asarray(bb[0])))
                        align = float(np.dot(ta, tb))
                        candidates.append((d, align, aa, bb))
                d, align, aa, bb = min(
                    candidates, key=lambda x: x[0] + (1.0 - x[1]) * gap
                )
                if d <= gap and align >= alignment:
                    score = d + (1.0 - align) * gap
                    if best is None or score < best[0]:
                        best = (score, i, j, aa, bb)
        if best is None:
            break
        _, i, j, aa, bb = best
        work[i] = aa + bb[1:]
        work.pop(j)
        changed = True
    return work


def _bbox(path):
    a = np.asarray(path, dtype=np.float32)
    return (
        float(a[:, 0].min()),
        float(a[:, 1].min()),
        float(a[:, 0].max()),
        float(a[:, 1].max()),
    )


def _direction(path, from_start=True, look=8):
    p = np.asarray(path if from_start else list(reversed(path)), dtype=np.float32)
    if len(p) < 2:
        return np.zeros(2, dtype=np.float32)
    span = min(look, len(p) - 1)
    return _unit(p[span] - p[0])


def _start_prior(path):
    x0, y0, x1, y1 = _bbox(path)
    length = _path_length(path)
    span = math.hypot(x1 - x0, y1 - y0)
    # Weak prior only: detached dots should not become the first stroke.
    return 0.002 * (x0 + y0) - 0.015 * length - 0.01 * span


def _transition_score(prev, candidate):
    prev_end = np.asarray(prev[-1], dtype=np.float32)
    prev_dir = _direction(prev, False)
    start = np.asarray(candidate[0], dtype=np.float32)
    travel = float(np.linalg.norm(start - prev_end))
    cand_dir = _direction(candidate, True)
    turn = 1.0 - float(np.clip(np.dot(prev_dir, cand_dir), -1.0, 1.0))
    short_penalty = 1.5 if _path_length(candidate) < 12.0 else 0.0
    return travel + 3.0 * turn + short_penalty


def _order_probable_strokes(paths):
    candidates = [np.asarray(p, dtype=np.float32) for p in paths if len(p) >= 2]
    if not candidates:
        return []

    max_len = max(_path_length(p) for p in candidates)
    substantial = [
        p for p in candidates if _path_length(p) >= max(5.0, 0.30 * max_len)
    ]
    first = min(substantial or candidates, key=_start_prior)
    remaining = [p for p in candidates if p is not first]

    first_rev = first[::-1].copy()
    first = first if _start_prior(first) <= _start_prior(first_rev) else first_rev
    ordered = [first]

    while remaining:
        prev = ordered[-1]
        best = None
        for index, candidate in enumerate(remaining):
            for reverse in (False, True):
                oriented = candidate[::-1].copy() if reverse else candidate.copy()
                score = _transition_score(prev, oriented)
                item = (score, index, reverse, oriented)
                if best is None or item[0] < best[0]:
                    best = item
        _, index, _, chosen = best
        ordered.append(chosen)
        remaining.pop(index)

    return [p.tolist() for p in ordered]


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
        commands.append(
            f"C{c1[0]:.2f},{c1[1]:.2f} {c2[0]:.2f},{c2[1]:.2f} "
            f"{p2[0]:.2f},{p2[1]:.2f}"
        )
    return " ".join(commands)


def handwriting_to_svg(processed, quality=3):
    """Convert a handwriting mask into a probable human-like stroke sequence."""
    if processed is None or processed.ndim != 2:
        raise ValueError("A feldolgozott kép érvénytelen.")

    q = max(1, min(5, int(quality)))
    cfg = QUALITY[q]
    ink = (processed < 128).astype(np.uint8)
    if not np.any(ink):
        raise ValueError("Nem található vektorozható kézírás.")

    skeleton = skeletonize(ink > 0)
    side = min(processed.shape)
    skeleton = _prune_spurs(skeleton, max(4, int(round(side / 300))))
    if not np.any(skeleton):
        raise ValueError("A középvonal-képzés után nem maradt rajzolható vonal.")

    paths = trace_strokes(skeleton)
    paths = _merge_near_strokes(
        paths,
        gap=max(2.0, min(5.0, side / 450.0)),
        alignment=0.92,
    )
    if not paths:
        raise ValueError("A vonalpálya nem állítható elő.")

    prepared = []
    for path in paths:
        if _path_length(path) < cfg["min_len"]:
            continue
        smooth = _smooth(path, q)
        simple = _rdp(smooth, cfg["epsilon"])
        if len(simple) < 2 or _path_length(simple) < cfg["min_len"]:
            continue
        prepared.append(simple)

    if not prepared:
        raise ValueError("A vonalpálya üres lett a zajszűrés után.")

    ordered = _order_probable_strokes(prepared)
    body = []
    total_points = 0
    total_length = 0.0

    for rank, path in enumerate(ordered, 1):
        d = _svg_path(path)
        if not d:
            continue
        length = _path_length(path)
        body.append(
            f'<path id="stroke-{rank:04d}" data-stroke-order="{rank}" '
            f'data-length="{length:.2f}" d="{escape(d)}" fill="none" '
            'stroke="#000000" stroke-width="1" stroke-linecap="round" '
            'stroke-linejoin="round"/>'
        )
        total_points += len(path)
        total_length += length

    if not body:
        raise ValueError("Nem sikerült rajzolható vonalpályát készíteni.")

    h, w = processed.shape
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="210mm" height="297mm" viewBox="0 0 {w} {h}" '
        'preserveAspectRatio="xMidYMid meet">\n'
        '  <metadata id="idraw-reconstruction">'
        'centerline=true;order=geometric-probable;telemetry=unavailable;'
        'stroke_direction=probable;source=raster</metadata>\n'
        '  <g id="idraw-handwriting" fill="none" stroke="#000000" '
        'stroke-linecap="round" stroke-linejoin="round">\n'
        + "\n".join("    " + x for x in body)
        + '\n  </g>\n</svg>\n'
    )

    return svg, {
        "paths": len(body),
        "points": total_points,
        "centerline_length_px": round(total_length, 1),
        "quality": q,
        "order": "geometric-probable",
        "telemetry": "not_recoverable_from_flat_raster",
    }
