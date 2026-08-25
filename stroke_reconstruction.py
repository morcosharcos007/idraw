from __future__ import annotations

import math
from xml.sax.saxutils import escape

import cv2
import numpy as np
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize

QUALITY = {
    1: {"spacing": 4.2, "window": 5, "epsilon": 2.2, "min_len": 8.0, "merge_gap": 2.2, "spur": 8},
    2: {"spacing": 3.0, "window": 7, "epsilon": 1.5, "min_len": 6.0, "merge_gap": 2.8, "spur": 7},
    3: {"spacing": 2.2, "window": 7, "epsilon": 0.9, "min_len": 5.0, "merge_gap": 3.5, "spur": 6},
    4: {"spacing": 1.6, "window": 9, "epsilon": 0.55, "min_len": 4.0, "merge_gap": 4.2, "spur": 5},
    5: {"spacing": 1.15, "window": 9, "epsilon": 0.28, "min_len": 3.0, "merge_gap": 4.8, "spur": 4},
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
        result = np.column_stack((
            savgol_filter(arr[:, 0], window, 3, mode="interp"),
            savgol_filter(arr[:, 1], window, 3, mode="interp"),
        )).astype(np.float32)
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
    for _ in range(4):
        points = [tuple(p) for p in np.argwhere(skel)]
        point_set = set(points)
        if not point_set:
            break
        nb = {p: _neighbors(p, point_set) for p in points}
        degree = {p: len(v) for p, v in nb.items()}
        remove = set()
        for start, d in degree.items():
            if d != 1:
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
            choices = [q for q in nb[cur] if q != prev and _edge_key(cur, q) not in visited]
            if not choices:
                break
            incoming = _unit(np.asarray(cur) - np.asarray(prev))
            nxt2 = max(choices, key=lambda q: float(np.dot(incoming, _unit(np.asarray(q) - np.asarray(cur)))))
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


def _pair_junctions(items, edges):
    """Pair only the most directionally continuous branches at a junction."""
    if len(items) < 2:
        return []

    pairs = []
    remaining = list(items)
    while len(remaining) >= 2:
        best = None
        for ai in range(len(remaining)):
            for bi in range(ai + 1, len(remaining)):
                a, b = remaining[ai], remaining[bi]
                ta = _endpoint_tangent(edges[a[0]], a[1])
                tb = _endpoint_tangent(edges[b[0]], b[1])
                # At a junction, a smooth continuation has opposing travel vectors.
                straightness = (1.0 + float(np.dot(ta, tb))) / 2.0
                score = straightness
                if best is None or score < best[0]:
                    best = (score, ai, bi, a, b)
        if best is None or best[0] > 0.36:
            break
        _, ai, bi, a, b = best
        pairs.append((a, b))
        for idx in sorted((ai, bi), reverse=True):
            remaining.pop(idx)
    return pairs


def trace_strokes(skeleton):
    edges, incident = _extract_edges(skeleton)
    if not edges:
        return []
    paired = {}
    for items in incident.values():
        for a, b in _pair_junctions(items, edges):
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


def _merge_near_strokes(paths, gap, alignment=0.86):
    work = [list(p) for p in paths if len(p) >= 2]
    while len(work) > 1:
        best = None
        for i, a in enumerate(work):
            la = _path_length(a)
            for j in range(i + 1, len(work)):
                b = work[j]
                lb = _path_length(b)
                # Never glue two substantial strokes merely because they touch visually.
                if min(la, lb) > 22.0:
                    local_gap = min(gap, 3.0)
                else:
                    local_gap = gap
                candidates = []
                for aa in (a, list(reversed(a))):
                    ta = _endpoint_tangent(aa)
                    for bb in (b, list(reversed(b))):
                        tb = _endpoint_tangent(bb)
                        d = float(np.linalg.norm(np.asarray(aa[-1]) - np.asarray(bb[0])))
                        align = float(np.dot(ta, tb))
                        candidates.append((d, align, aa, bb))
                d, align, aa, bb = min(candidates, key=lambda x: x[0] + (1.0 - x[1]) * local_gap)
                if d <= local_gap and align >= alignment:
                    score = d + (1.0 - align) * local_gap
                    if best is None or score < best[0]:
                        best = (score, i, j, aa, bb)
        if best is None:
            break
        _, i, j, aa, bb = best
        work[i] = aa + bb[1:]
        work.pop(j)
    return work


def _bbox(path):
    a = np.asarray(path, dtype=np.float32)
    return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())


def _direction(path, from_start=True, look=8):
    p = np.asarray(path if from_start else list(reversed(path)), dtype=np.float32)
    if len(p) < 2:
        return np.zeros(2, dtype=np.float32)
    span = min(look, len(p) - 1)
    return _unit(p[span] - p[0])


def _is_dot(path):
    length = _path_length(path)
    x0, y0, x1, y1 = _bbox(path)
    w, h = x1 - x0, y1 - y0
    return length <= 14.0 and max(w, h) <= 10.0 and min(w, h) > 0.0


def _start_prior(path):
    x0, y0, x1, y1 = _bbox(path)
    length = _path_length(path)
    span = math.hypot(x1 - x0, y1 - y0)
    dot_penalty = 80.0 if _is_dot(path) else 0.0
    return 0.002 * (x0 + y0) - 0.015 * length - 0.01 * span + dot_penalty


def _transition_score(prev, candidate, mode="balanced"):
    prev_end = np.asarray(prev[-1], dtype=np.float32)
    prev_dir = _direction(prev, False)
    start = np.asarray(candidate[0], dtype=np.float32)
    travel = float(np.linalg.norm(start - prev_end))
    cand_dir = _direction(candidate, True)
    turn = 1.0 - float(np.clip(np.dot(prev_dir, cand_dir), -1.0, 1.0))
    dot_penalty = 8.0 if _is_dot(candidate) else 0.0
    if mode == "efficient":
        return 2.4 * travel + 1.0 * turn + dot_penalty
    if mode == "natural":
        return 1.0 * travel + 4.0 * turn + dot_penalty
    return 1.7 * travel + 2.5 * turn + dot_penalty


def _order_probable_strokes(paths, mode="balanced"):
    candidates = [np.asarray(p, dtype=np.float32) for p in paths if len(p) >= 2]
    if not candidates:
        return []
    first = min(candidates, key=_start_prior)
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
                score = _transition_score(prev, oriented, mode)
                # For natural/balanced modes, keep compact detached marks toward the end.
                if _is_dot(oriented) and len(remaining) > 3:
                    score += 15.0
                item = (score, index, reverse)
                if best is None or item < best[:3]:
                    best = (score, index, reverse)
        _, index, reverse = best
        chosen = remaining.pop(index)
        ordered.append(chosen[::-1].copy() if reverse else chosen)
    return [p.tolist() for p in ordered]


def _bezier_path(points):
    """Cubic Bézier approximation with bounded local tangent handles."""
    p = np.asarray(points, dtype=np.float32)
    if len(p) < 2:
        return ""
    if len(p) == 2:
        return f"M{p[0,0]:.2f},{p[0,1]:.2f} L{p[1,0]:.2f},{p[1,1]:.2f}"
    commands = [f"M{p[0,0]:.2f},{p[0,1]:.2f}"]
    for i in range(len(p) - 1):
        p0, p1 = p[max(0, i - 1)], p[i]
        p2, p3 = p[i + 1], p[min(len(p) - 1, i + 2)]
        chord = float(np.linalg.norm(p2 - p1))
        if chord <= 1e-5:
            continue
        t1 = _unit(p2 - p0)
        t2 = _unit(p3 - p1)
        handle = min(chord * 0.34, 4.5)
        c1 = p1 + t1 * handle
        c2 = p2 - t2 * handle
        commands.append(f"C{c1[0]:.2f},{c1[1]:.2f} {c2[0]:.2f},{c2[1]:.2f} {p2[0]:.2f},{p2[1]:.2f}")
    return " ".join(commands)


def handwriting_to_svg(processed, quality=3, ordering="balanced"):
    """Convert a handwriting mask into a plotter-oriented probable stroke sequence."""
    if processed is None or processed.ndim != 2:
        raise ValueError("A feldolgozott kép érvénytelen.")
    q = max(1, min(5, int(quality)))
    mode = ordering if ordering in {"natural", "balanced", "efficient"} else "balanced"
    cfg = QUALITY[q]
    ink = (processed < 128).astype(np.uint8)
    if not np.any(ink):
        raise ValueError("Nem található vektorozható kézírás.")

    # A small close before skeletonization repairs camera-scale pinholes without blurring the writing.
    kernel = np.ones((3, 3), np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel, iterations=1)
    skeleton = skeletonize(ink > 0)
    side = min(processed.shape)
    skeleton = _prune_spurs(skeleton, cfg["spur"])
    if not np.any(skeleton):
        raise ValueError("A középvonal-képzés után nem maradt rajzolható vonal.")

    paths = trace_strokes(skeleton)
    paths = _merge_near_strokes(paths, gap=cfg["merge_gap"], alignment=0.86)
    if not paths:
        raise ValueError("A vonalpálya nem állítható elő.")

    prepared = []
    for path in paths:
        length = _path_length(path)
        if length < cfg["min_len"]:
            continue
        smooth = _smooth(path, q)
        simple = _rdp(smooth, cfg["epsilon"])
        if len(simple) >= 2 and _path_length(simple) >= cfg["min_len"]:
            prepared.append(simple)
    if not prepared:
        raise ValueError("A vonalpálya üres lett a zajszűrés után.")

    ordered = _order_probable_strokes(prepared, mode=mode)
    body = []
    total_points = 0
    total_length = 0.0
    dots = 0
    for rank, path in enumerate(ordered, 1):
        d = _bezier_path(path)
        if not d:
            continue
        length = _path_length(path)
        if _is_dot(path):
            dots += 1
        body.append(
            f'<path id="stroke-{rank:04d}" data-stroke-order="{rank}" '
            f'data-length="{length:.2f}" d="{escape(d)}" fill="none" '
            'stroke="#000000" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"/>'
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
        f'centerline=true;order={mode};bezier=true;dots={dots};telemetry=unavailable;'
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
        "order": mode,
        "bezier": True,
        "dots": dots,
        "telemetry": "not_recoverable_from_flat_raster",
    }
