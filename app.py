import base64
import os
import re
import secrets
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from flask import Flask, Response, abort, render_template, request, url_for
from skimage.morphology import skeletonize
from scipy.signal import savgol_filter

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

STATE_DIR = Path(
    os.environ.get(
        "IDRAW_STATE_DIR",
        str(Path(os.environ.get("TMPDIR", "/tmp")) / "idraw-state"),
    )
)
STATE_DIR.mkdir(parents=True, exist_ok=True)

MAX_VECTOR_DIMENSION = 1800


def safe_token(value):
    return re.sub(r"[^A-Za-z0-9_-]", "", value or "")[:80]


def new_token():
    return secrets.token_urlsafe(24)


def encode_png(img):
    ok, data = cv2.imencode(".png", img)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(data.tobytes()).decode("ascii")


def save_png(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("A kép mentése nem sikerült.")
    path.write_bytes(data.tobytes())


def load_png(path, flags):
    if not path.is_file():
        return None
    return cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), flags)


def state_paths(token):
    token = safe_token(token)
    if not token:
        return None
    folder = STATE_DIR / token
    return {
        "folder": folder,
        "original": folder / "original.png",
        "processed": folder / "processed.png",
        "svg": folder / "output.svg",
    }


def save_state(token, original, processed):
    p = state_paths(token)
    if not p:
        raise ValueError("Érvénytelen feldolgozási azonosító.")
    p["folder"].mkdir(parents=True, exist_ok=True)
    save_png(p["original"], original)
    save_png(p["processed"], processed)
    return p


def load_state(token):
    p = state_paths(token)
    if not p or not p["processed"].is_file():
        return None
    original = load_png(p["original"], cv2.IMREAD_COLOR)
    processed = load_png(p["processed"], cv2.IMREAD_GRAYSCALE)
    if original is None or processed is None:
        return None
    return p, original, processed


def cleanup_old_states(max_age_hours=6):
    cutoff = time.time() - max_age_hours * 3600
    if not STATE_DIR.exists():
        return
    for folder in list(STATE_DIR.iterdir()):
        try:
            if folder.is_dir() and folder.stat().st_mtime < cutoff:
                for child in folder.iterdir():
                    child.unlink(missing_ok=True)
                folder.rmdir()
        except OSError:
            pass


def process(img, threshold, denoise):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    scale = min(1.0, MAX_VECTOR_DIMENSION / float(max(h, w)))
    if scale < 1.0:
        size = (
            max(1, int(round(w * scale))),
            max(1, int(round(h * scale))),
        )
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    k = max(1, min(9, int(denoise)))
    if k % 2 == 0:
        k += 1
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    height, width = gray.shape
    side = min(height, width)

    block = max(15, min(81, (int(round(side / 35)) | 1)))
    adaptive_c = int(np.interp(int(threshold), [80, 240], [14, 3]))

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        adaptive_c,
    )

    paper = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=max(8.0, min(32.0, side / 28)),
    )
    darkness = paper.astype(np.int16) - gray.astype(np.int16)
    cutoff = int(np.interp(int(threshold), [80, 240], [24, 7]))
    contrast = np.where(darkness >= cutoff, 255, 0).astype(np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    blue = (
        (hue >= 85)
        & (hue <= 140)
        & (sat >= 18)
        & (val <= 252)
        & (darkness >= max(3, cutoff // 2))
        & (adaptive > 0)
    )

    ink = np.where((contrast > 0) | blue, 255, 0).astype(np.uint8)
    ink = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8),
        iterations=1,
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    scale_area = height * width / 1_000_000.0
    min_area = max(4, int(round(scale_area * (2 + k))))
    main_area = max(min_area * 3, int(round(scale_area * (8 + k))))

    main = np.zeros_like(ink)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        boundary = (
            x <= 0 or y <= 0
            or x + cw >= width
            or y + ch >= height
        )
        if not boundary and area >= main_area:
            main[labels == i] = 255

    radius = max(4, min(18, int(round(side / 100))))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1),
    )
    near = cv2.dilate(main, kernel)

    cleaned = np.zeros_like(ink)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        boundary = (
            x <= 0 or y <= 0
            or x + cw >= width
            or y + ch >= height
        )
        if boundary or area < min_area:
            continue
        mask = labels == i
        if area >= main_area or np.any(near[mask] > 0):
            cleaned[mask] = 255

    return cv2.bitwise_not(cleaned)



def neighbors(point, point_set):
    y, x = point
    return [
        (y + dy, x + dx)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dx or dy) and (y + dy, x + dx) in point_set
    ]


def edge_key(a, b):
    return (a, b) if a < b else (b, a)


def _path_length(points):
    if len(points) < 2:
        return 0.0
    arr = np.asarray(points, dtype=np.float32)
    return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.zeros(2, dtype=np.float32)


def prune_skeleton(skeleton, iterations=2, min_branch_length=10):
    """
    Remove short skeleton spurs before path tracing.

    Camera noise and thresholding often produce tiny branches. Those branches
    are the main source of the "hairy / zig-zag" plotter paths. We repeatedly
    walk from endpoints and delete branches that terminate quickly at a real
    stroke or junction.
    """
    skel = skeleton.copy().astype(bool)

    for _ in range(max(1, int(iterations))):
        points = [tuple(p) for p in np.argwhere(skel)]
        if not points:
            return skel

        point_set = set(points)
        nb = {p: neighbors(p, point_set) for p in points}
        degree = {p: len(v) for p, v in nb.items()}
        endpoints = [p for p, d in degree.items() if d == 1]

        remove = set()

        for start in endpoints:
            if start in remove or start not in point_set:
                continue

            path = [start]
            prev = None
            cur = start

            while True:
                choices = [q for q in nb.get(cur, []) if q != prev]
                if not choices:
                    break

                nxt = choices[0]
                if degree.get(nxt, 2) != 2:
                    path.append(nxt)
                    break

                path.append(nxt)
                prev, cur = cur, nxt

                if len(path) > min_branch_length + 2:
                    break

            # Only prune short branches. Long endpoints are real handwriting
            # starts/ends and must remain intact.
            if len(path) <= min_branch_length:
                remove.update(path[:-1])

        if not remove:
            break

        for y, x in remove:
            skel[y, x] = False

    return skel


def trace_skeleton_paths(skeleton):
    """
    Convert a one-pixel skeleton into continuous stroke candidates.

    Junctions are paired by tangent direction rather than arbitrary graph
    order. This preserves the natural continuation of a handwriting stroke
    through crossings and greatly reduces pen-up/pen-down fragmentation.
    """
    points = [tuple(p) for p in np.argwhere(skeleton)]
    if not points:
        return []

    point_set = set(points)
    nb = {p: neighbors(p, point_set) for p in points}
    degree = {p: len(v) for p, v in nb.items()}
    nodes = {p for p, d in degree.items() if d != 2}

    if not nodes:
        start = points[0]
        path = [start]
        prev = None
        cur = start
        seen = set()

        while True:
            choices = [
                q for q in nb[cur]
                if q != prev and edge_key(cur, q) not in seen
            ]
            if not choices:
                break

            nxt = choices[0]
            seen.add(edge_key(cur, nxt))
            prev, cur = cur, nxt
            path.append(cur)

            if cur == start:
                break

        return [[(x, y) for y, x in path]]

    edges = []
    seen = set()

    for node in nodes:
        for nxt in nb[node]:
            first_key = edge_key(node, nxt)
            if first_key in seen:
                continue

            path = [node]
            prev, cur = node, nxt
            seen.add(first_key)
            path.append(cur)

            while cur not in nodes:
                choices = [q for q in nb[cur] if q != prev]
                if not choices:
                    break

                # If there are multiple choices here, prefer the direction
                # that continues the incoming tangent.
                incoming = _unit(np.asarray(cur) - np.asarray(prev))
                best_q = None
                best_score = -2.0

                for q in choices:
                    if edge_key(cur, q) in seen:
                        continue
                    outgoing = _unit(np.asarray(q) - np.asarray(cur))
                    score = float(np.dot(incoming, outgoing))
                    if score > best_score:
                        best_score = score
                        best_q = q

                if best_q is None:
                    break

                q = best_q
                key = edge_key(cur, q)
                seen.add(key)
                prev, cur = cur, q
                path.append(cur)

            if len(path) >= 2:
                edges.append(path)

    incident = {}
    for i, edge in enumerate(edges):
        incident.setdefault(edge[0], []).append((i, False))
        incident.setdefault(edge[-1], []).append((i, True))

    paired = {}

    for node, items in incident.items():
        remaining = list(items)
        node_xy = np.asarray(node, dtype=np.float32)

        while len(remaining) >= 2:
            best = None
            best_score = float("inf")

            for a in range(len(remaining)):
                for b in range(a + 1, len(remaining)):
                    ia, reverse_a = remaining[a]
                    ib, reverse_b = remaining[b]

                    ea = edges[ia]
                    eb = edges[ib]

                    # Direction away from the junction for each edge.
                    va_point = ea[-2] if reverse_a else ea[1]
                    vb_point = eb[-2] if reverse_b else eb[1]

                    va = _unit(np.asarray(va_point) - node_xy)
                    vb = _unit(np.asarray(vb_point) - node_xy)

                    # A natural stroke continuation enters the junction from
                    # one side and leaves on the opposite side.
                    cosine = float(np.dot(va, vb))
                    score = abs(cosine + 1.0)

                    if score < best_score:
                        best_score = score
                        best = (a, b)

            if best is None:
                break

            a, b = best
            ia, side_a = remaining[a]
            ib, side_b = remaining[b]

            paired[(ia, side_a)] = (ib, side_b)
            paired[(ib, side_b)] = (ia, side_a)

            for pos in sorted((a, b), reverse=True):
                remaining.pop(pos)

    used = set()
    paths = []

    def extend(edge_index, reverse):
        edge = edges[edge_index]
        coords = list(reversed(edge)) if reverse else list(edge)
        used.add(edge_index)

        current = (edge_index, reverse)

        while current in paired:
            next_edge, next_reverse = paired[current]
            if next_edge in used:
                break

            edge2 = edges[next_edge]
            coords2 = list(reversed(edge2)) if next_reverse else list(edge2)

            if coords[-1] == coords2[0]:
                coords.extend(coords2[1:])
            else:
                coords.extend(coords2)

            used.add(next_edge)
            current = (next_edge, next_reverse)

        return [(x, y) for y, x in coords]

    # Prefer real endpoints as stroke starts.
    for i, edge in enumerate(edges):
        if i in used:
            continue
        if len(incident.get(edge[0], [])) == 1:
            paths.append(extend(i, False))

    for i in range(len(edges)):
        if i not in used:
            paths.append(extend(i, False))

    return [p for p in paths if len(p) >= 2]



def merge_stroke_fragments(paths, max_gap=10.0, min_alignment=0.72):
    """
    Join path fragments that clearly belong to the same physical stroke.

    Skeletonization can split a perfectly continuous pen stroke at a noisy
    pixel or a tiny crossing. We only merge when endpoints are close AND the
    outgoing tangents agree, so separate letters are not casually joined.
    """
    work = [list(p) for p in paths if len(p) >= 2]

    if len(work) < 2:
        return work

    max_gap = float(max_gap)

    changed = True
    while changed and len(work) > 1:
        changed = False
        best = None
        best_score = float("inf")

        for i in range(len(work)):
            a = work[i]
            a_start = np.asarray(a[0], dtype=np.float32)
            a_end = np.asarray(a[-1], dtype=np.float32)
            a_dir_start = _unit(
                np.asarray(a[min(4, len(a)-1)]) - a_start
            )
            a_dir_end = _unit(
                a_end - np.asarray(a[max(0, len(a)-5)])
            )

            for j in range(i + 1, len(work)):
                b = work[j]
                b_start = np.asarray(b[0], dtype=np.float32)
                b_end = np.asarray(b[-1], dtype=np.float32)
                b_dir_start = _unit(
                    b[min(4, len(b)-1)] - b_start
                )
                b_dir_end = _unit(
                    b_end - b[max(0, len(b)-5)]
                )

                candidates = [
                    (np.linalg.norm(a_end-b_start),
                     float(np.dot(a_dir_end, b_dir_start)), False, False),
                    (np.linalg.norm(a_end-b_end),
                     float(np.dot(a_dir_end, -b_dir_end)), False, True),
                    (np.linalg.norm(a_start-b_start),
                     float(np.dot(-a_dir_start, b_dir_start)), True, False),
                    (np.linalg.norm(a_start-b_end),
                     float(np.dot(-a_dir_start, -b_dir_end)), True, True),
                ]

                for distance, alignment, reverse_a, reverse_b in candidates:
                    if distance > max_gap or alignment < min_alignment:
                        continue

                    score = float(distance) + (1.0 - alignment) * max_gap
                    if score < best_score:
                        best_score = score
                        best = (i, j, reverse_a, reverse_b)

        if best is None:
            break

        i, j, reverse_a, reverse_b = best
        a = list(reversed(work[i])) if reverse_a else work[i]
        b = list(reversed(work[j])) if reverse_b else work[j]

        if np.linalg.norm(np.asarray(a[-1]) - np.asarray(b[0])) <= max_gap:
            merged = a + b[1:]
        else:
            merged = b + a[1:]

        work[i] = merged
        work.pop(j)
        changed = True

    return work


def order_paths(paths):
    """Minimize pen-up travel while preserving each stroke direction."""
    remaining = [list(p) for p in paths if len(p) >= 2]
    if not remaining:
        return []

    # Start at the upper-left stroke, then greedily choose the nearest
    # endpoint. This is deliberately conservative: we never connect two
    # separate strokes just to save travel.
    remaining.sort(key=lambda p: (p[0][1], p[0][0]))
    ordered = [remaining.pop(0)]

    while remaining:
        tail = np.asarray(ordered[-1][-1], dtype=np.float32)
        best_index = 0
        best_distance = float("inf")
        reverse = False

        for i, path in enumerate(remaining):
            start = np.asarray(path[0], dtype=np.float32)
            end = np.asarray(path[-1], dtype=np.float32)

            d_start = float(np.linalg.norm(tail - start))
            d_end = float(np.linalg.norm(tail - end))

            if d_start < best_distance:
                best_index, best_distance, reverse = i, d_start, False
            if d_end < best_distance:
                best_index, best_distance, reverse = i, d_end, True

        path = remaining.pop(best_index)
        if reverse:
            path.reverse()
        ordered.append(path)

    return ordered


def resample_path(points, spacing=2.5):
    """Uniform arc-length resampling for stable smoothing and SVG output."""
    arr = np.asarray(points, dtype=np.float32)
    if len(arr) <= 2:
        return arr

    deltas = np.diff(arr, axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    total = float(cumulative[-1])

    if total < spacing * 2:
        return arr

    samples = np.arange(0.0, total + 1e-6, spacing, dtype=np.float32)
    x = np.interp(samples, cumulative, arr[:, 0])
    y = np.interp(samples, cumulative, arr[:, 1])
    return np.column_stack((x, y)).astype(np.float32)


def smooth_points(points, quality=3):
    """
    Smooth a stroke without turning handwriting into a polygon.

    Savitzky-Golay is used after arc-length resampling. It preserves the
    overall shape and local bends much better than aggressive Douglas-Peucker
    simplification, while removing the pixel-level staircase that the plotter
    would otherwise reproduce.
    """
    arr = resample_path(points, spacing=2.0)
    if len(arr) <= 4:
        return arr

    q = max(1, min(5, int(quality)))
    window = 5 + q * 4
    if window >= len(arr):
        window = len(arr) - 1 if len(arr) % 2 == 0 else len(arr)
    if window < 5:
        return arr

    if window % 2 == 0:
        window -= 1

    polyorder = 3
    try:
        x = savgol_filter(
            arr[:, 0], window_length=window, polyorder=polyorder,
            mode="interp"
        )
        y = savgol_filter(
            arr[:, 1], window_length=window, polyorder=polyorder,
            mode="interp"
        )
        result = np.column_stack((x, y)).astype(np.float32)
    except ValueError:
        result = arr

    # Never let smoothing move the endpoints. This is important for
    # handwriting joins and for plotter stroke placement.
    result[0] = arr[0]
    result[-1] = arr[-1]
    return result


def svg_path(points):
    """
    Convert a smoothed polyline to a compact cubic Bézier path.

    Curves are used instead of a dense polygon so the plotter software gets
    a genuinely continuous path. Round caps/joins are set on the SVG path.
    """
    p = [np.asarray(x, dtype=np.float32) for x in points]
    if len(p) < 2:
        return ""

    if len(p) == 2:
        return (
            f"M{p[0][0]:.2f},{p[0][1]:.2f} "
            f"L{p[1][0]:.2f},{p[1][1]:.2f}"
        )

    commands = [f"M{p[0][0]:.2f},{p[0][1]:.2f}"]

    for i in range(len(p) - 1):
        p0 = p[max(0, i - 1)]
        p1 = p[i]
        p2 = p[i + 1]
        p3 = p[min(len(p) - 1, i + 2)]

        # Catmull-Rom -> cubic Bézier conversion. A 0.75 tension factor is
        # deliberately restrained to avoid loops/overshoot in signatures.
        tension = 0.75
        c1 = p1 + (p2 - p0) * (tension / 6.0)
        c2 = p2 - (p3 - p1) * (tension / 6.0)

        commands.append(
            "C"
            f"{c1[0]:.2f},{c1[1]:.2f} "
            f"{c2[0]:.2f},{c2[1]:.2f} "
            f"{p2[0]:.2f},{p2[1]:.2f}"
        )

    return " ".join(commands)


def handwriting_to_svg(processed, quality=3):
    """
    Convert the processed handwriting bitmap to plotter-ready SVG.

    Pipeline:
      bitmap -> cleanup -> skeleton -> spur removal -> stroke graph ->
      arc-length resampling -> Savitzky-Golay smoothing -> cubic SVG paths.

    This is intentionally stroke-based (no fills), matching the SVG workflow
    recommended by UUNA TEK for pen plotters.
    """
    ink = (processed < 128).astype(np.uint8) * 255

    # Remove tiny components but keep genuine small handwriting marks.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    cleaned = np.zeros_like(ink)
    area_limit = max(4, int((ink.shape[0] * ink.shape[1]) / 2_000_000))

    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= area_limit:
            cleaned[labels == i] = 255

    # Close tiny raster gaps before skeletonization. The kernel is deliberately
    # small: over-closing would merge nearby letters.
    side = min(cleaned.shape)
    close_size = max(3, min(7, int(round(side / 600)) * 2 + 3))
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_size, close_size),
    )
    cleaned = cv2.morphologyEx(
        cleaned, cv2.MORPH_CLOSE, close_kernel, iterations=1
    )

    # Remove isolated salt-and-pepper pixels after closing.
    cleaned = cv2.morphologyEx(
        cleaned, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1
    )

    skeleton = skeletonize(cleaned > 0)
    skeleton = prune_skeleton(
        skeleton,
        iterations=3,
        min_branch_length=max(6, int(round(side / 220))),
    )

    if not np.any(skeleton):
        raise ValueError("Nem található vektorozható kézírás.")

    raw_paths = trace_skeleton_paths(skeleton)
    raw_paths = merge_stroke_fragments(
        raw_paths,
        max_gap=min(12.0, max(6.0, side / 20.0)),
        min_alignment=0.78,
    )
    paths = order_paths(raw_paths)

    height, width = processed.shape
    minimum_length = max(6.0, min(height, width) / 180.0)

    svg_paths = []
    total_points = 0

    for raw in paths:
        if _path_length(raw) < minimum_length:
            continue

        points = smooth_points(raw, quality=quality)
        if len(points) < 2:
            continue

        d = svg_path(points)
        if not d:
            continue

        total_points += len(points)
        svg_paths.append(
            f'<path d="{d}" fill="none" stroke="#000000" '
            'stroke-width="1" stroke-linecap="round" stroke-linejoin="round"/>'
        )

    if not svg_paths:
        raise ValueError("Nem sikerült rajzolható vonalpályát készíteni.")

    body = "\n  ".join(svg_paths)

    # Keep the page A4 and use the image pixels as the coordinate system.
    # This is easy to import into Inkscape and then scale/position to the
    # actual paper. No fills are used, so it remains pen-plotter friendly.
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
        'width="210mm" height="297mm" '
        f'viewBox="0 0 {width} {height}" '
        'preserveAspectRatio="xMidYMid meet">\n'
        '  <g id="idraw-handwriting" fill="none" '
        'stroke="#000000" stroke-linecap="round" stroke-linejoin="round">\n'
        f'  {body}\n'
        '  </g>\n'
        '</svg>\n'
    )

    # Return metadata for the UI without changing the SVG contract.
    return svg, {
        "paths": len(svg_paths),
        "points": total_points,
        "quality": int(quality),
    }


def validate_svg(svg_text):
    if not svg_text or "<svg" not in svg_text:
        raise ValueError("Az SVG dokumentum hiányzik.")
    if "<path " not in svg_text:
        raise ValueError("Az SVG nem tartalmaz rajzolható path elemeket.")

    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError as exc:
        raise ValueError("A generált SVG XML formátuma hibás.") from exc

    if root.tag.split("}")[-1] != "svg":
        raise ValueError("A dokumentum gyökéreleme nem SVG.")


def render_index(**values):
    defaults = {
        "original": None,
        "processed": None,
        "threshold": 160,
        "denoise": 3,
        "state_token": None,
        "svg_preview": None,
        "svg_download": None,
        "svg_stats": None,
        "status": None,
        "error": None,
    }
    defaults.update(values)
    return render_template("index.html", **defaults)


@app.route("/")
def index():
    cleanup_old_states()
    return render_index()


@app.route("/health")
def health():
    return {"status": "ok"}


@app.route("/process", methods=["POST"])
def process_image():
    try:
        threshold = max(
            80,
            min(240, int(request.form.get("threshold", 160))),
        )
    except (TypeError, ValueError):
        threshold = 160

    try:
        denoise = max(
            1,
            min(9, int(request.form.get("denoise", 3))),
        )
        if denoise % 2 == 0:
            denoise += 1
    except (TypeError, ValueError):
        denoise = 3

    try:
        upload = request.files.get("image")

        if not upload or not upload.filename:
            return render_index(
                threshold=threshold,
                denoise=denoise,
                error="Hiba: nem érkezett fájl.",
            )

        img = cv2.imdecode(
            np.frombuffer(upload.read(), np.uint8),
            cv2.IMREAD_COLOR,
        )

        if img is None:
            return render_index(
                threshold=threshold,
                denoise=denoise,
                error="A feltöltött kép nem olvasható.",
            )

        original_preview = encode_png(img)
        processed_img = process(img, threshold, denoise)
        processed_preview = encode_png(processed_img)

        if original_preview is None or processed_preview is None:
            raise ValueError("A kép kódolása nem sikerült.")

        token = new_token()
        save_state(token, img, processed_img)

        return render_index(
            original=original_preview,
            processed=processed_preview,
            threshold=threshold,
            denoise=denoise,
            state_token=token,
            status="A feldolgozás sikerült.",
        )

    except Exception as exc:
        app.logger.exception("PROCESS FAILED: %s", exc)
        return render_index(
            threshold=threshold,
            denoise=denoise,
            error=f"A feldolgozás nem sikerült: {exc}",
        ), 500


@app.route("/generate-svg", methods=["POST"])
def generate_svg():
    token = safe_token(request.form.get("state_token", ""))
    state = load_state(token)

    if state is None:
        return render_index(
            error=(
                "Az SVG generálása nem sikerült: a feldolgozott kép "
                "munkamenete lejárt vagy nem található."
            )
        ), 400

    paths, original_img, processed_img = state

    try:
        quality = max(1, min(5, int(request.form.get("quality", 3))))
        svg, svg_stats = handwriting_to_svg(processed_img, quality=quality)
        validate_svg(svg)

        paths["svg"].write_text(svg, encoding="utf-8")

        encoded_svg = base64.b64encode(
            svg.encode("utf-8")
        ).decode("ascii")

        return render_index(
            original=encode_png(original_img),
            processed=encode_png(processed_img),
            state_token=token,
            svg_preview="data:image/svg+xml;base64," + encoded_svg,
            svg_download=url_for(
                "download_svg",
                state_token=token,
            ),
            svg_stats=svg_stats,
            status="Az SVG sikeresen elkészült és el lett mentve.",
        )

    except Exception as exc:
        app.logger.exception("SVG GENERATION FAILED: %s", exc)
        return render_index(
            original=encode_png(original_img),
            processed=encode_png(processed_img),
            state_token=token,
            error=f"Az SVG generálása nem sikerült: {exc}",
        ), 500


@app.route("/download-svg")
def download_svg():
    token = safe_token(
        request.args.get("state_token", "")
    )
    paths = state_paths(token)

    if not paths or not paths["svg"].is_file():
        abort(404)

    svg = paths["svg"].read_text(encoding="utf-8")

    try:
        validate_svg(svg)
    except ValueError:
        abort(404)

    return Response(
        svg,
        mimetype="image/svg+xml",
        headers={
            "Content-Disposition": (
                'attachment; filename="idraw-vonalpalya.svg"'
            )
        },
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
        debug=False,
    )
