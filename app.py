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
from scipy.interpolate import splprep, splev
from skimage.morphology import medial_axis

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
    """Create a clean binary ink mask while preserving thin handwriting strokes."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    scale = min(1.0, MAX_VECTOR_DIMENSION / float(max(h, w)))
    if scale < 1.0:
        size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
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
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block, adaptive_c
    )

    paper = cv2.GaussianBlur(
        gray, (0, 0), sigmaX=max(8.0, min(32.0, side / 28))
    )
    darkness = paper.astype(np.int16) - gray.astype(np.int16)
    cutoff = int(np.interp(int(threshold), [80, 240], [24, 7]))
    contrast = np.where(darkness >= cutoff, 255, 0).astype(np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    blue = (
        (hue >= 85) & (hue <= 140) & (sat >= 18) & (val <= 252)
        & (darkness >= max(3, cutoff // 2)) & (adaptive > 0)
    )

    ink = np.where((contrast > 0) | blue, 255, 0).astype(np.uint8)

    # Do not aggressively erode/open the handwriting: that was a source of broken strokes.
    if k >= 5:
        ink = cv2.morphologyEx(
            ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1
        )

    gap = max(1, min(3, int(round(side / 800))))
    if gap > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (gap * 2 + 1, gap * 2 + 1)
        )
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    min_area = max(3, int(round((height * width) / 2_000_000)))
    cleaned = np.zeros_like(ink)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    return cv2.bitwise_not(cleaned)


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


def _trace_edges(skeleton):
    """Convert a one-pixel centerline graph into node-to-node edge polylines."""
    points = [tuple(p) for p in np.argwhere(skeleton)]
    if not points:
        return []

    point_set = set(points)
    nb = {p: _neighbors(p, point_set) for p in points}
    degree = {p: len(v) for p, v in nb.items()}
    nodes = {p for p, d in degree.items() if d != 2}

    if not nodes:
        start = points[0]
        path = [start]
        prev = None
        cur = start
        while True:
            choices = [q for q in nb[cur] if q != prev]
            if not choices:
                break
            nxt = choices[0]
            if nxt == start and len(path) > 2:
                break
            path.append(nxt)
            prev, cur = cur, nxt
            if len(path) > len(points) + 2:
                break
        return [path] if len(path) > 2 else []

    edges = []
    seen = set()
    for node in nodes:
        for nxt in nb[node]:
            key = _edge_key(node, nxt)
            if key in seen:
                continue

            path = [node, nxt]
            seen.add(key)
            prev, cur = node, nxt

            while cur not in nodes:
                choices = [q for q in nb[cur] if q != prev]
                if not choices:
                    break
                q = choices[0]
                key = _edge_key(cur, q)
                if key in seen:
                    break
                seen.add(key)
                path.append(q)
                prev, cur = cur, q

            if len(path) >= 2:
                edges.append(path)

    return edges


def _edge_direction(edge, from_start):
    """Estimate the local tangent at an edge endpoint in image coordinates."""
    if from_start:
        a, b = edge[0], edge[min(4, len(edge) - 1)]
    else:
        a, b = edge[-1], edge[max(0, len(edge) - 5)]
    v = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.zeros(2, dtype=np.float32)


def trace_centerline_paths(skeleton):
    """
    Turn the medial-axis skeleton into long pen strokes.

    At a junction, continuation is selected by the smallest turning angle.
    """
    edges = _trace_edges(skeleton)
    if not edges:
        return []

    incident = {}
    for i, edge in enumerate(edges):
        incident.setdefault(edge[0], []).append((i, True))
        incident.setdefault(edge[-1], []).append((i, False))

    paired = {}
    for node, items in incident.items():
        remaining = list(items)
        while len(remaining) >= 2:
            best = None
            best_cost = float("inf")
            for a in range(len(remaining)):
                for b in range(a + 1, len(remaining)):
                    ia, sa = remaining[a]
                    ib, sb = remaining[b]
                    va = _edge_direction(edges[ia], sa)
                    vb = _edge_direction(edges[ib], sb)
                    dot = float(np.clip(np.dot(va, vb), -1.0, 1.0))
                    cost = abs(dot + 1.0)
                    if cost < best_cost:
                        best_cost = cost
                        best = (a, b)
            if best is None:
                break
            a, b = best
            ia, sa = remaining[a]
            ib, sb = remaining[b]
            paired[(ia, sa)] = (ib, sb)
            paired[(ib, sb)] = (ia, sa)
            for pos in sorted((a, b), reverse=True):
                remaining.pop(pos)

    used = set()
    paths = []

    def extend(index, side):
        edge = edges[index]
        coords = list(reversed(edge)) if side else list(edge)
        used.add(index)
        cur_index, cur_side = index, side

        while True:
            nxt = paired.get((cur_index, cur_side))
            if nxt is None or nxt[0] in used:
                break
            ni, ns = nxt
            next_coords = list(reversed(edges[ni])) if ns else list(edges[ni])
            if coords[-1] == next_coords[0]:
                coords.extend(next_coords[1:])
            else:
                coords.extend(next_coords)
            used.add(ni)
            cur_index, cur_side = ni, ns

        return [(x, y) for y, x in coords]

    for i, edge in enumerate(edges):
        if i not in used and len(incident.get(edge[0], [])) == 1:
            paths.append(extend(i, False))
    for i in range(len(edges)):
        if i not in used:
            paths.append(extend(i, False))

    return [p for p in paths if len(p) >= 2]


def _path_length(points):
    arr = np.asarray(points, dtype=np.float32)
    if len(arr) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())


def _resample(points, spacing=2.0):
    """Resample a polyline at near-uniform arc-length spacing."""
    arr = np.asarray(points, dtype=np.float32)
    if len(arr) < 2:
        return arr

    seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    dist = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(dist[-1])
    if total < 1e-6:
        return arr

    count = max(3, int(round(total / spacing)) + 1)
    target = np.linspace(0.0, total, count)
    x = np.interp(target, dist, arr[:, 0])
    y = np.interp(target, dist, arr[:, 1])
    return np.column_stack((x, y)).astype(np.float32)


def smooth_centerline(points, image_size):
    """
    Fit a smoothing cubic B-spline to the centerline.

    This follows the important UUNA TEK/Inkscape principle for signatures:
    centerline tracing + a smooth B-spline-like path rather than a jagged
    polygonal outline.
    """
    arr = _resample(points, spacing=max(1.2, min(image_size) / 900.0))
    if len(arr) < 4:
        return arr

    arr = cv2.GaussianBlur(arr.reshape(-1, 1, 2), (0, 0), 0.8).reshape(-1, 2)
    length = _path_length(arr)
    smoothness = max(0.5, min(length * 0.012, length * 0.045))

    try:
        k = min(3, len(arr) - 1)
        tck, _ = splprep(
            [arr[:, 0], arr[:, 1]], s=smoothness, k=k, per=False
        )
        samples = max(12, int(length / max(1.0, min(image_size) / 500.0)))
        u = np.linspace(0.0, 1.0, samples)
        x, y = splev(u, tck)
        return np.column_stack((x, y)).astype(np.float32)
    except (ValueError, TypeError):
        return arr


def _simplify_for_svg(points, min_spacing=0.8):
    """Drop redundant samples without turning the curve into line segments."""
    arr = np.asarray(points, dtype=np.float32)
    if len(arr) <= 2:
        return arr
    keep = [arr[0]]
    for p in arr[1:-1]:
        if float(np.linalg.norm(p - keep[-1])) >= min_spacing:
            keep.append(p)
    keep.append(arr[-1])
    return np.asarray(keep, dtype=np.float32)


def svg_path(points):
    """Emit the already-smoothed centerline as cubic Bezier segments."""
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
        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0
        commands.append(
            f"C{c1[0]:.2f},{c1[1]:.2f} "
            f"{c2[0]:.2f},{c2[1]:.2f} "
            f"{p2[0]:.2f},{p2[1]:.2f}"
        )
    return " ".join(commands)


def _order_paths(paths):
    """Minimize pen-up travel while keeping each stroke direction intact."""
    remaining = [list(p) for p in paths if len(p) >= 2]
    if not remaining:
        return []
    remaining.sort(key=lambda p: (p[0][1], p[0][0]))
    ordered = [remaining.pop(0)]

    while remaining:
        tail = np.asarray(ordered[-1][-1], dtype=np.float32)
        best_i, best_d, reverse = 0, float("inf"), False
        for i, path in enumerate(remaining):
            a = np.asarray(path[0], dtype=np.float32)
            b = np.asarray(path[-1], dtype=np.float32)
            da = float(np.linalg.norm(tail - a))
            db = float(np.linalg.norm(tail - b))
            if da < best_d:
                best_i, best_d, reverse = i, da, False
            if db < best_d:
                best_i, best_d, reverse = i, db, True
        path = remaining.pop(best_i)
        if reverse:
            path.reverse()
        ordered.append(path)

    return ordered


def handwriting_to_svg(processed):
    ink = processed < 128
    if not np.any(ink):
        raise ValueError("Nem található vektorozható kézírás.")

    # Key change: medial-axis centerline instead of tracing the raster edge.
    skeleton, _ = medial_axis(ink, return_distance=True)
    if not np.any(skeleton):
        raise ValueError("Nem sikerült középvonalat találni.")

    paths = trace_centerline_paths(skeleton)
    if not paths:
        raise ValueError("Nem sikerült rajzolható középvonalat készíteni.")

    h, w = processed.shape
    min_length = max(10.0, min(h, w) / 120.0)
    svg_paths = []

    for raw in paths:
        if _path_length(raw) < min_length:
            continue
        smooth = smooth_centerline(raw, min(h, w))
        smooth = _simplify_for_svg(
            smooth, min_spacing=max(0.65, min(h, w) / 2200.0)
        )
        if len(smooth) < 2:
            continue
        svg_paths.append(
            f'<path d="{svg_path(smooth)}" fill="none" stroke="black" '
            'stroke-width="1" stroke-linecap="round" stroke-linejoin="round"/>'
        )

    if not svg_paths:
        raise ValueError("Nem sikerült rajzolható vonalpályát készíteni.")

    body = "\n  ".join(svg_paths)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="210mm" height="297mm" '
        f'viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet">\n'
        f"  {body}\n"
        "</svg>\n"
    )


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
        threshold = max(80, min(240, int(request.form.get("threshold", 160))))
    except (TypeError, ValueError):
        threshold = 160

    try:
        denoise = max(1, min(9, int(request.form.get("denoise", 3))))
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
            np.frombuffer(upload.read(), np.uint8), cv2.IMREAD_COLOR
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
        svg = handwriting_to_svg(processed_img)
        validate_svg(svg)
        paths["svg"].write_text(svg, encoding="utf-8")
        encoded_svg = base64.b64encode(svg.encode("utf-8")).decode("ascii")
        return render_index(
            original=encode_png(original_img),
            processed=encode_png(processed_img),
            state_token=token,
            svg_preview="data:image/svg+xml;base64," + encoded_svg,
            svg_download=url_for("download_svg", state_token=token),
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
    token = safe_token(request.args.get("state_token", ""))
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
            "Content-Disposition": 'attachment; filename="idraw-vonalpalya.svg"'
        },
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
        debug=False,
    )
