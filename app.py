
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

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# IMPORTANT:
# The process request and the SVG request may be handled by different
# Gunicorn workers. Therefore the intermediate images are stored on disk,
# not in a Python dictionary.
STATE_DIR = Path(
    os.environ.get(
        "IDRAW_STATE_DIR",
        str(Path(os.environ.get("TMPDIR", "/tmp")) / "idraw-state"),
    )
)
STATE_DIR.mkdir(parents=True, exist_ok=True)

MAX_VECTOR_DIMENSION = 1800


def safe_token(value):
    if not value:
        return ""
    value = re.sub(r"[^A-Za-z0-9_-]", "", value)
    return value[:80]


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
    data = path.read_bytes()
    return cv2.imdecode(np.frombuffer(data, np.uint8), flags)


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
    paths = state_paths(token)
    if not paths:
        raise ValueError("Érvénytelen feldolgozási azonosító.")
    paths["folder"].mkdir(parents=True, exist_ok=True)
    save_png(paths["original"], original)
    save_png(paths["processed"], processed)
    return paths


def load_state(token):
    paths = state_paths(token)
    if not paths or not paths["processed"].is_file():
        return None
    original = load_png(paths["original"], cv2.IMREAD_COLOR)
    processed = load_png(paths["processed"], cv2.IMREAD_GRAYSCALE)
    if original is None or processed is None:
        return None
    return paths, original, processed


def cleanup_old_states(max_age_hours=6):
    cutoff = time.time() - max_age_hours * 3600
    if not STATE_DIR.exists():
        return
    for folder in STATE_DIR.iterdir():
        try:
            if folder.is_dir() and folder.stat().st_mtime < cutoff:
                for child in folder.iterdir():
                    child.unlink(missing_ok=True)
                folder.rmdir()
        except OSError:
            pass


def process(img, threshold, denoise):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Very large phone photos are unnecessary for handwriting extraction.
    # Resize once here so later skeleton/vector operations stay predictable.
    h, w = gray.shape
    scale = min(1.0, MAX_VECTOR_DIMENSION / float(max(h, w)))
    if scale < 1.0:
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    k = max(1, min(9, int(denoise)))
    if k % 2 == 0:
        k += 1
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    height, width = gray.shape
    smallest_side = min(height, width)

    block_size = max(15, int(round(smallest_side / 35)))
    block_size = min(block_size | 1, 81)

    adaptive_c = int(np.interp(int(threshold), [80, 240], [14, 3]))
    adaptive_ink = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        adaptive_c,
    )

    paper = cv2.GaussianBlur(
        gray, (0, 0), sigmaX=max(8.0, min(32.0, smallest_side / 28))
    )
    local_darkness = paper.astype(np.int16) - gray.astype(np.int16)
    darkness_cutoff = int(np.interp(int(threshold), [80, 240], [24, 7]))
    contrast_ink = np.where(local_darkness >= darkness_cutoff, 255, 0).astype(np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    blue_pen = (
        (hue >= 85)
        & (hue <= 140)
        & (saturation >= 18)
        & (value <= 252)
        & (local_darkness >= max(3, darkness_cutoff // 2))
        & (adaptive_ink > 0)
    )

    ink = np.where((contrast_ink > 0) | blue_pen, 255, 0).astype(np.uint8)

    # Remove isolated dust without erasing the thin handwriting strokes.
    speck_kernel = np.ones((2, 2), np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, speck_kernel, iterations=1)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink, connectivity=8
    )
    image_scale = (height * width) / 1_000_000.0
    min_component_area = max(4, int(round(image_scale * (2 + k))))
    main_component_area = max(
        min_component_area * 3,
        int(round(image_scale * (8 + k))),
    )

    main_ink = np.zeros_like(ink)
    for component_id in range(1, component_count):
        x, y, cw, ch, area = stats[component_id]
        touches_boundary = (
            x <= 0 or y <= 0 or x + cw >= width or y + ch >= height
        )
        if not touches_boundary and area >= main_component_area:
            main_ink[labels == component_id] = 255

    connection_radius = max(4, min(18, int(round(smallest_side / 100))))
    connection_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (connection_radius * 2 + 1, connection_radius * 2 + 1),
    )
    nearby_main_ink = cv2.dilate(main_ink, connection_kernel)

    cleaned_ink = np.zeros_like(ink)
    for component_id in range(1, component_count):
        x, y, cw, ch, area = stats[component_id]
        touches_boundary = (
            x <= 0 or y <= 0 or x + cw >= width or y + ch >= height
        )
        if touches_boundary or area < min_component_area:
            continue

        component_mask = labels == component_id
        is_main = area >= main_component_area
        is_near_main = bool(np.any(nearby_main_ink[component_mask] > 0))
        if is_main or is_near_main:
            cleaned_ink[component_mask] = 255

    return cv2.bitwise_not(cleaned_ink)


def _neighbors(point, point_set):
    y, x = point
    result = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not dx and not dy:
                continue
            candidate = (y + dy, x + dx)
            if candidate in point_set:
                result.append(candidate)
    return result


def _trace_skeleton_paths(skeleton):
    """Trace a skeleton graph into continuous pen paths.

    We deliberately do not connect distant paths: a pen lift must remain
    a pen lift. This prevents the plotter from drawing artificial diagonals.
    """
    points = [tuple(p) for p in np.argwhere(skeleton)]
    if not points:
        return []

    point_set = set(points)
    neighbors = {p: _neighbors(p, point_set) for p in points}
    degree = {p: len(n) for p, n in neighbors.items()}
    nodes = [p for p, d in degree.items() if d != 2]
    visited = set()

    def edge_key(a, b):
        return (a, b) if a < b else (b, a)

    def trace(start, nxt):
        path = [start]
        previous, current = start, nxt
        visited.add(edge_key(start, nxt))
        path.append(current)

        while degree.get(current, 0) == 2:
            candidates = [
                p for p in neighbors[current]
                if p != previous and edge_key(current, p) not in visited
            ]
            if not candidates:
                break
            following = candidates[0]
            visited.add(edge_key(current, following))
            previous, current = current, following
            path.append(current)

        return [(x, y) for y, x in path]

    paths = []
    for node in nodes:
        for nxt in neighbors[node]:
            if edge_key(node, nxt) not in visited:
                paths.append(trace(node, nxt))

    # Closed loops have no degree-1/3 nodes. Pick an arbitrary point.
    for point in points:
        for nxt in neighbors[point]:
            if edge_key(point, nxt) not in visited:
                loop = trace(point, nxt)
                if len(loop) > 2 and loop[-1] == loop[0]:
                    loop.pop()
                paths.append(loop)

    return [p for p in paths if len(p) >= 2]


def _order_paths(paths):
    """Order independent strokes so the pen travels as little as possible."""
    remaining = [list(p) for p in paths if len(p) >= 2]
    if not remaining:
        return []

    remaining.sort(key=lambda p: (p[0][1], p[0][0]))
    ordered = [remaining.pop(0)]

    while remaining:
        tail = np.asarray(ordered[-1][-1], dtype=np.float32)

        best_index = 0
        best_distance = float("inf")
        reverse = False

        for index, path in enumerate(remaining):
            start = np.asarray(path[0], dtype=np.float32)
            end = np.asarray(path[-1], dtype=np.float32)
            d_start = float(np.linalg.norm(tail - start))
            d_end = float(np.linalg.norm(tail - end))
            if d_start < best_distance:
                best_distance = d_start
                best_index = index
                reverse = False
            if d_end < best_distance:
                best_distance = d_end
                best_index = index
                reverse = True

        path = remaining.pop(best_index)
        if reverse:
            path.reverse()
        ordered.append(path)

    return ordered


def _smooth_points(points, epsilon):
    if len(points) <= 3:
        return points

    array = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(array, epsilon, closed=False).reshape(-1, 2)

    # Never let simplification turn a handwriting stroke into a 2-point line
    # unless the source stroke itself is essentially straight.
    if len(simplified) < 3 and len(points) >= 5:
        return np.asarray(points, dtype=np.float32)[::2]
    return simplified


def _svg_path_data(points, tension=0.75):
    points = [np.asarray(p, dtype=np.float32) for p in points]
    if len(points) == 2:
        return (
            f"M{points[0][0]:.2f},{points[0][1]:.2f} "
            f"L{points[1][0]:.2f},{points[1][1]:.2f}"
        )

    commands = [f"M{points[0][0]:.2f},{points[0][1]:.2f}"]
    for i in range(len(points) - 1):
        p0 = points[max(0, i - 1)]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[min(len(points) - 1, i + 2)]

        c1 = p1 + (p2 - p0) * (tension / 6.0)
        c2 = p2 - (p3 - p1) * (tension / 6.0)

        commands.append(
            "C"
            f"{c1[0]:.2f},{c1[1]:.2f} "
            f"{c2[0]:.2f},{c2[1]:.2f} "
            f"{p2[0]:.2f},{p2[1]:.2f}"
        )

    return " ".join(commands)


def handwriting_to_svg(processed):
    """Convert the processed image into smooth, open centerline SVG paths."""
    # The processed preview is black ink on white.
    ink = (processed < 128).astype(np.uint8) * 255

    # A second small cleanup before skeletonization prevents paper speckles
    # from becoming thousands of tiny graph nodes.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    cleaned = np.zeros_like(ink)
    area_limit = max(6, int((ink.shape[0] * ink.shape[1]) / 1_500_000))
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= area_limit:
            cleaned[labels == i] = 255

    skeleton = skeletonize(cleaned > 0)
    if int(np.count_nonzero(skeleton)) == 0:
        raise ValueError("Nem található vektorozható kézírás.")

    raw_paths = _trace_skeleton_paths(skeleton)
    raw_paths = _order_paths(raw_paths)

    height, width = processed.shape
    minimum_length = max(10.0, min(height, width) / 180.0)
    epsilon = max(0.55, min(1.25, min(height, width) / 1400.0))

    svg_paths = []
    for raw in raw_paths:
        if len(raw) < 2:
            continue

        raw_array = np.asarray(raw, dtype=np.float32)
        length = float(np.linalg.norm(np.diff(raw_array, axis=0), axis=1).sum())
        if length < minimum_length:
            continue

        points = _smooth_points(raw, epsilon)
        if len(points) < 2:
            continue

        d = _svg_path_data(points)
        svg_paths.append(
            f'<path d="{d}" fill="none" stroke="black" '
            'stroke-width="1" stroke-linecap="round" stroke-linejoin="round"/>'
        )

    if not svg_paths:
        raise ValueError("Nem sikerült rajzolható vonalpályát készíteni.")

    body = "\n  ".join(svg_paths)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="210mm" height="297mm" viewBox="0 0 {width} {height}" '
        'preserveAspectRatio="xMidYMid meet">\n'
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

        raw = upload.read()
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
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
            error="Az SVG generálása nem sikerült: a feldolgozott kép munkamenete lejárt vagy nem található."
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
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
