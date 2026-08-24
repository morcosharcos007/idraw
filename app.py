import os
import secrets
from pathlib import Path
import xml.etree.ElementTree as ET

from flask import Flask, Response, abort, render_template, request, url_for
import cv2, numpy as np, base64
from skimage.morphology import skeletonize

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

_processed_states = {}

# SVG files are stored on disk instead of only in worker memory.
# This makes SVG generation/download reliable when Gunicorn handles
# the generation and download requests in different workers.
GENERATED_DIR = Path(
    os.environ.get(
        "IDRAW_GENERATED_DIR",
        Path(os.environ.get("TMPDIR", "/tmp")) / "idraw-generated",
    )
)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def encode(img):
    ok, data = cv2.imencode(".png", img)
    return (
        "data:image/png;base64," + base64.b64encode(data.tobytes()).decode()
        if ok
        else None
    )


def decode_image_data(data_url, flags):
    """Decode an image data URL carried between the process and SVG submits."""
    if not data_url or "," not in data_url:
        return None
    try:
        encoded = data_url.split(",", 1)[1]
        raw = base64.b64decode(encoded, validate=True)
        return cv2.imdecode(np.frombuffer(raw, np.uint8), flags)
    except (ValueError, TypeError):
        return None


def remember_processed_state(original, processed):
    """Persist processed images so SVG generation works across workers."""
    token = secrets.token_urlsafe(24)
    state_dir = GENERATED_DIR / f"state-{token}"
    state_dir.mkdir(parents=True, exist_ok=False)

    try:
        original_img = decode_image_data(original, cv2.IMREAD_COLOR)
        processed_img = decode_image_data(processed, cv2.IMREAD_GRAYSCALE)

        if original_img is None or processed_img is None:
            raise ValueError("A feldolgozási állapot képe nem olvasható.")

        if not cv2.imwrite(str(state_dir / "original.png"), original_img):
            raise ValueError("Az eredeti kép mentése nem sikerült.")

        if not cv2.imwrite(str(state_dir / "processed.png"), processed_img):
            raise ValueError("A feldolgozott kép mentése nem sikerült.")

        _processed_states[token] = {
            "original": original,
            "processed": processed,
        }

        while len(_processed_states) > 32:
            del _processed_states[next(iter(_processed_states))]

        return token
    except Exception:
        import shutil
        shutil.rmtree(state_dir, ignore_errors=True)
        raise


def load_persisted_state(state_token):
    """Load a processed state from shared local storage."""
    safe_token = _safe_svg_token(state_token)
    if not safe_token:
        return None

    state_dir = GENERATED_DIR / f"state-{safe_token}"
    original_path = state_dir / "original.png"
    processed_path = state_dir / "processed.png"

    if not processed_path.is_file():
        return None

    processed_img = cv2.imread(
        str(processed_path),
        cv2.IMREAD_GRAYSCALE,
    )
    if processed_img is None:
        return None

    original_img = None
    if original_path.is_file():
        original_img = cv2.imread(
            str(original_path),
            cv2.IMREAD_COLOR,
        )

    return {
        "original": encode(original_img) if original_img is not None else None,
        "processed": encode(processed_img),
        "processed_img": processed_img,
    }


def process(img, threshold, denoise):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = int(denoise)
    if k < 1:
        k = 1
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
    contrast_ink = np.where(
        local_darkness >= darkness_cutoff, 255, 0
    ).astype(np.uint8)

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

    speck_kernel = np.ones((2, 2), np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, speck_kernel, iterations=1)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink, connectivity=8
    )
    image_scale = (height * width) / 1_000_000
    min_component_area = max(
        4, int(round(image_scale * (2 + int(denoise)))
    )
    )
    main_component_area = max(
        min_component_area * 3,
        int(round(image_scale * (8 + int(denoise)))),
    )
    main_ink = np.zeros_like(ink)

    for component_id in range(1, component_count):
        x, y, component_width, component_height, area = stats[component_id]
        touches_boundary = (
            x <= 0
            or y <= 0
            or x + component_width >= width
            or y + component_height >= height
        )
        if not touches_boundary and area >= main_component_area:
            main_ink[labels == component_id] = 255

    connection_radius = max(
        4, min(18, int(round(smallest_side / 100)))
    )
    connection_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (connection_radius * 2 + 1, connection_radius * 2 + 1),
    )
    nearby_main_ink = cv2.dilate(main_ink, connection_kernel)
    cleaned_ink = np.zeros_like(ink)

    for component_id in range(1, component_count):
        x, y, component_width, component_height, area = stats[component_id]
        touches_boundary = (
            x <= 0
            or y <= 0
            or x + component_width >= width
            or y + component_height >= height
        )
        if touches_boundary or area < min_component_area:
            continue

        component_mask = labels == component_id
        is_main = area >= main_component_area
        is_near_main = np.any(nearby_main_ink[component_mask] > 0)
        if not is_main and not is_near_main:
            continue

        cleaned_ink[labels == component_id] = 255

    return cv2.bitwise_not(cleaned_ink)



def _edge_key(a, b):
    return (a, b) if a < b else (b, a)


def _trace_skeleton_paths(skeleton):
    """Convert a skeleton into ordered paths without quadratic path stitching."""
    ys, xs = np.where(skeleton)
    if len(xs) == 0:
        return []

    points = set(zip(ys.tolist(), xs.tolist()))
    neighbors = {}

    for point in points:
        y, x = point
        adjacent = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                candidate = (y + dy, x + dx)
                if candidate in points:
                    adjacent.append(candidate)
        neighbors[point] = adjacent

    degree = {point: len(adjacent) for point, adjacent in neighbors.items()}
    visited_edges = set()
    paths = []

    def walk(start, next_point):
        path = [start, next_point]
        visited_edges.add(_edge_key(start, next_point))
        previous, current = start, next_point

        while degree[current] == 2:
            candidates = [
                candidate
                for candidate in neighbors[current]
                if candidate != previous
                and _edge_key(current, candidate) not in visited_edges
            ]
            if not candidates:
                break
            candidate = candidates[0]
            visited_edges.add(_edge_key(current, candidate))
            previous, current = current, candidate
            path.append(current)

        return [(x, y) for y, x in path]

    # Endpoints and junctions first.
    for node, node_degree in degree.items():
        if node_degree == 2:
            continue
        for neighbor in neighbors[node]:
            if _edge_key(node, neighbor) not in visited_edges:
                paths.append(walk(node, neighbor))

    # Closed loops have no endpoints/junctions.
    for point in points:
        for neighbor in neighbors[point]:
            if _edge_key(point, neighbor) not in visited_edges:
                paths.append(walk(point, neighbor))

    return paths


def _catmull_rom_svg(points):
    """Create smooth cubic SVG curves from a simplified polyline."""
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
        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0
        commands.append(
            "C"
            f"{c1[0]:.2f},{c1[1]:.2f} "
            f"{c2[0]:.2f},{c2[1]:.2f} "
            f"{p2[0]:.2f},{p2[1]:.2f}"
        )
    return " ".join(commands)


def handwriting_to_svg(processed):
    """Generate a bounded-resolution centerline SVG for the pen plotter."""
    if processed is None or processed.ndim != 2:
        raise ValueError("A feldolgozott kép érvénytelen.")

    original_height, original_width = processed.shape
    if original_height < 2 or original_width < 2:
        raise ValueError("A feldolgozott kép túl kicsi.")

    # Never skeletonize a multi-megapixel photo at full resolution.
    max_dimension = 1800
    scale = min(1.0, max_dimension / float(max(original_height, original_width)))

    if scale < 1.0:
        work_width = max(2, int(round(original_width * scale)))
        work_height = max(2, int(round(original_height * scale)))
        work = cv2.resize(processed, (work_width, work_height), interpolation=cv2.INTER_AREA)
    else:
        work = processed
        work_height, work_width = work.shape

    ink = work < 128
    ys, xs = np.where(ink)
    if len(xs) == 0:
        raise ValueError("A feldolgozott képen nem található rajzolható tinta.")

    # Crop to the actual ink before skeletonization. This is a major memory/time reduction.
    margin = max(2, int(round(min(work_height, work_width) * 0.01)))
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(work_width, int(xs.max()) + margin + 1)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(work_height, int(ys.max()) + margin + 1)
    ink = ink[y0:y1, x0:x1]

    skeleton = skeletonize(ink)
    raw_paths = _trace_skeleton_paths(skeleton)
    if not raw_paths:
        raise ValueError("A vonalpálya nem állítható elő.")

    simplify_epsilon = max(0.7, min(2.5, min(ink.shape) / 900.0))
    minimum_length = max(5.0, min(ink.shape) / 300.0)
    svg_paths = []

    for raw_path in raw_paths:
        if len(raw_path) < 2:
            continue

        points = np.asarray(raw_path, dtype=np.float32).reshape(-1, 1, 2)
        simplified = cv2.approxPolyDP(points, simplify_epsilon, closed=False).reshape(-1, 2)
        if len(simplified) < 2:
            continue

        length = float(np.linalg.norm(np.diff(simplified, axis=0), axis=1).sum())
        if length < minimum_length:
            continue

        # Restore original-image coordinates.
        simplified[:, 0] = (simplified[:, 0] + x0) / scale
        simplified[:, 1] = (simplified[:, 1] + y0) / scale
        d = _catmull_rom_svg(simplified)

        svg_paths.append(
            (
                float(simplified[0][1]),
                float(simplified[0][0]),
                f'<path d="{d}" fill="none" stroke="black" stroke-linecap="round" stroke-linejoin="round" stroke-width="1"/>'
            )
        )

    if not svg_paths:
        raise ValueError("A vonalpálya üres lett a zajszűrés után.")

    # Stable order only. The old all-pairs stitching algorithm is intentionally gone.
    svg_paths.sort(key=lambda item: (item[0], item[1]))
    body = "\n  ".join(item[2] for item in svg_paths)

    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="210mm" height="297mm" '
        f'viewBox="0 0 {original_width} {original_height}" '
        'preserveAspectRatio="xMidYMid meet">\n'
        f"  {body}\n"
        "</svg>\n"
    )
    validate_svg(svg)
    return svg

def validate_svg(svg_text):
    """Validate that the generated SVG is well-formed XML and drawable."""
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

    return True


def _safe_svg_token(state_token):
    if not state_token:
        return ""

    safe_token = "".join(
        character
        for character in state_token
        if character.isalnum() or character in "-_"
    )
    return safe_token


def save_svg_file(state_token, svg_text):
    """Save SVG to worker-independent local storage."""
    safe_token = _safe_svg_token(state_token)
    if not safe_token:
        raise ValueError("Érvénytelen SVG azonosító.")

    path = GENERATED_DIR / f"{safe_token}.svg"
    path.write_text(svg_text, encoding="utf-8")
    return path


def load_svg_file(state_token):
    """Load a previously generated SVG from local storage."""
    safe_token = _safe_svg_token(state_token)
    if not safe_token:
        return None

    path = GENERATED_DIR / f"{safe_token}.svg"
    if not path.is_file():
        return None

    return path.read_text(encoding="utf-8")


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
    return render_index()


@app.route("/health")
def health():
    return {"status": "ok"}


@app.route("/process", methods=["POST"])
def process_image():
    app.logger.info("/process POST RECEIVED")

    try:
        threshold = max(
            80, min(240, int(request.form.get("threshold", 160)))
        )
    except (TypeError, ValueError):
        threshold = 160

    try:
        denoise = max(
            1, min(9, int(request.form.get("denoise", 3)))
        )
        if denoise % 2 == 0:
            denoise += 1
    except (TypeError, ValueError):
        denoise = 3

    f = request.files.get("image")
    if not f or not f.filename:
        return render_index(
            threshold=threshold,
            denoise=denoise,
            error="Hiba: nem érkezett fájl.",
        )

    app.logger.info("FILE RECEIVED: %s", f.filename)

    img = cv2.imdecode(
        np.frombuffer(f.read(), np.uint8),
        cv2.IMREAD_COLOR,
    )

    if img is None:
        return render_index(
            threshold=threshold,
            denoise=denoise,
            error="A feltöltött kép nem olvasható.",
        )

    original = encode(img)
    processed_img = process(img, threshold, denoise)
    processed = encode(processed_img)

    if original is None or processed is None:
        return render_index(
            threshold=threshold,
            denoise=denoise,
            error="A kép kódolása nem sikerült.",
        )

    state_token = remember_processed_state(original, processed)

    return render_index(
        original=original,
        processed=processed,
        threshold=threshold,
        denoise=denoise,
        state_token=state_token,
        status="A feldolgozás sikerült.",
    )


@app.route("/generate-svg", methods=["POST"])
def generate_svg():
    state_token = request.form.get("state_token", "").strip()

    state = _processed_states.get(state_token)
    if state is None:
        state = load_persisted_state(state_token)

    if state is None:
        return render_index(
            error=(
                "Az SVG generálása nem sikerült: "
                "a feldolgozási munkamenet nem található. "
                "Futtasd újra a FELDOLGOZÁST."
            ),
        )

    processed_img = state.get("processed_img")
    if processed_img is None:
        processed_img = decode_image_data(
            state.get("processed", ""),
            cv2.IMREAD_GRAYSCALE,
        )

    if processed_img is None:
        return render_index(
            original=state.get("original"),
            processed=state.get("processed"),
            state_token=state_token,
            error=(
                "Az SVG generálása nem sikerült: "
                "a feldolgozott kép nem olvasható."
            ),
        )

    try:
        svg = handwriting_to_svg(processed_img)

        if "<path " not in svg:
            raise ValueError("nem található rajzolható vonal")

        validate_svg(svg)
        save_svg_file(state_token, svg)

        _processed_states[state_token] = {
            "original": state.get("original"),
            "processed": state.get("processed"),
            "svg": svg,
        }

    except Exception as exc:
        app.logger.exception("SVG GENERATION/SAVE FAILED: %s", exc)
        return render_index(
            original=state.get("original"),
            processed=state.get("processed"),
            state_token=state_token,
            error=f"Az SVG mentése nem sikerült: {exc}",
        )

    encoded_svg = base64.b64encode(
        svg.encode("utf-8")
    ).decode("ascii")

    return render_index(
        original=state.get("original"),
        processed=state.get("processed"),
        state_token=state_token,
        svg_preview="data:image/svg+xml;base64," + encoded_svg,
        svg_download=url_for(
            "download_svg",
            state_token=state_token,
        ),
        status="Az SVG sikeresen elkészült és el lett mentve.",
    )


@app.route("/download-svg")
def download_svg():
    state_token = request.args.get("state_token", "")

    # Persistent file first: this works even when another Gunicorn
    # worker handles the download request.
    svg = load_svg_file(state_token)

    # Backwards-compatible fallback for an SVG generated in memory.
    if svg is None:
        state = _processed_states.get(state_token)
        if state:
            svg = state.get("svg")

    if not svg:
        abort(404)

    try:
        validate_svg(svg)
    except ValueError:
        abort(404)

    return Response(
        svg,
        mimetype="image/svg+xml",
        headers={
            "Content-Disposition": (
                "attachment; filename=idraw-vonalpalya.svg"
            ),
            "Cache-Control": "no-store",
        },
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
    )
