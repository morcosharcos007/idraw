import os
import secrets

from flask import Flask, Response, abort, render_template, request, url_for
import cv2, numpy as np, base64
from skimage.morphology import skeletonize

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit
_processed_states = {}

def encode(img):
    ok, data = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(data.tobytes()).decode() if ok else None

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
    """Keep the processed image server-side until the SVG action is submitted."""
    token = secrets.token_urlsafe(24)
    _processed_states[token] = {
        "original": original,
        "processed": processed,
    }
    # Bound memory use if the app is used repeatedly without restarting.
    while len(_processed_states) > 32:
        del _processed_states[next(iter(_processed_states))]
    return token

def process(img, threshold, denoise):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = int(denoise)
    if k < 1: k = 1
    if k % 2 == 0: k += 1
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    height, width = gray.shape
    smallest_side = min(height, width)

    # Measure ink against the nearby paper instead of using one global
    # grayscale cutoff. This keeps faint strokes on a shaded photograph while
    # rejecting low-contrast paper texture.
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
    contrast_ink = np.where(local_darkness >= darkness_cutoff, 255, 0).astype(
        np.uint8
    )

    # Blue/gray pen can be visually light after conversion to grayscale.
    # Rescue blue chroma only when it also has local contrast, so colored
    # paper texture is not promoted to handwriting.
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

    # Only use a very small kernel: it removes isolated paper specks while
    # avoiding the thick, filled shapes produced by larger closing kernels.
    speck_kernel = np.ones((2, 2), np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, speck_kernel, iterations=1)

    # Find the main handwriting components first. Small marks such as the dot
    # on an "i" are kept later when they sit close to one of these components,
    # while isolated paper dots are discarded.
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink, connectivity=8
    )
    image_scale = (height * width) / 1_000_000
    min_component_area = max(4, int(round(image_scale * (2 + int(denoise)))))
    main_component_area = max(
        min_component_area * 3, int(round(image_scale * (8 + int(denoise))))
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
        if (
            not touches_boundary
            and area >= main_component_area
        ):
            main_ink[labels == component_id] = 255

    # Components within this small gap are likely detached handwriting marks;
    # anything farther away is treated as isolated paper texture.
    connection_radius = max(4, min(18, int(round(smallest_side / 100))))
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

    # The component mask is white ink on black; invert it for the existing
    # preview format of black handwriting on a white page.
    return cv2.bitwise_not(cleaned_ink)

def _skeleton_paths(skeleton):
    """Trace and stitch skeleton branches into ordered pen-stroke paths.

    Skeletons naturally contain many degree-3/4 pixels at crossings and
    corners.  Treating every branch as an SVG path makes a single handwritten
    stroke look like a collection of short disconnected marks, so the branch
    graph is stitched back together using endpoint direction and distance.
    """
    points = [tuple(point) for point in np.argwhere(skeleton)]
    point_set = set(points)
    neighbors = {}

    for y, x in points:
        adjacent = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                candidate = (y + dy, x + dx)
                if candidate in point_set:
                    adjacent.append(candidate)
        neighbors[(y, x)] = adjacent

    degrees = {point: len(adjacent) for point, adjacent in neighbors.items()}
    nodes = [point for point, degree in degrees.items() if degree != 2]
    visited_edges = set()

    def edge_key(first, second):
        return (first, second) if first < second else (second, first)

    def trace(first, second):
        path = [first]
        previous, current = first, second
        visited_edges.add(edge_key(first, second))
        path.append(current)

        while current != first and degrees[current] == 2:
            next_points = [
                candidate
                for candidate in neighbors[current]
                if candidate != previous
                and edge_key(current, candidate) not in visited_edges
            ]
            if not next_points:
                break
            next_point = next_points[0]
            visited_edges.add(edge_key(current, next_point))
            previous, current = current, next_point
            path.append(current)

        return [(x, y) for y, x in path]

    paths = []
    for point in nodes:
        for neighbor in neighbors[point]:
            if edge_key(point, neighbor) not in visited_edges:
                paths.append(trace(point, neighbor))

    # A closed handwriting loop has no endpoint or junction. Walk any edges
    # that were not consumed above so loops remain separate SVG paths.
    for point in points:
        for neighbor in neighbors[point]:
            if edge_key(point, neighbor) not in visited_edges:
                loop = trace(point, neighbor)
                if len(loop) > 1 and loop[-1] == loop[0]:
                    loop.pop()
                paths.append(loop)

    if len(paths) < 2:
        return paths

    height, width = skeleton.shape
    # This is deliberately small: it bridges rasterisation gaps, but does not
    # join separate letters or the dot of an i to a distant stroke.
    gap_limit = max(2.0, min(8.0, min(height, width) / 180.0))

    def endpoint_info(path, endpoint):
        index = 0 if endpoint == 0 else -1
        other_index = 1 if endpoint == 0 else -2
        point = np.asarray(path[index], dtype=np.float32)
        direction = np.asarray(path[other_index], dtype=np.float32) - point
        length = float(np.linalg.norm(direction))
        if length == 0:
            return point, np.zeros(2, dtype=np.float32)
        return point, direction / length

    def join(first, first_endpoint, second, second_endpoint):
        first = list(first)
        second = list(second)
        if first_endpoint == 0:
            first.reverse()
        if second_endpoint == 1:
            second.reverse()
        if np.linalg.norm(
            np.asarray(first[-1], dtype=np.float32)
            - np.asarray(second[0], dtype=np.float32)
        ) < 1.5:
            return first + second[1:]
        # The SVG line between the two endpoints is the intended pen-down
        # bridge. An explicit midpoint gives simplification a stable sample.
        a = np.asarray(first[-1], dtype=np.float32)
        b = np.asarray(second[0], dtype=np.float32)
        midpoint = tuple(((a + b) / 2.0).tolist())
        return first + [midpoint] + second

    # Repeatedly take the best continuation.  The direction test prefers
    # nearly straight continuation through junction pixels and rejects most
    # accidental perpendicular joins.
    while True:
        best = None
        for first_index, first in enumerate(paths):
            if len(first) < 2 or first[0] == first[-1]:
                continue
            for second_index in range(first_index + 1, len(paths)):
                second = paths[second_index]
                if len(second) < 2 or second[0] == second[-1]:
                    continue
                for first_endpoint in (0, 1):
                    first_point, first_direction = endpoint_info(
                        first, first_endpoint
                    )
                    for second_endpoint in (0, 1):
                        second_point, second_direction = endpoint_info(
                            second, second_endpoint
                        )
                        distance = float(np.linalg.norm(first_point - second_point))
                        if distance > gap_limit:
                            continue
                        alignment = float(np.dot(first_direction, second_direction))
                        # At a shared junction use a little more tolerance for
                        # naturally curved writing; across a gap be stricter.
                        alignment_limit = 0.55 if distance <= 1.5 else 0.05
                        if alignment > alignment_limit:
                            continue
                        score = (1.0 + alignment) * 2.0 + distance / gap_limit
                        if best is None or score < best[0]:
                            best = (
                                score,
                                first_index,
                                first_endpoint,
                                second_index,
                                second_endpoint,
                            )

        if best is None:
            break
        _, first_index, first_endpoint, second_index, second_endpoint = best
        joined = join(
            paths[first_index],
            first_endpoint,
            paths[second_index],
            second_endpoint,
        )
        paths[first_index] = joined
        del paths[second_index]

    # Greedy nearest-end ordering reduces pen-up travel without changing the
    # geometry or merging separate strokes.
    ordered = []
    remaining = [list(path) for path in paths if len(path) > 1]
    if remaining:
        remaining.sort(key=lambda path: (path[0][1], path[0][0]))
        current = remaining.pop(0)
        ordered.append(current)
        while remaining:
            tail = np.asarray(ordered[-1][-1], dtype=np.float32)
            nearest_index, reverse = min(
                (
                    (
                        index,
                        np.linalg.norm(
                            tail
                            - np.asarray(
                                path[-1 if end == 1 else 0],
                                dtype=np.float32,
                            )
                        ),
                    )
                    for index, path in enumerate(remaining)
                    for end in (0, 1)
                ),
                key=lambda item: item[1],
            )
            nearest_path = remaining.pop(nearest_index)
            start_distance = np.linalg.norm(
                tail - np.asarray(nearest_path[0], dtype=np.float32)
            )
            end_distance = np.linalg.norm(
                tail - np.asarray(nearest_path[-1], dtype=np.float32)
            )
            if end_distance < start_distance:
                nearest_path.reverse()
            ordered.append(nearest_path)

    return ordered

def handwriting_to_svg(processed):
    """Convert the black-on-white preview into open centerline SVG paths."""
    ink = processed < 128
    skeleton = skeletonize(ink)
    raw_paths = _skeleton_paths(skeleton)
    height, width = processed.shape
    minimum_length = max(8.0, min(height, width) / 220.0)
    simplify_epsilon = max(0.7, min(height, width) / 1400.0)
    svg_paths = []

    def path_length(path):
        if len(path) < 2:
            return 0.0
        points = np.asarray(path, dtype=np.float32)
        return float(
            np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
        )

    def svg_path_data(points):
        """Use gentle cubic interpolation instead of a jagged L-only path."""
        if len(points) == 2:
            return (
                f"M{points[0][0]:.2f},{points[0][1]:.2f} "
                f"L{points[1][0]:.2f},{points[1][1]:.2f}"
            )

        commands = [f"M{points[0][0]:.2f},{points[0][1]:.2f}"]
        for index in range(len(points) - 1):
            previous = points[max(0, index - 1)]
            current = points[index]
            following = points[index + 1]
            after = points[min(len(points) - 1, index + 2)]
            control_one = current + (following - previous) / 6.0
            control_two = following - (after - current) / 6.0
            commands.append(
                "C"
                f"{control_one[0]:.2f},{control_one[1]:.2f} "
                f"{control_two[0]:.2f},{control_two[1]:.2f} "
                f"{following[0]:.2f},{following[1]:.2f}"
            )
        return " ".join(commands)

    for raw_path in raw_paths:
        if len(raw_path) < 2:
            continue
        if path_length(raw_path) < minimum_length:
            continue

        points = np.asarray(raw_path, dtype=np.float32).reshape(-1, 1, 2)
        simplified = cv2.approxPolyDP(
            points, simplify_epsilon, closed=False
        ).reshape(-1, 2)
        if len(simplified) < 2:
            continue
        coordinates = svg_path_data(simplified)
        svg_paths.append(
            f'<path d="{coordinates}" fill="none" stroke="black" '
            'stroke-linecap="round" stroke-linejoin="round" stroke-width="1"/>'
        )

    body = "\n  ".join(svg_paths)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="210mm" '
        f'height="297mm" viewBox="0 0 {width} {height}" '
        'preserveAspectRatio="xMidYMid meet">\n'
        f"  {body}\n"
        "</svg>\n"
    )

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
        threshold = max(80, min(240, int(request.form.get("threshold", 160))))
    except (TypeError, ValueError):
        threshold = 160
    try:
        denoise = max(1, min(9, int(request.form.get("denoise", 3))))
        if denoise % 2 == 0:
            denoise += 1
    except (TypeError, ValueError):
        denoise = 3

    # Must match <input name="image"> in templates/index.html.
    f = request.files.get("image")
    if not f or not f.filename:
        return render_index(
            threshold=threshold,
            denoise=denoise,
            error="Hiba: nem érkezett fájl.",
        )
    app.logger.info("FILE RECEIVED: %s", f.filename)

    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return render_index(
            threshold=threshold,
            denoise=denoise,
            error="A feltöltött kép nem olvasható.",
        )

    original = encode(img)
    processed_img = process(img, threshold, denoise)
    processed = encode(processed_img)
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
    state_token = request.form.get("state_token", "")
    state = _processed_states.get(state_token)

    # Render/Gunicorn can serve the second request from a different worker,
    # so an in-memory dictionary is not guaranteed to contain the token.
    # The browser also sends the processed image with the SVG request, which
    # gives us a reliable fallback.
    if not state:
        original = request.form.get("original", "")
        processed = request.form.get("processed", "")
        if processed:
            state = {
                "original": original,
                "processed": processed,
            }
        else:
            return render_index(
                error="Az SVG generálása nem sikerült: nincs elérhető feldolgozott kép.",
            )

    processed_img = decode_image_data(state["processed"], cv2.IMREAD_GRAYSCALE)
    if processed_img is None:
        return render_index(
            original=state["original"],
            processed=state["processed"],
            state_token=state_token,
            error="Az SVG generálása nem sikerült: a feldolgozott kép nem olvasható.",
        )

    try:
        svg = handwriting_to_svg(processed_img)
        if "<path " not in svg:
            raise ValueError("nem található rajzolható vonal")
    except Exception:
        return render_index(
            original=state["original"],
            processed=state["processed"],
            state_token=state_token,
            error="Az SVG generálása nem sikerült. Próbáld meg újra más beállításokkal.",
        )

    state["svg"] = svg
    encoded_svg = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return render_index(
        original=state["original"],
        processed=state["processed"],
        state_token=state_token,
        svg_preview="data:image/svg+xml;base64," + encoded_svg,
        svg_download=url_for("download_svg", state_token=state_token),
    )

@app.route("/download-svg")
def download_svg():
    state = _processed_states.get(request.args.get("state_token", ""))
    if not state or not state.get("svg"):
        abort(404)
    return Response(
        state["svg"],
        mimetype="image/svg+xml",
        headers={"Content-Disposition": "attachment; filename=idraw-vonalpalya.svg"},
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))