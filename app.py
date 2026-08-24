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


def trace_skeleton_paths(skeleton):
    """
    Build long handwriting strokes from the skeleton graph.

    The old method stopped at every graph junction. That created many short
    paths, so the plotter would repeatedly lift/lower the pen and the SVG
    looked segmented. Here junction edges are paired by direction so a stroke
    continues through a natural crossing/junction whenever possible.
    """
    points = [tuple(p) for p in np.argwhere(skeleton)]
    if not points:
        return []

    point_set = set(points)
    nb = {p: neighbors(p, point_set) for p in points}
    degree = {p: len(v) for p, v in nb.items()}
    nodes = {p for p, d in degree.items() if d != 2}

    # Closed loop with no endpoints/junctions.
    if not nodes:
        start = points[0]
        nxt = nb[start][0]
        path = [start]
        prev, cur = start, nxt
        seen = {edge_key(start, nxt)}

        while True:
            path.append(cur)
            choices = [
                q for q in nb[cur]
                if q != prev and edge_key(cur, q) not in seen
            ]
            if not choices:
                break
            q = choices[0]
            seen.add(edge_key(cur, q))
            prev, cur = cur, q
            if cur == start:
                break

        return [[(x, y) for y, x in path]]

    # Build graph edges between meaningful nodes.
    edges = []
    seen = set()

    for node in nodes:
        for nxt in nb[node]:
            if edge_key(node, nxt) in seen:
                continue

            path = [node]
            prev, cur = node, nxt
            seen.add(edge_key(node, nxt))
            path.append(cur)

            while cur not in nodes:
                choices = [q for q in nb[cur] if q != prev]
                if not choices:
                    break

                q = choices[0]
                key = edge_key(cur, q)
                if key in seen:
                    break

                seen.add(key)
                prev, cur = cur, q
                path.append(cur)

            if len(path) >= 2:
                edges.append(path)

    incident = {}
    for i, edge in enumerate(edges):
        incident.setdefault(edge[0], []).append((i, True))
        incident.setdefault(edge[-1], []).append((i, False))

    # At each junction pair the edges whose directions are most opposite.
    # That is the natural continuation of a handwritten stroke.
    paired = {}

    for node, items in incident.items():
        remaining = list(items)

        while len(remaining) >= 2:
            best = None
            best_score = float("inf")
            node_xy = np.asarray(node, dtype=np.float32)

            for a in range(len(remaining)):
                for b in range(a + 1, len(remaining)):
                    ia, side_a = remaining[a]
                    ib, side_b = remaining[b]

                    va = (
                        np.asarray(edges[ia][1 if side_a else -2], dtype=np.float32)
                        - node_xy
                    )
                    vb = (
                        np.asarray(edges[ib][-2 if side_b else 1], dtype=np.float32)
                        - node_xy
                    )

                    na = float(np.linalg.norm(va))
                    nbv = float(np.linalg.norm(vb))

                    if na < 1e-6 or nbv < 1e-6:
                        score = 999.0
                    else:
                        cosine = float(np.dot(va, vb) / (na * nbv))
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

    def extend(edge_index, side):
        edge = edges[edge_index]
        coords = list(reversed(edge)) if side else list(edge)
        used.add(edge_index)

        current_index = edge_index
        current_side = side

        while True:
            nxt = paired.get((current_index, current_side))
            if nxt is None or nxt[0] in used:
                break

            next_index, next_side = nxt
            next_edge = edges[next_index]
            next_coords = (
                list(reversed(next_edge))
                if next_side
                else list(next_edge)
            )

            if coords[-1] == next_coords[0]:
                coords.extend(next_coords[1:])
            else:
                coords.extend(next_coords)

            used.add(next_index)
            current_index = next_index
            current_side = next_side

        return [(x, y) for y, x in coords]

    # Start at real endpoints first.
    for i, edge in enumerate(edges):
        if i not in used and len(incident.get(edge[0], [])) == 1:
            paths.append(extend(i, False))

    # Consume any closed/branch leftovers.
    for i in range(len(edges)):
        if i not in used:
            paths.append(extend(i, False))

    return [p for p in paths if len(p) >= 2]


def order_paths(paths):
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


def smooth_points(points, epsilon):
    if len(points) <= 3:
        return np.asarray(points, dtype=np.float32)

    arr = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(arr, epsilon, closed=False).reshape(-1, 2)

    if len(simplified) < 3 and len(points) >= 5:
        step = max(1, len(points) // 12)
        return np.asarray(points, dtype=np.float32)[::step]

    return simplified


def svg_path(points, tension=0.90):
    p = [np.asarray(x, dtype=np.float32) for x in points]

    if len(p) == 2:
        return (
            f"M{p[0][0]:.2f},{p[0][1]:.2f} "
            f"L{p[1][0]:.2f},{p[1][1]:.2f}"
        )

    commands = [
        f"M{p[0][0]:.2f},{p[0][1]:.2f}"
    ]

    for i in range(len(p) - 1):
        p0 = p[max(0, i - 1)]
        p1 = p[i]
        p2 = p[i + 1]
        p3 = p[min(len(p) - 1, i + 2)]

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
    ink = (processed < 128).astype(np.uint8) * 255

    # Remove tiny components.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    cleaned = np.zeros_like(ink)
    area_limit = max(
        6,
        int((ink.shape[0] * ink.shape[1]) / 1_500_000),
    )

    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= area_limit:
            cleaned[labels == i] = 255

    # IMPORTANT: smooth the bitmap before skeletonization.
    # Skeletonizing the raw pixel boundary creates the little zig-zags seen
    # in the previous preview.
    close_size = max(
        3,
        min(
            7,
            int(round(min(processed.shape) / 450)) * 2 + 3,
        ),
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_size, close_size),
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1,
    )
    cleaned = cv2.GaussianBlur(cleaned, (0, 0), 0.65)
    cleaned = np.where(cleaned >= 120, 255, 0).astype(np.uint8)

    skeleton = skeletonize(cleaned > 0)

    if not np.any(skeleton):
        raise ValueError("Nem található vektorozható kézírás.")

    paths = order_paths(trace_skeleton_paths(skeleton))

    height, width = processed.shape
    minimum_length = max(12.0, min(height, width) / 150.0)

    # Much stronger geometric simplification than the old 0.55–1.25 px.
    # This removes camera/pixel jitter while preserving the handwriting form.
    epsilon = max(
        1.8,
        min(3.4, min(height, width) / 650.0),
    )

    svg_paths = []

    for raw in paths:
        if len(raw) < 2:
            continue

        raw_array = np.asarray(raw, dtype=np.float32)
        length = float(
            np.linalg.norm(np.diff(raw_array, axis=0), axis=1).sum()
        )

        if length < minimum_length:
            continue

        points = smooth_points(raw, epsilon)

        if len(points) < 2:
            continue

        svg_paths.append(
            f'<path d="{svg_path(points)}" fill="none" stroke="black" '
            'stroke-width="1" stroke-linecap="round" stroke-linejoin="round"/>'
        )

    if not svg_paths:
        raise ValueError("Nem sikerült rajzolható vonalpályát készíteni.")

    body = "\n  ".join(svg_paths)

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="210mm" height="297mm" '
        f'viewBox="0 0 {width} {height}" '
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
        svg = handwriting_to_svg(processed_img)
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
