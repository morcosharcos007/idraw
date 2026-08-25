import base64
import os
import re
import secrets
import time
from itertools import combinations
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from flask import Flask, Response, abort, render_template, request, url_for
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

STATE_DIR = Path(os.environ.get("IDRAW_STATE_DIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "idraw-state")))
STATE_DIR.mkdir(parents=True, exist_ok=True)
MAX_VECTOR_DIMENSION = 2200


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
    return {"folder": folder, "original": folder / "original.png", "processed": folder / "processed.png", "svg": folder / "output.svg"}


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
    for folder in list(STATE_DIR.iterdir()) if STATE_DIR.exists() else []:
        try:
            if folder.is_dir() and folder.stat().st_mtime < cutoff:
                for child in folder.iterdir():
                    child.unlink(missing_ok=True)
                folder.rmdir()
        except OSError:
            pass


def _odd(value, minimum=3, maximum=81):
    value = int(max(minimum, min(maximum, value)))
    return value if value % 2 else value + 1


def _is_straight_thin_artifact(mask, stats_row):
    x, y, cw, ch, area = map(int, stats_row)
    length = max(cw, ch)
    thickness = min(cw, ch)
    if length < 45 or thickness > 3:
        return False
    ys, xs = np.where(mask)
    if len(xs) < 20:
        return False
    pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    center = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - center, full_matrices=False)
    axis = vt[0]
    perpendicular = np.abs((pts - center) @ np.array([-axis[1], axis[0]], dtype=np.float32))
    return float(np.percentile(perpendicular, 90)) < 0.85


def process(img, threshold, denoise):
    """Prepare a clean black-on-white handwriting mask.

    The important change is conservative cleanup: thin ink is never removed
    merely because it is small. We remove only isolated components and obvious
    page-edge artefacts, because UUNA TEK's centerline workflow needs the full
    stroke topology before vectorization.
    """
    if img is None or img.size == 0:
        raise ValueError("Üres kép.")

    h0, w0 = img.shape[:2]
    scale = min(1.0, MAX_VECTOR_DIMENSION / float(max(h0, w0)))
    if scale < 1.0:
        img = cv2.resize(img, (max(1, round(w0 * scale)), max(1, round(h0 * scale))), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = _odd(denoise, 1, 9)
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    height, width = gray.shape
    side = min(height, width)

    # Two complementary detectors: adaptive threshold handles uneven paper,
    # while local contrast keeps faint but continuous pen strokes.
    block = _odd(round(side / 35), 15, 81)
    adaptive_c = int(np.interp(int(threshold), [80, 240], [14, 3]))
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, adaptive_c)

    paper = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(8.0, min(36.0, side / 28)))
    darkness = paper.astype(np.int16) - gray.astype(np.int16)
    cutoff = int(np.interp(int(threshold), [80, 240], [24, 7]))
    contrast = (darkness >= cutoff).astype(np.uint8) * 255

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    blue = ((hue >= 80) & (hue <= 145) & (sat >= 16) & (val <= 253) & (darkness >= max(3, cutoff // 2)))
    absolute_cutoff = int(np.interp(int(threshold), [80, 240], [90, 225]))
    absolute = (gray <= absolute_cutoff).astype(np.uint8) * 255
    ink = np.where((adaptive > 0) | (contrast > 0) | blue | (absolute > 0), 255, 0).astype(np.uint8)

    # A tiny close repairs one-pixel breaks without broadening letters.
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    # Remove only genuinely isolated specks. Never use an aggressive opening.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    min_area = max(3, int(round((height * width / 1_000_000.0) * (1.5 + k / 2))))
    cleaned = np.zeros_like(ink)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        touches = x <= 0 or y <= 0 or x + cw >= width or y + ch >= height
        # A component touching the edge is retained unless it is clearly a
        # huge page border/background artefact.
        component_mask = labels == i
        edge_border = touches and ((cw > width * 0.85 and ch < height * 0.08) or (ch > height * 0.85 and cw < width * 0.08) or area > height * width * 0.80)
        straight_artifact = _is_straight_thin_artifact(component_mask, stats[i])
        if area >= min_area and not edge_border and not straight_artifact:
            cleaned[component_mask] = 255

    return cv2.bitwise_not(cleaned)


def _neighbors(point, point_set):
    y, x = point
    return [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx or dy) and (y + dy, x + dx) in point_set]


def _edge_key(a, b):
    return (a, b) if a < b else (b, a)


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.zeros(2, dtype=np.float32)


def _path_length(points):
    if len(points) < 2:
        return 0.0
    a = np.asarray(points, dtype=np.float32)
    return float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum())


def _endpoint_tangent(edge, reverse, look=8):
    seq = list(reversed(edge)) if reverse else edge
    if len(seq) < 2:
        return np.zeros(2, dtype=np.float32)
    span = min(look, len(seq) - 1)
    p0 = np.asarray(seq[0], dtype=np.float32)
    p1 = np.asarray(seq[span], dtype=np.float32)
    return _unit(p1 - p0)


def _prune_spurs(skeleton, max_branch_length):
    """Iteratively remove only short terminal branches from a skeleton."""
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
            choices = [q for q in nb[cur] if q != prev and _edge_key(cur, q) not in visited]
            if not choices:
                break
            # At a degree-2 skeleton pixel there should be one choice. If
            # rasterization creates a local branch, follow the least-turning one.
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

    # Pure loops have no graph nodes.
    if not nodes and points:
        start = points[0]
        nxt = nb[start][0]
        edges.append(walk(start, nxt))

    incident = {}
    for i, edge in enumerate(edges):
        incident.setdefault(edge[0], []).append((i, False))
        incident.setdefault(edge[-1], []).append((i, True))
    return edges, incident


def _best_pairing(items, edges, node):
    """Find the minimum-cost tangent pairing at a skeleton junction.

    For the small junction degrees found in handwriting, exact matching is
    cheap and much safer than greedy pairing. Opposite tangent directions are
    strongly preferred because they represent the continuation of one stroke.
    """
    if len(items) < 2:
        return []
    node_xy = np.asarray(node, dtype=np.float32)

    def cost(a, b):
        ia, ra = a
        ib, rb = b
        ta = _endpoint_tangent(edges[ia], ra)
        tb = _endpoint_tangent(edges[ib], rb)
        opposite = (1.0 + float(np.dot(ta, tb))) / 2.0
        # Penalize a large direction change; add a small distance-free tie-break.
        return opposite

    memo = {}
    def solve(rest):
        key = tuple(rest)
        if key in memo:
            return memo[key]
        if len(rest) < 2:
            result = (0.0, [])
            memo[key] = result
            return result
        first = rest[0]
        best = (float("inf"), None)
        for pos in range(1, len(rest)):
            second = rest[pos]
            remaining = rest[1:pos] + rest[pos + 1:]
            score, pairs = solve(remaining)
            score += cost(first, second)
            if score < best[0]:
                best = (score, [(first, second)] + pairs)
        memo[key] = best
        return best

    _, pairs = solve(tuple(items))
    return pairs


def trace_strokes(skeleton):
    edges, incident = _extract_edges(skeleton)
    if not edges:
        return []

    paired = {}
    for node, items in incident.items():
        for a, b in _best_pairing(items, edges, node):
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
            if coords[-1] == seq[0]:
                coords.extend(seq[1:])
            else:
                coords.extend(seq)
            used.add(nxt[0])
            current = nxt
        return [(x, y) for y, x in coords]

    # Start at true endpoints so stroke direction follows a natural writing flow.
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
    """Repair tiny raster gaps without joining separate letters."""
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
                for ra, aa in ((False, a), (True, list(reversed(a)))):
                    ta = _endpoint_tangent(aa, False, 8)
                    for rb, bb in ((False, b), (True, list(reversed(b)))):
                        tb = _endpoint_tangent(bb, False, 8)
                        d = float(np.linalg.norm(np.asarray(aa[-1]) - np.asarray(bb[0])))
                        align = float(np.dot(ta, tb))
                        candidates.append((d, align, aa, bb))
                d, align, aa, bb = min(candidates, key=lambda x: x[0] + (1.0 - x[1]) * gap)
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


def _resample(points, spacing=2.0):
    arr = np.asarray(points, dtype=np.float32)
    if len(arr) <= 2:
        return arr
    dist = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(dist)))
    total = float(cumulative[-1])
    if total < spacing * 2:
        return arr
    samples = np.arange(0.0, total + 1e-5, spacing, dtype=np.float32)
    x = np.interp(samples, cumulative, arr[:, 0])
    y = np.interp(samples, cumulative, arr[:, 1])
    return np.column_stack((x, y)).astype(np.float32)


def _smooth(points, quality=3):
    arr = _resample(points, 1.8)
    if len(arr) < 7:
        return arr
    q = max(1, min(5, int(quality)))
    # Smaller windows than before: remove pixel stair-stepping, but keep real
    # handwriting curvature and loops.
    window = _odd(5 + q * 2, 5, 15)
    if window >= len(arr):
        window = len(arr) - 1 if len(arr) % 2 == 0 else len(arr)
    if window < 5:
        return arr
    try:
        x = savgol_filter(arr[:, 0], window, 3, mode="interp")
        y = savgol_filter(arr[:, 1], window, 3, mode="interp")
        result = np.column_stack((x, y)).astype(np.float32)
        result[0] = arr[0]
        result[-1] = arr[-1]
        return result
    except ValueError:
        return arr


def _svg_path(points):
    p = [np.asarray(v, dtype=np.float32) for v in points]
    if len(p) < 2:
        return ""
    if len(p) == 2:
        return f"M{p[0][0]:.2f},{p[0][1]:.2f} L{p[1][0]:.2f},{p[1][1]:.2f}"

    commands = [f"M{p[0][0]:.2f},{p[0][1]:.2f}"]
    # Catmull-Rom with a restrained tension gives smooth, continuous curves
    # without the overshoot seen with aggressive smoothing.
    tension = 0.52
    for i in range(len(p) - 1):
        p0 = p[max(0, i - 1)]
        p1 = p[i]
        p2 = p[i + 1]
        p3 = p[min(len(p) - 1, i + 2)]
        c1 = p1 + (p2 - p0) * (tension / 6.0)
        c2 = p2 - (p3 - p1) * (tension / 6.0)
        commands.append(f"C{c1[0]:.2f},{c1[1]:.2f} {c2[0]:.2f},{c2[1]:.2f} {p2[0]:.2f},{p2[1]:.2f}")
    return " ".join(commands)


def handwriting_to_svg(processed, quality=3):
    """Convert a handwriting mask into UUNA-TEK/iDraw-friendly centerline SVG."""
    ink = (processed < 128).astype(np.uint8)
    if not np.any(ink):
        raise ValueError("Nem található vektorozható kézírás.")

    # Skeletonize the filled ink, then remove only tiny terminal noise.
    skeleton = skeletonize(ink > 0)
    side = min(processed.shape)
    skeleton = _prune_spurs(skeleton, max(4, int(round(side / 300))))
    if not np.any(skeleton):
        raise ValueError("A középvonal-képzés után nem maradt rajzolható vonal.")

    paths = trace_strokes(skeleton)
    paths = _merge_near_strokes(paths, gap=max(2.0, min(5.0, side / 450.0)), alignment=0.92)

    min_len = max(5.0, side / 240.0)
    svg_paths = []
    total_points = 0
    for path in paths:
        if _path_length(path) < min_len:
            continue
        smooth = _smooth(path, quality)
        if len(smooth) < 2:
            continue
        d = _svg_path(smooth)
        if not d:
            continue
        total_points += len(smooth)
        svg_paths.append(f'<path d="{d}" fill="none" stroke="#000000" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"/>')

    if not svg_paths:
        raise ValueError("Nem sikerült rajzolható vonalpályát készíteni.")

    height, width = processed.shape
    body = "\n  ".join(svg_paths)
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="210mm" height="297mm" viewBox="0 0 {width} {height}" '
        'preserveAspectRatio="xMidYMid meet">\n'
        '  <g id="idraw-handwriting" fill="none" stroke="#000000" '
        'stroke-linecap="round" stroke-linejoin="round">\n'
        f'  {body}\n'
        '  </g>\n</svg>\n'
    )
    return svg, {"paths": len(svg_paths), "points": total_points, "quality": int(quality)}


def validate_svg(svg_text):
    if not svg_text or "<svg" not in svg_text or "<path " not in svg_text:
        raise ValueError("Az SVG nem tartalmaz rajzolható path elemeket.")
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError as exc:
        raise ValueError("A generált SVG XML formátuma hibás.") from exc
    if root.tag.split("}")[-1] != "svg":
        raise ValueError("A dokumentum gyökéreleme nem SVG.")


def render_index(**values):
    defaults = {"original": None, "processed": None, "threshold": 160, "denoise": 3, "quality": 3, "state_token": None, "svg_preview": None, "svg_download": None, "svg_stats": None, "status": None, "error": None}
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
        denoise = _odd(denoise, 1, 9)
    except (TypeError, ValueError):
        denoise = 3

    upload = request.files.get("image")
    if not upload or not upload.filename:
        return render_index(threshold=threshold, denoise=denoise, error="Hiba: nem érkezett fájl.")

    img = cv2.imdecode(np.frombuffer(upload.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return render_index(threshold=threshold, denoise=denoise, error="A feltöltött kép nem olvasható.")

    try:
        original = encode_png(img)
        processed_img = process(img, threshold, denoise)
        processed = encode_png(processed_img)
        if original is None or processed is None:
            raise ValueError("A kép kódolása nem sikerült.")
        token = new_token()
        save_state(token, img, processed_img)
        return render_index(original=original, processed=processed, threshold=threshold, denoise=denoise, state_token=token, status="A feldolgozás sikerült.")
    except Exception as exc:
        app.logger.exception("PROCESS FAILED: %s", exc)
        return render_index(threshold=threshold, denoise=denoise, error=f"A feldolgozás nem sikerült: {exc}"), 500


@app.route("/generate-svg", methods=["POST"])
def generate_svg():
    token = safe_token(request.form.get("state_token", ""))
    state = load_state(token)
    if state is None:
        return render_index(error="Az SVG generálása nem sikerült: a feldolgozott kép munkamenete lejárt vagy nem található."), 400
    paths, original_img, processed_img = state
    try:
        quality = max(1, min(5, int(request.form.get("quality", 3))))
        svg, stats = handwriting_to_svg(processed_img, quality=quality)
        validate_svg(svg)
        paths["svg"].write_text(svg, encoding="utf-8")
        encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
        return render_index(original=encode_png(original_img), processed=encode_png(processed_img), state_token=token, quality=quality, svg_preview="data:image/svg+xml;base64," + encoded, svg_download=url_for("download_svg", state_token=token), svg_stats=stats, status="Az SVG sikeresen elkészült és el lett mentve.")
    except Exception as exc:
        app.logger.exception("SVG GENERATION FAILED: %s", exc)
        return render_index(original=encode_png(original_img), processed=encode_png(processed_img), state_token=token, error=f"Az SVG generálása nem sikerült: {exc}"), 500


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
    return Response(svg, mimetype="image/svg+xml", headers={"Content-Disposition": 'attachment; filename="idraw-vonalpalya.svg"', "Cache-Control": "no-store"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), debug=False)
