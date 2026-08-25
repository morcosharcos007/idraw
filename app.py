
import base64
import json
import os
import re
import secrets
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from flask import Flask, Response, abort, render_template, request, url_for
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize

from stroke_reconstruction import handwriting_to_svg
from plotter_pipeline import DEFAULT_PROFILE, analyze_svg

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
    return {
        "folder": folder,
        "original": folder / "original.png",
        "processed": folder / "processed.png",
        "svg": folder / "output.svg",
        "plot_plan": folder / "plot-plan.json",
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
    """Create a conservative black-on-white handwriting mask."""
    if img is None or img.size == 0:
        raise ValueError("Üres kép.")

    h0, w0 = img.shape[:2]
    scale = min(1.0, MAX_VECTOR_DIMENSION / float(max(h0, w0)))
    if scale < 1.0:
        img = cv2.resize(
            img,
            (max(1, round(w0 * scale)), max(1, round(h0 * scale))),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = _odd(denoise, 1, 9)
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    height, width = gray.shape
    side = min(height, width)

    block = _odd(round(side / 35), 15, 81)
    adaptive_c = int(np.interp(int(threshold), [80, 240], [14, 3]))
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block, adaptive_c
    )

    paper = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(8.0, min(36.0, side / 28)))
    darkness = paper.astype(np.int16) - gray.astype(np.int16)
    cutoff = int(np.interp(int(threshold), [80, 240], [24, 7]))
    contrast = (darkness >= cutoff).astype(np.uint8) * 255

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    blue = (
        (hue >= 80) & (hue <= 145) & (sat >= 16) & (val <= 253)
        & (darkness >= max(3, cutoff // 2))
    )
    absolute_cutoff = int(np.interp(int(threshold), [80, 240], [90, 225]))
    absolute = (gray <= absolute_cutoff).astype(np.uint8) * 255
    ink = np.where(
        (adaptive > 0) | (contrast > 0) | blue | (absolute > 0),
        255, 0
    ).astype(np.uint8)

    ink = cv2.morphologyEx(
        ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    min_area = max(
        3, int(round((height * width / 1_000_000.0) * (1.5 + k / 2)))
    )
    cleaned = np.zeros_like(ink)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        touches = x <= 0 or y <= 0 or x + cw >= width or y + ch >= height
        component_mask = labels == i
        edge_border = touches and (
            (cw > width * 0.85 and ch < height * 0.08)
            or (ch > height * 0.85 and cw < width * 0.08)
            or area > height * width * 0.80
        )
        straight_artifact = _is_straight_thin_artifact(component_mask, stats[i])
        if area >= min_area and not edge_border and not straight_artifact:
            cleaned[component_mask] = 255

    return cv2.bitwise_not(cleaned)


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
    defaults = {
        "original": None,
        "processed": None,
        "threshold": 160,
        "denoise": 3,
        "quality": 3,
        "state_token": None,
        "svg_preview": None,
        "svg_download": None,
        "plot_plan_download": None,
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
    return {"status": "ok", "pipeline": "ultimate", "profile": DEFAULT_PROFILE.name}


@app.route("/process", methods=["POST"])
def process_image():
    try:
        threshold = max(80, min(240, int(request.form.get("threshold", 160))))
    except (TypeError, ValueError):
        threshold = 160
    try:
        denoise = _odd(int(request.form.get("denoise", 3)), 1, 9)
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
        return render_index(
            original=original, processed=processed, threshold=threshold,
            denoise=denoise, state_token=token, status="A feldolgozás sikerült."
        )
    except Exception as exc:
        app.logger.exception("PROCESS FAILED: %s", exc)
        return render_index(
            threshold=threshold, denoise=denoise,
            error=f"A feldolgozás nem sikerült: {exc}"
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
        quality = max(1, min(5, int(request.form.get("quality", 3))))
        raw_svg, reconstruction_stats = handwriting_to_svg(
            processed_img, quality=quality
        )
        validate_svg(raw_svg)

        machine_svg, plan = analyze_svg(raw_svg, DEFAULT_PROFILE)
        validate_svg(machine_svg)

        paths["svg"].write_text(machine_svg, encoding="utf-8")
        paths["plot_plan"].write_text(
            json.dumps(plan.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        encoded = base64.b64encode(machine_svg.encode("utf-8")).decode("ascii")
        stats = {
            **reconstruction_stats,
            **plan.to_dict(),
        }
        return render_index(
            original=encode_png(original_img),
            processed=encode_png(processed_img),
            state_token=token,
            quality=quality,
            svg_preview="data:image/svg+xml;base64," + encoded,
            svg_download=url_for("download_svg", state_token=token),
            plot_plan_download=url_for("download_plot_plan", state_token=token),
            svg_stats=stats,
            status=(
                "Ultimate SVG elkészült: középvonal + valószínű stroke-sorrend "
                "+ UUNA TEK gépi előellenőrzés."
            ),
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
            "Content-Disposition": 'attachment; filename="idraw-ultimate-uuna-tek.svg"',
            "Cache-Control": "no-store",
        },
    )


@app.route("/download-plot-plan")
def download_plot_plan():
    token = safe_token(request.args.get("state_token", ""))
    paths = state_paths(token)
    if not paths or not paths["plot_plan"].is_file():
        abort(404)
    plan = paths["plot_plan"].read_text(encoding="utf-8")
    try:
        json.loads(plan)
    except json.JSONDecodeError:
        abort(404)
    return Response(
        plan,
        mimetype="application/json",
        headers={
            "Content-Disposition": 'attachment; filename="idraw-plot-plan.json"',
            "Cache-Control": "no-store",
        },
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), debug=False)
