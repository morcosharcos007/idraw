
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np

COMMAND_RE = re.compile(r"([MLCQZmlcqz])|(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")


@dataclass(frozen=True)
class PlotterProfile:
    name: str = "UUNA TEK iDraw 2.0 A4"
    bed_width_mm: float = 210.0
    bed_height_mm: float = 297.0
    margin_mm: float = 8.0
    draw_speed_mm_s: float = 25.0
    travel_speed_mm_s: float = 100.0
    pen_lift_overhead_s: float = 0.08
    max_acceleration_mm_s2: float = 600.0

    @property
    def printable_width_mm(self) -> float:
        return max(1.0, self.bed_width_mm - 2 * self.margin_mm)

    @property
    def printable_height_mm(self) -> float:
        return max(1.0, self.bed_height_mm - 2 * self.margin_mm)


DEFAULT_PROFILE = PlotterProfile()


@dataclass
class Stroke:
    points: np.ndarray
    original_index: int
    closed: bool = False

    @property
    def length(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())

    @property
    def start(self) -> np.ndarray:
        return self.points[0]

    @property
    def end(self) -> np.ndarray:
        return self.points[-1]

    def oriented(self, reverse: bool) -> "Stroke":
        if not reverse:
            return self
        return Stroke(self.points[::-1].copy(), self.original_index, self.closed)


@dataclass
class PlotPlan:
    profile: str
    strokes: int
    draw_distance_mm: float
    pen_up_distance_mm: float
    total_distance_mm: float
    estimated_draw_seconds: float
    estimated_travel_seconds: float
    estimated_total_seconds: float
    bounds_mm: tuple[float, float, float, float]
    fits_bed: bool
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _numbers_for_command(tokens: list[float], command: str) -> int:
    return {"M": 2, "L": 2, "C": 6, "Q": 4, "Z": 0}[command.upper()]


def _tokenize_path(d: str):
    for match in COMMAND_RE.finditer(d or ""):
        token = match.group(1)
        if token:
            yield token
        else:
            yield float(match.group(2))


def _cubic(p0, p1, p2, p3, t):
    u = 1.0 - t
    return (u**3) * p0 + 3 * (u**2) * t * p1 + 3 * u * (t**2) * p2 + (t**3) * p3


def _quadratic(p0, p1, p2, t):
    u = 1.0 - t
    return (u * u) * p0 + 2 * u * t * p1 + (t * t) * p2


def path_to_points(d: str, curve_step: float = 1.5) -> np.ndarray:
    """Approximate an SVG path into XY points in document coordinates.

    Supports M/L/C/Q/Z, including repeated coordinate sets and relative forms.
    This is intentionally dependency-free so the web service remains small.
    """
    tokens = list(_tokenize_path(d))
    i = 0
    cmd = None
    current = np.zeros(2, dtype=float)
    start = current.copy()
    points: list[np.ndarray] = []

    def take(n):
        nonlocal i
        if i + n > len(tokens) or any(isinstance(x, str) for x in tokens[i:i+n]):
            raise ValueError("Hibás SVG path paraméterezés.")
        vals = tokens[i:i+n]
        i += n
        return [float(v) for v in vals]

    while i < len(tokens):
        if isinstance(tokens[i], str):
            cmd = tokens[i]
            i += 1
        if cmd is None:
            raise ValueError("SVG path parancs hiányzik.")

        absolute = cmd.isupper()
        base = cmd.upper()
        if base == "Z":
            if np.linalg.norm(current - start) > 1e-9:
                points.append(start.copy())
            current = start.copy()
            cmd = None
            continue

        n = _numbers_for_command([], cmd)
        if i + n > len(tokens):
            raise ValueError("Hiányos SVG path.")

        vals = take(n)
        if base == "M":
            p = np.array(vals[:2], dtype=float)
            if not absolute:
                p += current
            current = p
            start = p.copy()
            points.append(p.copy())
            # Subsequent coordinate pairs are implicit L commands.
            cmd = "L" if absolute else "l"
        elif base == "L":
            p = np.array(vals[:2], dtype=float)
            if not absolute:
                p += current
            current = p
            points.append(p.copy())
        elif base == "C":
            p1, p2, p3 = (
                np.array(vals[0:2], dtype=float),
                np.array(vals[2:4], dtype=float),
                np.array(vals[4:6], dtype=float),
            )
            if not absolute:
                p1 += current
                p2 += current
                p3 += current
            chord = float(np.linalg.norm(p3 - current))
            steps = max(2, int(math.ceil(max(chord, 1.0) / curve_step)))
            for t in np.linspace(0.0, 1.0, steps + 1)[1:]:
                points.append(_cubic(current, p1, p2, p3, float(t)))
            current = p3
        elif base == "Q":
            p1, p2 = (
                np.array(vals[0:2], dtype=float),
                np.array(vals[2:4], dtype=float),
            )
            if not absolute:
                p1 += current
                p2 += current
            chord = float(np.linalg.norm(p2 - current))
            steps = max(2, int(math.ceil(max(chord, 1.0) / curve_step)))
            for t in np.linspace(0.0, 1.0, steps + 1)[1:]:
                points.append(_quadratic(current, p1, p2, float(t)))
            current = p2
        else:
            raise ValueError(f"Nem támogatott SVG parancs: {cmd}")

    return np.asarray(points, dtype=float)


def extract_strokes(svg_text: str) -> tuple[list[Stroke], tuple[float, float]]:
    root = ET.fromstring(svg_text)
    viewbox = root.attrib.get("viewBox")
    if viewbox:
        vals = [float(v) for v in re.split(r"[\s,]+", viewbox.strip())]
        if len(vals) != 4:
            raise ValueError("Érvénytelen SVG viewBox.")
        vb_x, vb_y, vb_w, vb_h = vals
    else:
        vb_x = vb_y = 0.0
        vb_w = float(root.attrib.get("width", "210").replace("mm", ""))
        vb_h = float(root.attrib.get("height", "297").replace("mm", ""))

    strokes = []
    for idx, elem in enumerate(root.iter()):
        if elem.tag.split("}")[-1] != "path":
            continue
        d = elem.attrib.get("d", "")
        pts = path_to_points(d)
        if len(pts) >= 2:
            pts[:, 0] -= vb_x
            pts[:, 1] -= vb_y
            strokes.append(Stroke(pts, idx))
    return strokes, (vb_w, vb_h)


def _tangent(stroke: Stroke, at_start: bool) -> np.ndarray:
    pts = stroke.points if at_start else stroke.points[::-1]
    if len(pts) < 2:
        return np.zeros(2)
    span = min(6, len(pts) - 1)
    v = pts[span] - pts[0]
    n = np.linalg.norm(v)
    return v / n if n else np.zeros(2)


def _transition_cost(prev: Stroke, nxt: Stroke, reverse: bool) -> float:
    oriented = nxt.oriented(reverse)
    travel = float(np.linalg.norm(oriented.start - prev.end))
    if travel == 0:
        travel = 0.0
    a = _tangent(prev, False)
    b = _tangent(oriented, True)
    turn_penalty = 1.8 * (1.0 - float(np.clip(np.dot(a, b), -1.0, 1.0)))
    return travel + turn_penalty


def optimize_strokes(strokes: Iterable[Stroke]) -> list[Stroke]:
    remaining = list(strokes)
    if not remaining:
        return []
    # Deterministic, geometry-first start: longest meaningful stroke near upper-left.
    first = min(
        remaining,
        key=lambda s: (
            -s.length,
            float(s.start[1]),
            float(s.start[0]),
            s.original_index,
        ),
    )
    remaining.remove(first)

    ordered = [first]
    while remaining:
        prev = ordered[-1]
        candidates = []
        for j, s in enumerate(remaining):
            for reverse in (False, True):
                candidates.append((_transition_cost(prev, s, reverse), j, reverse))
        _, j, reverse = min(candidates, key=lambda x: (x[0], remaining[x[1]].original_index))
        chosen = remaining.pop(j).oriented(reverse)
        ordered.append(chosen)

    # Small 2-opt pass. It is bounded so a web request cannot explode on dense input.
    changed = True
    passes = 0
    while changed and passes < 2 and len(ordered) >= 4:
        changed = False
        passes += 1
        for i in range(1, len(ordered) - 2):
            a, b = ordered[i - 1], ordered[i]
            for j in range(i + 1, min(len(ordered) - 1, i + 18)):
                c, d = ordered[j], ordered[j + 1]
                before = _transition_cost(a, b, False) + _transition_cost(c, d, False)
                after = _transition_cost(a, c, False) + _transition_cost(b, d, False)
                if after + 0.5 < before:
                    ordered[i:j+1] = [s.oriented(True) for s in reversed(ordered[i:j+1])]
                    changed = True
                    break
            if changed:
                break
    return ordered


def _bounds(strokes: Iterable[Stroke]) -> tuple[float, float, float, float]:
    all_points = [s.points for s in strokes if len(s.points)]
    if not all_points:
        return (0.0, 0.0, 0.0, 0.0)
    p = np.vstack(all_points)
    return (float(p[:,0].min()), float(p[:,1].min()), float(p[:,0].max()), float(p[:,1].max()))


def _svg_length_to_mm(bounds, viewbox):
    x0, y0, x1, y1 = bounds
    vw, vh = viewbox
    # Our SVG uses width=210mm/height=297mm, so this is the correct physical scale.
    return (x0 * 210.0 / vw, y0 * 297.0 / vh, x1 * 210.0 / vw, y1 * 297.0 / vh)



def analyze_svg(
    svg_text: str,
    profile: PlotterProfile = DEFAULT_PROFILE,
    auto_fit: bool = True,
) -> tuple[str, PlotPlan]:
    strokes, viewbox = extract_strokes(svg_text)
    ordered = optimize_strokes(strokes)
    if not ordered:
        raise ValueError("Az SVG nem tartalmaz legalább egy rajzolható path-ot.")

    vw, vh = viewbox
    sx, sy = 210.0 / vw, 297.0 / vh
    b = _bounds(ordered)
    raw_bounds_mm = (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy)

    transform_scale = 1.0
    tx_mm = ty_mm = 0.0
    content_w = raw_bounds_mm[2] - raw_bounds_mm[0]
    content_h = raw_bounds_mm[3] - raw_bounds_mm[1]

    if auto_fit and content_w > 0 and content_h > 0:
        scale_x = profile.printable_width_mm / content_w
        scale_y = profile.printable_height_mm / content_h
        transform_scale = min(1.0, scale_x, scale_y)
        if transform_scale < 1.0:
            content_w *= transform_scale
            content_h *= transform_scale

        # Center within the safe printable rectangle.
        tx_mm = profile.margin_mm + (profile.printable_width_mm - content_w) / 2.0 - raw_bounds_mm[0] * transform_scale
        ty_mm = profile.margin_mm + (profile.printable_height_mm - content_h) / 2.0 - raw_bounds_mm[1] * transform_scale

    # Physical distance is measured after the same fit transform the machine will see.
    draw = 0.0
    for s in ordered:
        if len(s.points) < 2:
            continue
        pts_mm = s.points * np.array([sx, sy])
        if transform_scale != 1.0:
            pts_mm = pts_mm * transform_scale + np.array([tx_mm, ty_mm])
        draw += float(np.linalg.norm(np.diff(pts_mm, axis=0), axis=1).sum())

    travel = 0.0
    for a, bstroke in zip(ordered, ordered[1:]):
        a_end = a.points[-1] * np.array([sx, sy])
        b_start = bstroke.points[0] * np.array([sx, sy])
        if transform_scale != 1.0:
            a_end = a_end * transform_scale + np.array([tx_mm, ty_mm])
            b_start = b_start * transform_scale + np.array([tx_mm, ty_mm])
        travel += float(np.linalg.norm(b_start - a_end))

    fitted_bounds_mm = (
        raw_bounds_mm[0] * transform_scale + tx_mm,
        raw_bounds_mm[1] * transform_scale + ty_mm,
        raw_bounds_mm[2] * transform_scale + tx_mm,
        raw_bounds_mm[3] * transform_scale + ty_mm,
    )
    x0, y0, x1, y1 = fitted_bounds_mm
    fits = (
        x0 >= profile.margin_mm - 1e-6
        and y0 >= profile.margin_mm - 1e-6
        and x1 <= profile.bed_width_mm - profile.margin_mm + 1e-6
        and y1 <= profile.bed_height_mm - profile.margin_mm + 1e-6
    )

    warnings = []
    if transform_scale < 0.999999:
        warnings.append(
            f"A rajz automatikusan {transform_scale * 100:.1f}% méretre lett igazítva a biztonságos A4 területhez."
        )
    if not fits:
        warnings.append("A rajz tartalma kilóg a biztonsági margóból.")
    if draw < 10:
        warnings.append("Nagyon rövid rajzolási pálya.")
    if travel > draw * 0.35:
        warnings.append("Sok pen-up utazás várható; stroke-sorrend optimalizálás ajánlott.")

    draw_s = draw / max(profile.draw_speed_mm_s, 1e-6)
    travel_s = travel / max(profile.travel_speed_mm_s, 1e-6)
    total_s = draw_s + travel_s + max(0, len(ordered) - 1) * profile.pen_lift_overhead_s

    plan = PlotPlan(
        profile=profile.name,
        strokes=len(ordered),
        draw_distance_mm=round(draw, 2),
        pen_up_distance_mm=round(travel, 2),
        total_distance_mm=round(draw + travel, 2),
        estimated_draw_seconds=round(draw_s, 1),
        estimated_travel_seconds=round(travel_s, 1),
        estimated_total_seconds=round(total_s, 1),
        bounds_mm=tuple(round(v, 2) for v in fitted_bounds_mm),
        fits_bed=fits,
        warnings=warnings,
    )

    body = []
    for rank, stroke in enumerate(ordered, 1):
        pts_mm = stroke.points * np.array([sx, sy])
        pts_mm = pts_mm * transform_scale + np.array([tx_mm, ty_mm])
        # Convert the final physical millimetres back into the original SVG viewBox.
        pts = pts_mm * np.array([vw / 210.0, vh / 297.0])
        d = "M" + f"{pts[0,0]:.2f},{pts[0,1]:.2f}"
        for p in pts[1:]:
            d += f" L{p[0]:.2f},{p[1]:.2f}"
        body.append(
            f'<path id="stroke-{rank:04d}" data-original-index="{stroke.original_index}" '
            f'data-stroke-order="{rank}" d="{d}" fill="none" stroke="#000" '
            'stroke-width="1" stroke-linecap="round" stroke-linejoin="round"/>'
        )

    metadata = (
        f'<metadata id="idraw-plot-plan">profile={profile.name};'
        f'strokes={plan.strokes};draw_mm={plan.draw_distance_mm};'
        f'travel_mm={plan.pen_up_distance_mm};estimated_s={plan.estimated_total_seconds};'
        f'auto_fit_scale={transform_scale:.6f};fits={str(plan.fits_bed).lower()}</metadata>'
    )
    optimized_svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm" '
        f'viewBox="0 0 {vw:.4f} {vh:.4f}" preserveAspectRatio="xMidYMid meet">\n'
        f'  {metadata}\n'
        '  <g id="idraw-machine-ready" fill="none" stroke="#000" '
        'stroke-linecap="round" stroke-linejoin="round">\n'
        + "\n".join("    " + x for x in body)
        + '\n  </g>\n</svg>\n'
    )
    return optimized_svg, plan
