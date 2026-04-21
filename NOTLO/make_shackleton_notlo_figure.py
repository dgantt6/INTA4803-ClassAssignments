"""Generate NOTLO (Notice to Lunar Operators) visualizations for the lunar south pole.

This script downloads basemap imagery (and, for the corridor example, a numeric DEM)
from the LROC QuickMap WMS and produces two-panel figures intended for classroom
discussion of operational coordination:

- Figure A (Shackleton region): a baseline traverse crosses a time-phased hazard
    footprint vs a NOTLO-informed reroute.
- Figure B (south-pole ridge corridor): terrain-aware traversal planning uses a
    DEM-derived slope cost surface and runs A*; a NOTLO keep-out mask triggers
    replanning.

Coordinate conventions
----------------------
Two coordinate/bounding-box conventions are used on purpose:

1) Plotting extents (kilometers) for Matplotlib `imshow(..., extent=...)`:
     - order: `(xmin_km, xmax_km, ymin_km, ymax_km)`
     - this matches Matplotlib's expected `extent=` ordering.

2) WMS request bounding boxes (meters) for `GetMap`:
     - order: `(xmin_m, ymin_m, xmax_m, ymax_m)`
     - this matches WMS 1.1.1 `BBOX` ordering.

Methodology notes (illustrative, not flight-grade)
--------------------------------------------------
- Hazard radii/timing are not calibrated to a specific vehicle; they illustrate
    the *structure* of a notice (time window + spatial footprint).
- Terrain-aware planning is slope-weighted and threshold-blocked; it is meant to
    produce defensible, repeatable "terrain-respecting" traverses for slides.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image


@dataclass(frozen=True)
class WmsRequest:
    """Container for a single WMS GetMap request.

    The LROC QuickMap WMS provides lunar datasets (imagery and DEM) via standard
    `GetMap` calls. Storing all request parameters in a dataclass keeps the data
    provenance explicit, which is useful when teammates need to reproduce figures
    or swap layers.

    Attributes:
        base_url: WMS endpoint URL (e.g., ``https://wms.im-ldi.com/``).
        layers: Comma-separated WMS layer names (e.g., ``luna_wac_roi_south_summer``).
        srs: Spatial reference string for WMS 1.1.1 (passed as `SRS`).
        bbox_m: Bounding box in meters as `(xmin_m, ymin_m, xmax_m, ymax_m)`.
        width_px: Requested output image width in pixels.
        height_px: Requested output image height in pixels.
        transparent: If True, request an alpha channel (typical for PNG overlays).
        image_format: MIME type passed as `FORMAT` (e.g. ``image/png`` or
            ``image/tiff; mode=32bit`` for numeric DEM tiles).
        version: WMS protocol version. This script uses 1.1.1 because it uses
            `SRS` and avoids the WMS 1.3.0 axis-order complications.
    """

    base_url: str
    layers: str
    srs: str
    bbox_m: tuple[float, float, float, float]
    width_px: int
    height_px: int
    transparent: bool = False
    image_format: str = "image/png"
    version: str = "1.1.1"


def download_wms_map(req: WmsRequest, out_path: Path) -> Path:
    """Download a WMS tile and write it to disk.

    This helper is the single network I/O entrypoint for the script. It builds
    a WMS `GetMap` request (WMS 1.1.1) from the provided :class:`WmsRequest` and
    writes the response bytes verbatim to `out_path`.

    Args:
        req: Parameters describing the WMS request.
        out_path: Destination path to write the returned image (PNG or TIFF).

    Returns:
        The same `out_path` for convenience.

    Raises:
        requests.HTTPError: If the server returns a non-2xx response.
        RuntimeError: If the server returns a non-image payload (commonly an
            XML/text error message from the WMS).

    Notes:
        - Uses a 60s timeout so a missing network/WMS outage fails fast.
        - The `BBOX` values are taken exactly from `req.bbox_m` and formatted
          with 6 decimals; units are meters in the requested `SRS`.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)

    params = {
        "SERVICE": "WMS",
        "VERSION": req.version,
        "REQUEST": "GetMap",
        "LAYERS": req.layers,
        "STYLES": "",
        "FORMAT": req.image_format,
        "TRANSPARENT": "true" if req.transparent else "false",
        "SRS": req.srs,
        "BBOX": ",".join(f"{v:.6f}" for v in req.bbox_m),
        "WIDTH": str(req.width_px),
        "HEIGHT": str(req.height_px),
    }

    r = requests.get(req.base_url, params=params, timeout=60)
    r.raise_for_status()

    content_type = (r.headers.get("content-type") or "").lower()
    if "image" not in content_type:
        raise RuntimeError(
            f"Unexpected content-type from WMS: {content_type}. First 200 chars: {r.text[:200]!r}"
        )

    out_path.write_bytes(r.content)
    return out_path


def _imshow_basemap(ax: plt.Axes, img_path: Path, extent_km: tuple[float, float, float, float]) -> None:
    """Render a downloaded basemap image into an axes.

    The basemap is downloaded once from QuickMap WMS and saved locally. For
    presentation clarity, the PNG is converted to grayscale and contrast-stretched
    (2nd�"98th percentile) before plotting.

    Args:
        ax: Matplotlib axes to draw into.
        img_path: Path to the cached basemap image.
        extent_km: Plot extent in km as `(xmin_km, xmax_km, ymin_km, ymax_km)`.

    Notes:
        - `origin="upper"` preserves the WMS raster orientation.
        - The grayscale stretch improves legibility in slides without altering
          geometry.
    """

    img = Image.open(img_path)
    img_gray = img.convert("L")
    arr = np.asarray(img_gray)

    # Contrast stretch for readability.
    lo, hi = np.percentile(arr, [2, 98])
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)

    ax.imshow(
        arr,
        cmap="gray",
        extent=extent_km,
        origin="upper",  # preserve the WMS image orientation
        interpolation="bilinear",
    )


def _draw_hazard_footprints(
    ax: plt.Axes,
    center_xy_km: tuple[float, float],
    radii_km: Iterable[float],
    styles: Iterable[dict],
) -> None:
    """Draw time-phased NOTLO hazard footprints as concentric circles.

    Each radius corresponds to a different time phase of the notice window.
    The circles are illustrative (not calibrated) and are meant to visually
    communicate that a NOTLO can have both a *time window* and a *spatial extent*.

    Args:
        ax: Axes to draw into.
        center_xy_km: Hazard center in km `(x_km, y_km)`.
        radii_km: Circle radii in km (one per phase).
        styles: Matplotlib patch style dictionaries (one per phase).
    """

    cx, cy = center_xy_km
    for radius_km, style in zip(radii_km, styles, strict=True):
        circ = plt.Circle((cx, cy), radius_km, fill=False, **style)
        ax.add_patch(circ)


def _draw_traverse_line(ax: plt.Axes, pts_km: list[tuple[float, float]], style: dict) -> None:
    """Plot a traverse polyline.

    Args:
        ax: Axes to draw into.
        pts_km: Ordered polyline points in km `(x_km, y_km)`.
        style: Matplotlib `plot()` keyword args (color, linewidth, alpha, etc.).
    """

    xs = [p[0] for p in pts_km]
    ys = [p[1] for p in pts_km]
    ax.plot(xs, ys, **style)


def _add_notlo_box(
    ax: plt.Axes,
    lines: list[str],
    *,
    xy_axes: tuple[float, float] = (0.02, 0.02),
    ha: str = "left",
    va: str = "bottom",
) -> None:
    """Add a NOTLO metadata textbox anchored in axes coordinates.

    This places the text *on top of the map* (inside the axes). It is useful
    when there is guaranteed empty map space, but it can obscure features if the
    map is dense.

    Args:
        ax: Axes to annotate.
        lines: List of text lines to render.
        xy_axes: Anchor location in axes-fraction coordinates `(x, y)`.
        ha: Horizontal alignment.
        va: Vertical alignment.

    See also:
        `_add_notlo_box_figure` for placing metadata in figure whitespace to
        guarantee no overlap with map features.
    """

    x, y = xy_axes
    ax.text(
        x,
        y,
        "\n".join(lines),
        transform=ax.transAxes,
        va=va,
        ha=ha,
        fontsize=11,
        color="white",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "black",
            "alpha": 0.65,
            "edgecolor": "none",
        },
    )


def _add_notlo_box_figure(
    fig: plt.Figure,
    lines: list[str],
    *,
    xy_fig: tuple[float, float] = (0.98, 0.06),
    ha: str = "right",
    va: str = "bottom",
) -> None:
    """Add a NOTLO metadata textbox anchored in figure coordinates.

    This places the text in *figure whitespace* (outside the map axes). It is
    the preferred option when you need to guarantee that labels/craters on the
    basemap are never covered by the metadata.

    Args:
        fig: The Matplotlib figure.
        lines: List of text lines to render.
        xy_fig: Anchor location in figure-fraction coordinates `(x, y)`.
        ha: Horizontal alignment.
        va: Vertical alignment.
    """

    x, y = xy_fig
    fig.text(
        x,
        y,
        "\n".join(lines),
        transform=fig.transFigure,
        va=va,
        ha=ha,
        fontsize=11,
        linespacing=1.0,
        color="white",
        bbox={
            "boxstyle": "round,pad=0.30",
            "facecolor": "black",
            "alpha": 0.65,
            "edgecolor": "none",
        },
    )


def _add_scale_bar(ax: plt.Axes, length_km: float = 10.0) -> None:
    """Add an approximate distance scale bar.

    The plot axes are in kilometers, so we can draw a simple linear scale bar in
    axes-fraction coordinates. This is an *approximation* (projection distortion
    is ignored), but over small distances it provides a useful visual reference.

    Args:
        ax: Axes to annotate.
        length_km: Scale bar length in km.
    """

    x0, x1 = ax.get_xlim()
    width_km = float(abs(x1 - x0))
    if width_km <= 0:
        return

    frac = min(max(length_km / width_km, 0.05), 0.45)
    x_right = 0.97
    y = 0.96
    x_left = x_right - frac

    ax.plot([x_left, x_right], [y, y], transform=ax.transAxes, color="white", lw=3)
    ax.text(
        (x_left + x_right) / 2,
        y - 0.055,
        f"{length_km:.0f} km",
        transform=ax.transAxes,
        color="white",
        ha="center",
        va="top",
        fontsize=11,
    )


def _bbox_km_from_points(
    points_xy_km: Iterable[tuple[float, float]],
    *,
    buffer_km: float,
    clip_to_extent_km: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float]:
    """Build a plotting-extent bbox around points (km).

    This is used to request a smaller DEM tile for path planning around the
    route of interest. Limiting the raster size keeps A* tractable while still
    using real terrain data.

    Args:
        points_xy_km: Iterable of `(x_km, y_km)` points.
        buffer_km: Extra margin (km) added on all sides.
        clip_to_extent_km: Optional global plotting extent to clip against.

    Returns:
        A km bbox in *plotting* order: `(xmin_km, xmax_km, ymin_km, ymax_km)`.
    """

    xs = [p[0] for p in points_xy_km]
    ys = [p[1] for p in points_xy_km]
    xmin = min(xs) - buffer_km
    xmax = max(xs) + buffer_km
    ymin = min(ys) - buffer_km
    ymax = max(ys) + buffer_km

    if clip_to_extent_km is not None:
        exmin, exmax, eymin, eymax = clip_to_extent_km
        xmin = max(xmin, exmin)
        xmax = min(xmax, exmax)
        ymin = max(ymin, eymin)
        ymax = min(ymax, eymax)

    return xmin, xmax, ymin, ymax


def _bbox_km_to_m(bbox_km: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Convert a plotting-extent bbox (km) to a WMS bbox (m).

    Args:
        bbox_km: Plotting bbox `(xmin_km, xmax_km, ymin_km, ymax_km)`.

    Returns:
        WMS bbox `(xmin_m, ymin_m, xmax_m, ymax_m)`.
    """

    xmin_km, xmax_km, ymin_km, ymax_km = bbox_km
    return xmin_km * 1000.0, ymin_km * 1000.0, xmax_km * 1000.0, ymax_km * 1000.0


def _load_float32_tiff(path: Path) -> np.ndarray:
    """Loads a 32-bit floating-point TIFF as a NumPy array.

    The DEM is requested from LROC QuickMap's WMS using `image/tiff; mode=32bit`.
    Pillow reads these tiles as mode 'F' (float32). Slope estimation only depends
    on *differences* between adjacent pixels, so any absolute offset (e.g., radial
    distance from the Moon's center) does not need normalization.

    Args:
        path: Path to the downloaded DEM TIFF.

    Returns:
        DEM raster as a float32 NumPy array with shape `(height, width)`.
    """

    img = Image.open(path)
    return np.asarray(img, dtype=np.float32)


def _xy_km_to_rc(
    xy_km: tuple[float, float],
    *,
    bbox_m: tuple[float, float, float, float],
    shape_hw: tuple[int, int],
) -> tuple[int, int]:
    """Converts map coordinates (km) into DEM row/col indices.

    The WMS image convention is assumed: the first row corresponds to `y = ymax`.

    Args:
        xy_km: Map coordinates `(x_km, y_km)`.
        bbox_m: WMS bbox `(xmin_m, ymin_m, xmax_m, ymax_m)` for the DEM tile.
        shape_hw: DEM shape `(h, w)`.

    Returns:
        `(row, col)` indices into the DEM array, clipped to bounds.
    """

    xmin_m, ymin_m, xmax_m, ymax_m = bbox_m
    h, w = shape_hw
    x_m = xy_km[0] * 1000.0
    y_m = xy_km[1] * 1000.0

    col_f = (x_m - xmin_m) / max(xmax_m - xmin_m, 1e-9) * (w - 1)
    row_f = (ymax_m - y_m) / max(ymax_m - ymin_m, 1e-9) * (h - 1)

    col = int(np.clip(round(col_f), 0, w - 1))
    row = int(np.clip(round(row_f), 0, h - 1))
    return row, col


def _rc_to_xy_km(
    rc: tuple[int, int],
    *,
    bbox_m: tuple[float, float, float, float],
    shape_hw: tuple[int, int],
) -> tuple[float, float]:
    """Convert DEM row/col indices back into map coordinates (km).

    Args:
        rc: `(row, col)` indices into the DEM array.
        bbox_m: WMS bbox `(xmin_m, ymin_m, xmax_m, ymax_m)` for the DEM tile.
        shape_hw: DEM shape `(h, w)`.

    Returns:
        `(x_km, y_km)` in plotting coordinates.
    """

    xmin_m, ymin_m, xmax_m, ymax_m = bbox_m
    h, w = shape_hw
    row, col = rc

    x_m = xmin_m + (col / max(w - 1, 1)) * (xmax_m - xmin_m)
    y_m = ymax_m - (row / max(h - 1, 1)) * (ymax_m - ymin_m)
    return x_m / 1000.0, y_m / 1000.0


def _snap_to_nearest_unblocked(
    blocked: np.ndarray,
    rc: tuple[int, int],
    *,
    max_radius_px: int = 25,
) -> tuple[int, int] | None:
    """Snaps a start/goal to the nearest unblocked cell.

    Terrain rasters can have isolated steep/invalid cells. This helper keeps the
    path planner robust by moving the endpoint a few pixels if needed.

    Args:
        blocked: Boolean array where True means impassable.
        rc: Desired `(row, col)`.
        max_radius_px: Maximum search radius, in pixels.

    Returns:
        The first unblocked `(row, col)` found within the search radius, or None.
    """

    r0, c0 = rc
    h, w = blocked.shape
    if not blocked[r0, c0]:
        return rc

    for rad in range(1, max_radius_px + 1):
        rmin = max(0, r0 - rad)
        rmax = min(h - 1, r0 + rad)
        cmin = max(0, c0 - rad)
        cmax = min(w - 1, c0 + rad)
        for r in range(rmin, rmax + 1):
            for c in range(cmin, cmax + 1):
                if not blocked[r, c]:
                    return r, c

    return None


def _compute_slope_deg(dem_m: np.ndarray, *, dx_m: float, dy_m: float) -> np.ndarray:
    """Computes slope (degrees) from a DEM tile.

    The methodology is a standard local-gradient estimate:
    - A horizontal gradient is computed in x and y using pixel spacing.
    - The slope angle is `atan(sqrt((dz/dx)^2 + (dz/dy)^2))`.

    Args:
        dem_m: DEM raster in meters.
        dx_m: Pixel spacing in the x direction (meters/pixel).
        dy_m: Pixel spacing in the y direction (meters/pixel).

    Returns:
        Slope raster in degrees, float32.
    """

    d_drow, d_dcol = np.gradient(dem_m, dy_m, dx_m)
    slope_rad = np.arctan(np.sqrt(d_dcol**2 + d_drow**2))
    return np.degrees(slope_rad).astype(np.float32)


def _build_traversal_cost(
    slope_deg: np.ndarray,
    *,
    slope_max_deg: float,
    slope_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds a cost surface and a hard-block mask from slope.

    The rationale mirrors common rover routing assumptions:
    - Slopes above a maximum threshold are treated as impassable.
    - Lower slopes are preferred via a smooth cost penalty.

    The output is intended for illustrative, terrain-aware path planning.

    Cost model:
        - `blocked = slope > slope_max_deg` (or non-finite)
        - `norm = clip(slope / slope_max_deg, 0, 2)`
        - `cost = 1 + slope_weight * norm^2`

    Args:
        slope_deg: Slope raster in degrees.
        slope_max_deg: Maximum traversable slope in degrees.
        slope_weight: Strength of the slope penalty in the cost multiplier.

    Returns:
        `(cost, blocked)` where:
        - `cost` is a float array (distance multiplier), with `inf` for blocked.
        - `blocked` is a boolean array where True means impassable.
    """

    slope_max = max(float(slope_max_deg), 1e-6)
    blocked = ~np.isfinite(slope_deg) | (slope_deg > slope_max)

    # The cost is a multiplier on physical distance.
    norm = np.clip(slope_deg / slope_max, 0.0, 2.0)
    cost = (1.0 + float(slope_weight) * (norm**2)).astype(np.float32)
    cost[blocked] = np.inf
    return cost, blocked


def _hazard_keepout_mask(
    *,
    shape_hw: tuple[int, int],
    bbox_m: tuple[float, float, float, float],
    center_xy_km: tuple[float, float],
    radius_km: float,
) -> np.ndarray:
    """Creates a circular keep-out mask for the hazard footprint.

    The mask is used only for the NOTLO-informed path, modeling the decision
    rule: 'do not enter the hazard footprint during its active window.'

    Args:
        shape_hw: Raster shape `(h, w)`.
        bbox_m: WMS bbox `(xmin_m, ymin_m, xmax_m, ymax_m)`.
        center_xy_km: Keep-out circle center `(x_km, y_km)`.
        radius_km: Keep-out circle radius (km).

    Returns:
        Boolean array where True indicates cells inside the keep-out radius.
    """

    xmin_m, ymin_m, xmax_m, ymax_m = bbox_m
    h, w = shape_hw
    xs_m = np.linspace(xmin_m, xmax_m, w, dtype=np.float32)
    ys_m = np.linspace(ymax_m, ymin_m, h, dtype=np.float32)
    xx, yy = np.meshgrid(xs_m, ys_m)

    cx_m = center_xy_km[0] * 1000.0
    cy_m = center_xy_km[1] * 1000.0
    rr_m = float(radius_km) * 1000.0
    return ((xx - cx_m) ** 2 + (yy - cy_m) ** 2) <= (rr_m**2)


def _astar_path(
    *,
    cost: np.ndarray,
    blocked: np.ndarray,
    start_rc: tuple[int, int],
    goal_rc: tuple[int, int],
    dx_m: float,
    dy_m: float,
) -> list[tuple[int, int]] | None:
    """Computes a least-cost path on a raster grid using A*.

    The planner minimizes a weighted physical distance:
    - Each step cost is `(step_distance_m * cost_multiplier(cell))`.
    - The heuristic is straight-line physical distance to the goal.

    This implementation is intended for transparent, reproducible classroom
    methodology rather than real-time operations.

    Args:
        cost: Cost multiplier raster (same shape as `blocked`).
        blocked: Boolean mask where True indicates impassable cells.
        start_rc: Start cell as `(row, col)`.
        goal_rc: Goal cell as `(row, col)`.
        dx_m: X pixel spacing (meters/pixel).
        dy_m: Y pixel spacing (meters/pixel).

    Returns:
        List of `(row, col)` cells from start to goal (inclusive), or None if no
        path exists.

    Implementation details (for teammates extending the planner):
        - 8-connected moves are allowed.
        - The open set priority queue stores `(f, g, r, c)`.
        - A stale-queue check `g_cur != g_score[r, c]` avoids extra bookkeeping.
    """
    h, w = cost.shape
    sr, sc = start_rc
    gr, gc = goal_rc

    if blocked[sr, sc] or blocked[gr, gc]:
        return None

    g_score = np.full((h, w), np.inf, dtype=np.float64)
    g_score[sr, sc] = 0.0
    parent_r = np.full((h, w), -1, dtype=np.int32)
    parent_c = np.full((h, w), -1, dtype=np.int32)

    def heuristic_m(r: int, c: int) -> float:
        return math.hypot((r - gr) * dy_m, (c - gc) * dx_m)

    # 8-connected neighborhood with physical step lengths.
    neighbors: list[tuple[int, int, float]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            step_m = math.hypot(dr * dy_m, dc * dx_m)
            neighbors.append((dr, dc, step_m))

    # Priority queue entries are (f = g + h, g, row, col).
    open_heap: list[tuple[float, float, int, int]] = []
    heapq.heappush(open_heap, (heuristic_m(sr, sc), 0.0, sr, sc))

    while open_heap:
        # Pop the node with minimum f-score.
        f, g_cur, r, c = heapq.heappop(open_heap)
        if g_cur != g_score[r, c]:
            # This is an outdated queue entry (we already found a better way here).
            continue
        if (r, c) == (gr, gc):
            # We reached the goal; reconstruct using parent pointers.
            break

        for dr, dc, step_m in neighbors:
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                continue
            if blocked[nr, nc]:
                continue

            # Step cost is physical distance times the terrain cost multiplier.
            tentative = g_cur + step_m * float(cost[nr, nc])
            if tentative < g_score[nr, nc]:
                g_score[nr, nc] = tentative
                parent_r[nr, nc] = r
                parent_c[nr, nc] = c
                heapq.heappush(open_heap, (tentative + heuristic_m(nr, nc), tentative, nr, nc))

    if not np.isfinite(g_score[gr, gc]):
        return None

    path: list[tuple[int, int]] = []
    r, c = gr, gc
    path.append((r, c))
    while (r, c) != (sr, sc):
        pr = int(parent_r[r, c])
        pc = int(parent_c[r, c])
        if pr < 0 or pc < 0:
            return None
        r, c = pr, pc
        path.append((r, c))

    path.reverse()
    return path



def _rc_path_to_xy_km(
    path_rc: list[tuple[int, int]],
    *,
    bbox_m: tuple[float, float, float, float],
    shape_hw: tuple[int, int],
    max_points: int = 260,
) -> list[tuple[float, float]]:
    """Converts an A* grid path into a polyline in km coordinates.

    The polyline is lightly decimated to keep the plotted line readable.

    Args:
        path_rc: List of `(row, col)` cells (typically the A* output).
        bbox_m: WMS bbox `(xmin_m, ymin_m, xmax_m, ymax_m)`.
        shape_hw: Raster shape `(h, w)`.
        max_points: Maximum plotted points after decimation.

    Returns:
        List of `(x_km, y_km)` points suitable for plotting.
    """

    xy = [_rc_to_xy_km(rc, bbox_m=bbox_m, shape_hw=shape_hw) for rc in path_rc]
    if len(xy) <= max_points:
        return xy

    idx = np.linspace(0, len(xy) - 1, max_points, dtype=int)
    return [xy[i] for i in idx]


# --- Inserted function: _plan_dem_traverses ---
def _plan_dem_traverses(
    *,
    figs_dir: Path,
    wms_srs: str,
    extent_km: tuple[float, float, float, float],
    landing_xy_km: tuple[float, float],
    traverse_start_xy_km: tuple[float, float],
    traverse_goal_xy_km: tuple[float, float],
    buffer_km: float = 12.0,
    slope_max_deg: float = 20.0,
    slope_weight: float = 6.0,
    keepout_km: float = 6.0,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]] | None]:
    """Plan baseline and NOTLO traverses from a DEM tile."""

    planner_bbox_km = _bbox_km_from_points(
        [traverse_start_xy_km, traverse_goal_xy_km, landing_xy_km],
        buffer_km=buffer_km,
        clip_to_extent_km=extent_km,
    )
    planner_bbox_m = _bbox_km_to_m(planner_bbox_km)

    bbox_width_km = float(planner_bbox_km[1] - planner_bbox_km[0])
    bbox_height_km = float(planner_bbox_km[3] - planner_bbox_km[2])
    aspect = bbox_width_km / max(bbox_height_km, 1e-6)
    base_px = 360.0
    scale = math.sqrt(max(aspect, 1e-6))
    dem_w = int(max(min(base_px * scale, 900.0), 220.0))
    dem_h = int(max(min(base_px / scale, 900.0), 220.0))

    dem_req = WmsRequest(
        base_url="https://wms.im-ldi.com/",
        layers="luna_wac_dtm_numeric_meters_absolute",
        srs=wms_srs,
        bbox_m=planner_bbox_m,
        width_px=dem_w,
        height_px=dem_h,
        transparent=False,
        image_format="image/tiff; mode=32bit",
    )

    dem_path = figs_dir / (
        f"south_pole_corridor_wac_dtm_"
        f"{int(round(planner_bbox_km[0]))}_"
        f"{int(round(planner_bbox_km[1]))}_"
        f"{int(round(planner_bbox_km[2]))}_"
        f"{int(round(planner_bbox_km[3]))}_"
        f"{dem_w}x{dem_h}.tif"
    )
    if not dem_path.exists():
        download_wms_map(dem_req, dem_path)

    dem_m = _load_float32_tiff(dem_path)
    h, w = dem_m.shape
    xmin_m, ymin_m, xmax_m, ymax_m = planner_bbox_m
    dx_m = (xmax_m - xmin_m) / max(w - 1, 1)
    dy_m = (ymax_m - ymin_m) / max(h - 1, 1)

    slope_deg = _compute_slope_deg(dem_m, dx_m=dx_m, dy_m=dy_m)
    cost, blocked = _build_traversal_cost(slope_deg, slope_max_deg=slope_max_deg, slope_weight=slope_weight)

    start_rc = _xy_km_to_rc(traverse_start_xy_km, bbox_m=planner_bbox_m, shape_hw=(h, w))
    goal_rc = _xy_km_to_rc(traverse_goal_xy_km, bbox_m=planner_bbox_m, shape_hw=(h, w))
    start_rc = _snap_to_nearest_unblocked(blocked, start_rc) or start_rc
    goal_rc = _snap_to_nearest_unblocked(blocked, goal_rc) or goal_rc

    baseline_path_rc = _astar_path(
        cost=cost,
        blocked=blocked,
        start_rc=start_rc,
        goal_rc=goal_rc,
        dx_m=dx_m,
        dy_m=dy_m,
    )

    if baseline_path_rc is None:
        baseline_traverse = [traverse_start_xy_km, traverse_goal_xy_km]
    else:
        baseline_traverse = _rc_path_to_xy_km(baseline_path_rc, bbox_m=planner_bbox_m, shape_hw=(h, w))

    hazard_mask = _hazard_keepout_mask(
        shape_hw=(h, w),
        bbox_m=planner_bbox_m,
        center_xy_km=landing_xy_km,
        radius_km=keepout_km,
    )
    blocked_notlo = blocked | hazard_mask
    start_rc_n = _snap_to_nearest_unblocked(blocked_notlo, start_rc) or start_rc
    goal_rc_n = _snap_to_nearest_unblocked(blocked_notlo, goal_rc) or goal_rc

    notlo_path_rc = _astar_path(
        cost=cost,
        blocked=blocked_notlo,
        start_rc=start_rc_n,
        goal_rc=goal_rc_n,
        dx_m=dx_m,
        dy_m=dy_m,
    )

    if notlo_path_rc is None:
        notlo_traverse = None
    else:
        notlo_traverse = _rc_path_to_xy_km(notlo_path_rc, bbox_m=planner_bbox_m, shape_hw=(h, w))

    return baseline_traverse, notlo_traverse


def make_two_panel_notlo_figure(
    basemap_path: Path,
    out_png: Path,
    out_pdf: Path,
    *,
    extent_km: tuple[float, float, float, float],
    landing_xy_km: tuple[float, float],
    show_context_label: bool = True,
    context_label: str = "Shackleton crater (context)",
    context_xy_km: tuple[float, float] = (-2.5, 0.0),
    context_target_xy_km: tuple[float, float] | None = None,
    baseline_traverse_pts_km: list[tuple[float, float]] | None = None,
    notlo_traverse_pts_km: list[tuple[float, float]] | None = None,
    notlo_box_lines: list[str] | None = None,
    notlo_box_xy_axes: tuple[float, float] = (0.02, -2),
    notlo_box_ha: str = "left",
    notlo_box_va: str = "bottom",
    notlo_box_xy_fig: tuple[float, float] | None = None,
    notlo_box_fig_ha: str = "right",
    notlo_box_fig_va: str = "bottom",
) -> None:
    """Create and save a two-panel NOTLO operational impact figure.

    The figure structure is consistent across scenarios:
        - Left panel: baseline behavior (no notice).
        - Right panel: NOTLO-informed behavior (hazard avoidance).

    The same NOTLO hazard footprints (three time phases) are drawn on both
    panels; the difference is the traverse line.

    Args:
        basemap_path: Local basemap image path (downloaded from QuickMap WMS).
        out_png: Destination PNG path.
        out_pdf: Destination PDF path.
        extent_km: Plot extent `(xmin_km, xmax_km, ymin_km, ymax_km)`.
        landing_xy_km: Hazard center / lander site `(x_km, y_km)`.
        show_context_label: If True, annotate a named feature for defensibility.
        context_label: Text for the context annotation.
        context_xy_km: Map location of the context annotation arrow tip.
        baseline_traverse_pts_km: Optional polyline for the baseline traverse.
        notlo_traverse_pts_km: Optional polyline for the NOTLO-aware traverse.
        notlo_box_lines: Optional metadata text lines.
        notlo_box_xy_axes: Axes-fraction anchor if placing the box inside axes.
        notlo_box_ha: Horizontal alignment for axes-anchored box.
        notlo_box_va: Vertical alignment for axes-anchored box.
        notlo_box_xy_fig: If provided, place metadata in figure whitespace using
            figure-fraction coordinates; this guarantees no overlap with map
            features.
        notlo_box_fig_ha: Horizontal alignment for figure-anchored box.
        notlo_box_fig_va: Vertical alignment for figure-anchored box.

    Notes:
        - Fonts are intentionally sized for slide readability from the back of
          the room. Layout spacing is tuned to avoid clipping at those sizes.
        - This function does not return a Figure object; it writes files and
          closes the figure to keep batch runs memory-safe.
    """

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.3))

    titles = ["Baseline: No Notice", "NOTLO: Time-Bounded Hazard Notice"]

    for ax, title in zip(axes, titles, strict=True):
        _imshow_basemap(ax, basemap_path, extent_km=extent_km)
        ax.set_title(title, fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    if show_context_label:
        target_xy_km = context_target_xy_km if context_target_xy_km is not None else context_xy_km
        for ax in axes:
            ax.annotate(
                context_label,
                xy=target_xy_km,
                xytext=context_xy_km,
                color="white",
                fontsize=12,
                ha="center",
                va="center",
                arrowprops={"arrowstyle": "-", "color": "white", "lw": 1.2, "alpha": 0.9},
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.35, "edgecolor": "none"},
            )

    # Notional plume hazard footprints (illustrative, not calibrated).
    radii_km = [2.0, 5.0, 3.0]
    phase_labels = [
        "Approach/Descent (T−10 min → T0)",
        "Touchdown/Peak (T0 → T+10 min)",
        "Post-landing (T+10 min → T+2 hr)",
    ]
    styles = [
        {"ec": "#ffcc80", "lw": 2.0, "ls": "--", "alpha": 0.95},
        {"ec": "#ff9800", "lw": 2.6, "ls": "-", "alpha": 0.95},
        {"ec": "#ffb74d", "lw": 2.0, "ls": ":", "alpha": 0.95},
    ]

    for ax in axes:
        _draw_hazard_footprints(ax, landing_xy_km, radii_km, styles)
        ax.plot(
            [landing_xy_km[0]],
            [landing_xy_km[1]],
            marker="*",
            markersize=12,
            markeredgecolor="black",
            markerfacecolor="#ffd54f",
            zorder=5,
        )

    # Example traverse (one "other user") to illustrate conflict vs avoidance.
    # Panel A: planned path intersects the active hazard footprint.
    if baseline_traverse_pts_km is None:
        baseline_traverse_pts_km = [(-38, landing_xy_km[1]), (38, landing_xy_km[1])]
    _draw_traverse_line(
        axes[0],
        baseline_traverse_pts_km,
        style={"color": "#ef5350", "lw": 2.6, "alpha": 0.95},
    )

    # Panel B: NOTLO-informed reroute around the hazard while keeping the same endpoints.
    if notlo_traverse_pts_km is None:
        r_detour = radii_km[1] + 3.0
        start_x = baseline_traverse_pts_km[0][0]
        end_x = baseline_traverse_pts_km[-1][0]
        baseline_y = baseline_traverse_pts_km[0][1]
        xmin, xmax, ymin, ymax = extent_km
        lower = min(ymin, ymax)
        upper = max(ymin, ymax)

        detour_y = baseline_y - r_detour
        if detour_y < (lower + 2.0):
            detour_y = baseline_y + r_detour
        if detour_y > (upper - 2.0):
            detour_y = baseline_y - r_detour

        notlo_traverse_pts_km = [
            (start_x, baseline_y),
            (landing_xy_km[0] - r_detour, baseline_y),
            (landing_xy_km[0] - r_detour, detour_y),
            (landing_xy_km[0] + r_detour, detour_y),
            (landing_xy_km[0] + r_detour, baseline_y),
            (end_x, baseline_y),
        ]
    _draw_traverse_line(
        axes[1],
        notlo_traverse_pts_km,
        style={"color": "#26a69a", "lw": 2.6, "alpha": 0.95},
    )

    # Legend for hazard phases + example traverse + lander site.
    handles = []
    labels = []
    for st, label in zip(styles, phase_labels, strict=True):
        h = plt.Line2D([0], [0], color=st["ec"], lw=st["lw"], ls=st["ls"], alpha=st["alpha"])
        handles.append(h)
        labels.append(label)

    handles += [
        plt.Line2D([0], [0], color="#ef5350", lw=2.6),
        plt.Line2D([0], [0], color="#26a69a", lw=2.6),
        plt.Line2D([0], [0], marker="*", color="none", markerfacecolor="#ffd54f", markeredgecolor="black", markersize=10),
    ]
    labels += [
        "Traverse (no notice): Rover crosses hazard",
        "Traverse (NOTLO): Rover avoids hazard",
        "Lander site",
    ]

    fig.legend(
        handles,
        labels,
        loc="lower left",
        ncol=2,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.02, 0.04),
    )

    # NOTLO "header" fields box.
    if notlo_box_lines is None:
        notlo_box_lines = [
            "NOTLO (illustrative)",
            "Area: South pole region (example)",
            "Access: Nearby PSRs",
            "T0: Touchdown",
            "Window: T−10 min → T+2 hr",
            "Max radius: 5 km (peak)",
            "Contact: Mission ops/registry",
        ]
    if notlo_box_xy_fig is None:
        _add_notlo_box(
            axes[1],
            notlo_box_lines,
            xy_axes=notlo_box_xy_axes,
            ha=notlo_box_ha,
            va=notlo_box_va,
        )
    else:
        # Place the metadata in figure whitespace so it cannot obscure map features
        # (e.g., the Shackleton crater annotation).
        _add_notlo_box_figure(
            fig,
            notlo_box_lines,
            xy_fig=notlo_box_xy_fig,
            ha=notlo_box_fig_ha,
            va=notlo_box_fig_va,
        )

    # Scale bar on both panels.
    for ax in axes:
        _add_scale_bar(ax, length_km=10)

    # Basemap credit.
    fig.text(
        0.01,
        0.985,
        "Basemap: LROC WAC South Pole Summer Mosaic (via LROC QuickMap WMS)",
        ha="left",
        va="top",
        fontsize=10,
        color="#444444",
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.24, wspace=0.02)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.show()
    plt.close(fig)


def main() -> None:
    """Generate all figure outputs for the project.

    This entrypoint:
        1) Defines the basemap WMS request (south-pole optimized mosaic).
        2) Downloads the basemap once (cached on disk).
        3) Produces Figure A (Shackleton region) using simple illustrative traverses.
        4) Produces Figure B (south-pole corridor) using a DEM tile to build a
           slope-based cost surface and A* to generate a terrain-aware traverse.

    Teammates can extend this by:
        - Swapping `wms.layers` to other QuickMap datasets.
        - Adjusting `slope_max_deg` / `slope_weight` to represent different rover capabilities.
        - Replacing the single-leg corridor traverse with a multi-station "Apollo-style"
          traverse pattern (see TODO block in code).
    """

    project_dir = Path(__file__).resolve().parent
    figs_dir = project_dir / "figs"

    # QuickMap / LROC WMS
    # - Projection used here is an LROC-provided IAU2000 parameterized SRS.
    # - c_lon=0, c_lat=-90 centers the map on the lunar south pole.
    # NOTE on the "patchy" look:
    # `luna_wac_global` is a global mosaic that can show wedge-like gaps/seams when
    # reprojected very near the pole. For a more "continuous" south-pole
    # cutout, use a polar-optimized mosaic.
    wms = WmsRequest(
        base_url="https://wms.im-ldi.com/",
        layers="luna_wac_roi_south_summer",
        srs="IAU2000:30166,9001,0,-90",
        bbox_m=(-40_000, -40_000, 40_000, 40_000),
        width_px=1400,
        height_px=1400,
        transparent=False,
    )

    basemap_path = figs_dir / "shackleton_basemap_wac_south_summer_80km.png"
    if not basemap_path.exists():
        download_wms_map(wms, basemap_path)

    # Axis extent in km (matches bbox above).
    extent_km = (-40, 40, -40, 40)

    # FIGURE A: Shackleton-area example.
    out_png_a = figs_dir / "shackleton_notlo_no_notice_vs_notlo.png"
    out_pdf_a = figs_dir / "shackleton_notlo_no_notice_vs_notlo.pdf"
    landing_xy_km_a = (0, 10)
    traverse_start_xy_km_a = (-38.0, landing_xy_km_a[1])
    traverse_goal_xy_km_a = (38.0, landing_xy_km_a[1])
    baseline_traverse_a, notlo_traverse_a = _plan_dem_traverses(
        figs_dir=figs_dir,
        wms_srs=wms.srs,
        extent_km=extent_km,
        landing_xy_km=landing_xy_km_a,
        traverse_start_xy_km=traverse_start_xy_km_a,
        traverse_goal_xy_km=traverse_goal_xy_km_a,
        buffer_km=12.0,
        slope_max_deg=20.0,
        slope_weight=6.0,
        keepout_km=6.0,
    )
    make_two_panel_notlo_figure(
        basemap_path,
        out_png_a,
        out_pdf_a,
        extent_km=extent_km,
        landing_xy_km=landing_xy_km_a,
        show_context_label=True,
        context_label="Shackleton crater",
        context_xy_km=(-2.5, 0),
        context_target_xy_km=(7.0, -7.0),
        baseline_traverse_pts_km=baseline_traverse_a,
        notlo_traverse_pts_km=notlo_traverse_a,
        notlo_box_lines=[
            "NOTLO (illustrative)",
            "Area: Shackleton ridge (S pole)",
            "Reference time: T0 = touchdown",
            "Window: T−10 min → T+2 hr",
            "Max radius: 5 km (peak)",
            "Contact: mission ops / registry",
        ],
        notlo_box_xy_fig=(0.98, 0.01),
        notlo_box_fig_ha="right",
        notlo_box_fig_va="bottom",
    )

    # FIGURE B: Alternate south-pole corridor example (traverse does not cross a major crater).
    # This keeps the scenario plausible (operations near PSRs) without implying a rover would
    # drive into a crater interior.
    out_png_b = figs_dir / "lunar_south_pole_notlo_viz.png"
    out_pdf_b = figs_dir / "lunar_south_pole_notlo_viz.pdf"
    landing_xy_km_b = (-20.7, -25.4)

    # Terrain-aware traverse planning (illustrative).
    # - The map uses a real imagery basemap.
    # - The example traverse is generated using a real lunar DEM tile to favor lower-slope
    #   terrain, yielding a more mission-plausible path shape than a perfectly straight line.
    # - The baseline path ignores the NOTLO keep-out zone; the NOTLO-informed path treats
    #   the hazard footprint as impassable during its active window and replans around it.
    # TODO... *** maybe / potentially *** (Apollo-pattern realism while staying at the south pole):
    # - Apollo LRV traverses were executed as sequences of stations/waypoints, not a single
    #   uninterrupted straight-line drive. A station-style pattern can be approximated by
    #   defining intermediate waypoints (chosen to be safe/flat) and running A* on each leg,
    #   then concatenating the resulting polylines.
    # - A curvature/turn penalty can discourage sharp zig-zags by expanding the A* state to
    #   include heading (the previous move) and adding an extra cost when the heading changes.
    # - The plotted polyline can be smoothed/decimated after planning (e.g.,
    #   some simplification or a light moving-average), which improves visual
    #   plausibility without changing the underlying terrain-aware routing rationale.
    traverse_start_xy_km = (-38.0, landing_xy_km_b[1])
    traverse_goal_xy_km = (0.0, landing_xy_km_b[1])
    baseline_traverse_b, notlo_traverse_b = _plan_dem_traverses(
        figs_dir=figs_dir,
        wms_srs=wms.srs,
        extent_km=extent_km,
        landing_xy_km=landing_xy_km_b,
        traverse_start_xy_km=traverse_start_xy_km,
        traverse_goal_xy_km=traverse_goal_xy_km,
        buffer_km=12.0,
        slope_max_deg=20.0,
        slope_weight=6.0,
        keepout_km=6.0,
    )
    make_two_panel_notlo_figure(
        basemap_path,
        out_png_b,
        out_pdf_b,
        extent_km=extent_km,
        landing_xy_km=landing_xy_km_b,
        show_context_label=True,
        context_label="Shackleton crater",
        context_xy_km=(-2.5, 0),
        context_target_xy_km=(7.0, -7.0),
        baseline_traverse_pts_km=baseline_traverse_b,
        notlo_traverse_pts_km=notlo_traverse_b,
        notlo_box_lines=[
            "NOTLO (illustrative)",
            "Area: S pole sunlit ridge corridor",
            "Reference time: T0 = touchdown",
            "Window: T−10 min → T+2 hr",
            "Max radius: 5 km (peak)",
            "Contact: mission ops / registry",
        ],
        # Put metadata in bottom figure whitespace so it cannot cover Shackleton...
        # Not sure if we like it here or prefer it in the axes. Can adjust as needed.
        notlo_box_xy_fig=(0.98, 0.01),
        notlo_box_fig_ha="right",
        notlo_box_fig_va="bottom",
    )


if __name__ == "__main__":
    main()
