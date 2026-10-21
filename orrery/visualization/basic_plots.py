'''Static plots of a finished run.
    viz.plot_altitude(traj, reentry_altitude=100e3)
    viz.plot_ground_track(traj, ground_stations=[("GS", 37.94, -75.47)])
    viz.plot_orbit_3d(traj)                       # colored by sunlight / eclipse
    viz.plot_orbit_3d(traj, color_by="altitude")  # or "time", e.g. for a decay spiral
    viz.plt.show()

Every function returns (fig, ax). Coastlines and land come from Natural Earth
through cartopy; the first run downloads that data (needs internet once).
'''

from __future__ import annotations
import warnings
from datetime import datetime, timedelta
from functools import lru_cache

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.path import Path
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from orrery.constants import MU_EARTH, R_EARTH
from orrery.frames import J2000, ecef_to_eci, ecef_to_latlon, eci_to_ecef, gmst_at
from orrery.propagate import Trajectory
from orrery.sun import in_eclipse, sun_position_eci

# Colors
SPACE = "#0B0F1A"
OCEAN_DAY = np.array([0.16, 0.38, 0.66])
LAND_DAY = np.array([0.42, 0.50, 0.34])
ICE_DAY = np.array([0.86, 0.89, 0.93])
NIGHT = {"ocean": np.array([0.03, 0.05, 0.11]),
         "land": np.array([0.07, 0.08, 0.10]),
         "ice": np.array([0.12, 0.13, 0.16])}
SUNLIT = "#FFB347"
ECLIPSE = "#8EA7FF"
HIDDEN = "#9AA5B1"
START = "#4ADE80"
END = "#F87171"
TEXT_DARK_BG = "#C9D3E0"
MAP_OCEAN = "#DCE8F2"
MAP_LAND = "#F4F1EA"
MAP_SUNLIT = "#E8890C"
MAP_ECLIPSE = "#3B5BA9"
MAP_NIGHT = (0.04, 0.09, 0.28)
TRACK_CMAP = ListedColormap(colormaps["plasma"](np.linspace(0.35, 1.0, 256)))


#  Helpers --------
def _epoch(traj: Trajectory, epoch: datetime | None) -> datetime:
    if epoch is not None:
        return epoch
    if traj.epoch is not None:
        return traj.epoch
    warnings.warn("No epoch given; assuming J2000 (2000-01-01 12:00 UTC). "
                  "Pass epoch=... to propagate() for correct longitudes and lighting.")
    return J2000


def _sample(traj: Trajectory, n_points: int | None = None, max_spacing: float = 30.0,
            max_points: int = 200_000) -> Trajectory:
    '''Evenly spaced samples for plotting.
    By default, at most max_spacing seconds apart, so long runs (like a
    multi-week decay) don't alias the once-per-orbit altitude wiggle.
    '''
    span = traj.t[-1] - traj.t[0]
    if n_points is None:
        n_points = int(np.clip(span / max_spacing, 2000, max_points))
    if traj.interpolant is not None:
        return traj.resample(n_points)
    if len(traj.t) > n_points:
        idx = np.linspace(0, len(traj.t) - 1, n_points).astype(int)
        return Trajectory(traj.t[idx], traj.y[idx], epoch=traj.epoch)
    return traj


def _time_axis(t: np.ndarray) -> tuple[np.ndarray, str]:
    elapsed = t - t[0]
    if elapsed[-1] < 2 * 3600:
        return elapsed / 60.0, "Time (min)"
    if elapsed[-1] < 3 * 86400:
        return elapsed / 3600.0, "Time (hours)"
    return elapsed / 86400.0, "Time (days)"


def _duration(seconds: float) -> str:
    if seconds < 2 * 3600:
        return f"{seconds / 60:.1f} min"
    if seconds < 3 * 86400:
        return f"{seconds / 3600:.1f} hours"
    return f"{seconds / 86400:.2f} days"


def _orbit_summary(r: np.ndarray, v: np.ndarray, mu: float = MU_EARTH) -> str:
    '''Inclination, eccentricity, and period of the osculating orbit at one state.'''
    r_mag, v_sq = np.linalg.norm(r), v @ v
    h = np.cross(r, v)
    inc = np.degrees(np.arccos(h[2] / np.linalg.norm(h)))
    e = np.linalg.norm(((v_sq - mu / r_mag) * r - (r @ v) * v) / mu)  # eccentricity vector
    a = 1.0 / (2.0 / r_mag - v_sq / mu)                                 # vis-viva
    parts = [f"i {inc:.1f}°", f"e {e:.4f}"]
    if a > 0:
        parts.append(f"T {_duration(2 * np.pi * np.sqrt(a**3 / mu))}")
    return "  ·  ".join(parts)


def _view_vector(elev: float, azim: float) -> np.ndarray:
    '''Unit vector from the origin toward the viewer for a matplotlib 3D view.'''
    e, a = np.radians(elev), np.radians(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def _hidden(points: np.ndarray, view: np.ndarray, radius: float) -> np.ndarray:
    '''True for points Earth blocks from the viewer.'''
    points = np.atleast_2d(points)
    along = points @ view
    perp = np.linalg.norm(points - np.outer(along, view), axis=1)
    return (along < 0.0) & (perp < radius)


def _mask(points: np.ndarray, keep: np.ndarray) -> np.ndarray:
    '''NaN out dropped points; matplotlib breaks the line there.'''
    out = points.astype(float).copy()
    out[~keep] = np.nan
    return out


def _grow(m: np.ndarray) -> np.ndarray:
    '''Extend a boolean mask by one sample each way to close gaps between colored arcs'''
    g = m.copy()
    g[1:] |= m[:-1]
    g[:-1] |= m[1:]
    return g


def _lonlat_to_unit(lines) -> np.ndarray:
    '''Stack lon/lat polylines into one NaN-separated array of Earth-fixed unit vectors.'''
    chunks = []
    for line in lines:
        lon, lat = np.radians(line[:, 0]), np.radians(line[:, 1])
        chunks.append(np.column_stack((np.cos(lat) * np.cos(lon),
                                       np.cos(lat) * np.sin(lon), np.sin(lat))))
        chunks.append(np.full((1, 3), np.nan))
    return np.vstack(chunks) if chunks else np.empty((0, 3))


@lru_cache(maxsize=1)
def _coastlines() -> tuple[np.ndarray, ...]:
    '''Coastlines as (N, 2) lon/lat arrays (Natural Earth 110 m).'''
    lines = []
    for geom in cfeature.COASTLINE.with_scale("110m").geometries():
        for line in getattr(geom, "geoms", [geom]):
            lines.append(np.asarray(line.coords)[:, :2])
    return tuple(lines)


@lru_cache(maxsize=4)
def _land_mask(n_lon: int, n_lat: int) -> np.ndarray:
    '''(n_lat, n_lon) booleans: is the center of each globe face on land?'''
    lon = np.linspace(-180, 180, n_lon)
    lat = np.linspace(-90, 90, n_lat)
    lon_c = lon + 0.5 * (lon[1] - lon[0])
    lat_c = np.clip(lat + 0.5 * (lat[1] - lat[0]), -90, 90)
    LON, LAT = np.meshgrid(lon_c, lat_c)
    pts = np.column_stack((LON.ravel(), LAT.ravel()))
    land = np.zeros(len(pts), dtype=bool)
    for geom in cfeature.LAND.with_scale("110m").geometries():
        for poly in getattr(geom, "geoms", [geom]):
            ring = np.asarray(poly.exterior.coords)[:, :2]
            lo, hi = ring.min(axis=0), ring.max(axis=0)
            box = np.all((pts >= lo) & (pts <= hi), axis=1)
            if box.any():
                land[box] |= Path(ring).contains_points(pts[box])
    return land.reshape(LAT.shape)


def _earth_surface(epoch: datetime, t: float, sun_hat: np.ndarray, n_lon=241, n_lat=121):
    '''Globe mesh in ECI (km), rotated to Earth's orientation at t, with continents
    and day/night shading. Falls back to plain ocean if land data is unavailable.'''
    lon = np.radians(np.linspace(-180, 180, n_lon))
    lat = np.radians(np.linspace(-90, 90, n_lat))
    LON, LAT = np.meshgrid(lon, lat)
    LON_ECI = LON + gmst_at(epoch, t)
    n = np.stack([np.cos(LAT) * np.cos(LON_ECI), np.cos(LAT) * np.sin(LON_ECI), np.sin(LAT)], -1)

    try:
        land = _land_mask(n_lon, n_lat)
    except Exception as err:
        warnings.warn(f"Couldn't load land outlines ({err}); drawing ocean only.")
        land = np.zeros(LAT.shape, dtype=bool)
    ice = land & ((np.degrees(LAT) < -60) | (np.degrees(LAT) > 66))

    day_rgb = np.where(land[..., None], LAND_DAY, OCEAN_DAY)
    day_rgb = np.where(ice[..., None], ICE_DAY, day_rgb)
    night_rgb = np.where(land[..., None], NIGHT["land"], NIGHT["ocean"])
    night_rgb = np.where(ice[..., None], NIGHT["ice"], night_rgb)

    light = n @ sun_hat
    day = np.clip((light + 0.08) / 0.16, 0.0, 1.0)[..., None] # soft terminator
    brightness = (0.55 + 0.45 * np.clip(light, 0.0, 1.0))[..., None] # limb darkening
    rgb = (1 - day) * night_rgb + day * day_rgb * brightness
    face = np.concatenate([rgb, np.ones(rgb.shape[:-1] + (1,))], axis=-1)
    R = R_EARTH / 1e3
    return R * n[..., 0], R * n[..., 1], R * n[..., 2], face


# Altitude -------------
def plot_altitude(traj: Trajectory, reentry_altitude: float | None = None,
                  body_radius: float = R_EARTH, title: str | None = None, ax=None):
    """Altitude vs. time.
    reentry_altitude (m): draws that level and, if the run stopped on an event
    (e.g. altitude_event), labels the lifetime.
    """
    tr = _sample(traj)
    t, xlabel = _time_axis(tr.t)
    alt_km = (np.linalg.norm(tr.r, axis=1) - body_radius) / 1e3

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t, alt_km, lw=0.8, color="#2563EB")

    if reentry_altitude is not None:
        ax.axhline(reentry_altitude / 1e3, color="#6B7280", ls="--", lw=1.0)
        ax.text(0.0, reentry_altitude / 1e3, f"  {reentry_altitude / 1e3:.0f} km",
                va="bottom", fontsize=9, color="#6B7280", transform=ax.get_yaxis_transform())
    if traj.stopped_by_event:
        lifetime = traj.t[-1] - traj.t[0]
        ax.plot(t[-1], alt_km[-1], "o", color=END, mec="black", mew=0.6, zorder=3)
        ax.annotate(f"end after {_duration(lifetime)}", (t[-1], alt_km[-1]),
                    xytext=(-10, 14), textcoords="offset points", ha="right", va="bottom",
                    fontsize=9, color="#374151")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Altitude (km)")
    ax.set_title(title or "Altitude", loc="left")
    ax.grid(True, alpha=0.3, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.figure.tight_layout()
    return ax.figure, ax


# Ground track -----------
def plot_ground_track(
    traj: Trajectory,
    epoch: datetime | None = None,
    ground_stations: list[tuple[str, float, float]] | None = None,
    show_night: bool = True,
    night_at: float | None = None,
    title: str | None = None,
    ax=None):
    '''Ground track colored by sunlight/eclipse, with night-side shading.
    ground_stations: list of (name, latitude_deg, longitude_deg).
    night_at: seconds after epoch for the night shading (default: start of run).
    ax: optional cartopy axes, e.g. fig.add_subplot(projection=ccrs.PlateCarree()).
    '''
    epoch = _epoch(traj, epoch)
    tr = _sample(traj, max_points=50_000)
    lat, lon = ecef_to_latlon(eci_to_ecef(tr.r, epoch, tr.t))

    if ax is None:
        fig = plt.figure(figsize=(12, 6.2))
        ax = fig.add_subplot(projection=ccrs.PlateCarree())
    elif not hasattr(ax, "projection"):
        raise ValueError("plot_ground_track needs cartopy axes: "
                         "fig.add_subplot(projection=ccrs.PlateCarree())")
    fig = ax.figure

    # Base map. On a PlateCarree map, data coordinates are lon/lat in degrees,
    # so everything below is drawn in plain data coordinates.
    ax.set_global()
    ax.add_feature(cfeature.OCEAN, facecolor=MAP_OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor=MAP_LAND, zorder=0)
    ax.coastlines(resolution="110m", linewidth=0.5, color="#8A96A3", zorder=1)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="#9AA5B1", alpha=0.5,
                      xlocs=list(range(-180, 181, 30)), ylocs=list(range(-90, 91, 30)))
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 8, "color": "#4B5563"}

    if show_night:
        t_night = tr.t[0] if night_at is None else night_at
        sun_ecef = eci_to_ecef(sun_position_eci(epoch, t_night), epoch, t_night)
        s_lat, s_lon = (np.radians(v[0]) for v in ecef_to_latlon(sun_ecef))
        glon, glat = np.meshgrid(np.radians(np.linspace(-180, 180, 721)),
                                 np.radians(np.linspace(-90, 90, 361)))
        cos_sza = (np.sin(glat) * np.sin(s_lat)
                   + np.cos(glat) * np.cos(s_lat) * np.cos(glon - s_lon))
        img = np.zeros(cos_sza.shape + (4,))
        img[..., :3] = MAP_NIGHT
        img[..., 3] = np.clip(-cos_sza / 0.12, 0.0, 1.0) * 0.38
        ax.imshow(img, extent=(-180, 180, -90, 90), origin="lower", interpolation="bilinear",
                  transform=ax.transData, zorder=2)   # transData: no reprojection needed

    pts = np.column_stack((lon, lat))
    segs = np.stack((pts[:-1], pts[1:]), axis=1)
    keep = np.abs(np.diff(lon)) < 180.0
    shadow = in_eclipse(tr.r, sun_position_eci(epoch, tr.t))[:-1]
    colors = np.where(shadow, MAP_ECLIPSE, MAP_SUNLIT)[keep]
    ax.add_collection(LineCollection(segs[keep], colors=colors, linewidths=1.4, zorder=3))

    ax.plot(lon[0], lat[0], ls="", marker="o", ms=8, color=START, mec="black", mew=0.6, zorder=5)
    ax.plot(lon[-1], lat[-1], ls="", marker="o", ms=8, color=END, mec="black", mew=0.6, zorder=5)
    for name, gs_lat, gs_lon in ground_stations or []:
        ax.plot(gs_lon, gs_lat, ls="", marker="^", ms=9, color="#111827",
                mec="white", mew=0.8, zorder=6)
        ax.annotate(name, (gs_lon, gs_lat), xytext=(6, 6), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="#111827", zorder=6)

    handles = [Line2D([], [], color=MAP_SUNLIT, lw=2, label="Sunlit"),
               Line2D([], [], color=MAP_ECLIPSE, lw=2, label="Eclipse"),
               Line2D([], [], marker="o", ls="", color=START, mec="black", label="Start"),
               Line2D([], [], marker="o", ls="", color=END, mec="black", label="End")]
    if ground_stations:
        handles.append(Line2D([], [], marker="^", ls="", color="#111827", label="Ground station"))
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.9)

    subtitle = f"{epoch:%Y-%m-%d %H:%M} UTC + {_duration(tr.t[-1] - tr.t[0])}"
    if show_night:
        subtitle += (f"   ·   night side shown at "
                     f"{epoch + timedelta(seconds=float(t_night)):%H:%M} UTC")
    ax.set_title(f"{title or 'Ground track'}\n", fontsize=13, loc="left")
    ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=9, color="#4B5563")
    fig.tight_layout()
    return fig, ax


# 3d view --------------
def plot_orbit_3d(
    traj: Trajectory,
    epoch: datetime | None = None,
    color_by: str = "lighting",
    elev: float = 22.0,
    azim: float | None = None,
    title: str | None = None,
    ax=None,
):
    '''Orbit around Earth, with continents and lighting as at the start of the run.

    color_by: "lighting" (sunlit / eclipse), "time", or "altitude"; the last two
        add a color bar and make multi-orbit runs like a decay spiral readable.
    elev, azim: camera angles in degrees (default azim: 45 deg from the Sun, so
        the day side and terminator are in view). matplotlib can't depth-sort 3D,
        so set the view here rather than by dragging the window.
    The whole orbit is drawn, including the parts behind Earth.
    The view is orthographic (no perspective) and is framed on the orbit,
    so high orbits fill the picture.
    '''
    if color_by not in ("lighting", "time", "altitude"):
        raise ValueError('color_by must be "lighting", "time", or "altitude".')
    epoch = _epoch(traj, epoch)
    tr = _sample(traj, max_points=40_000)
    t0 = tr.t[0]
    r_km = tr.r / 1e3
    R = R_EARTH / 1e3

    sun_hat = sun_position_eci(epoch, t0)
    sun_hat /= np.linalg.norm(sun_hat)
    if azim is None:
        azim = np.degrees(np.arctan2(sun_hat[1], sun_hat[0])) - 45.0
    view = _view_vector(elev, azim)

    if ax is None:
        fig = plt.figure(figsize=(8.5, 8.5))
        ax = fig.add_subplot(projection="3d")
    fig = ax.figure
    fig.patch.set_facecolor(SPACE)
    ax.set_facecolor(SPACE)
    ax.computed_zorder = False  # draw in order: Earth, grid, orbit, markers

    # Earth
    X, Y, Z, face = _earth_surface(epoch, t0, sun_hat)
    ax.plot_surface(X, Y, Z, facecolors=face, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False, zorder=1)

    # Lat/lon grid every 30 deg (equator a little stronger) and coastlines, front side only
    def _front_lines(lines, radius, **style):
        pts = ecef_to_eci(radius * R * _lonlat_to_unit(lines), epoch, t0)
        if len(pts):
            with np.errstate(invalid="ignore"):
                front = pts @ view > 0
            ax.plot(*_mask(pts, front).T, zorder=2, **style)

    meridians = [np.column_stack((np.full(91, lo), np.linspace(-90, 90, 91)))
                 for lo in range(-180, 180, 30)]
    parallels = [np.column_stack((np.linspace(-180, 180, 181), np.full(181, la)))
                 for la in (-60, -30, 30, 60)]
    equator = [np.column_stack((np.linspace(-180, 180, 361), np.zeros(361)))]
    _front_lines(meridians + parallels, 1.002, color="white", lw=0.4, alpha=0.14)
    _front_lines(equator, 1.002, color="white", lw=0.6, alpha=0.3)
    try:
        _front_lines(_coastlines(), 1.003, color="#E8EEF5", lw=0.5, alpha=0.55)
    except Exception as err:
        warnings.warn(f"Couldn't load coastlines ({err}).")

    # Orbit
    if color_by == "lighting":
        shadow = in_eclipse(tr.r, sun_position_eci(epoch, tr.t))
        ax.plot(*_mask(r_km, _grow(~shadow)).T, color=SUNLIT, lw=1.4, zorder=3)
        ax.plot(*_mask(r_km, _grow(shadow)).T, color=ECLIPSE, lw=1.4, zorder=3)
        handles = [Line2D([], [], color=SUNLIT, lw=2, label="Sunlit"),
                   Line2D([], [], color=ECLIPSE, lw=2, label="Eclipse")]
    else:
        if color_by == "time":
            values, label = _time_axis(tr.t)
        else:
            values, label = np.linalg.norm(r_km, axis=1) - R, "Altitude (km)"
        segs = np.stack((r_km[:-1], r_km[1:]), axis=1)
        norm = Normalize(values.min(), values.max())
        mid = 0.5 * (values[:-1] + values[1:])
        ax.add_collection3d(Line3DCollection(segs, colors=TRACK_CMAP(norm(mid)),
                                             linewidths=1.4, zorder=3))
        cax = fig.add_axes([0.62, 0.075, 0.32, 0.016])     # bottom right, clear of the globe
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=TRACK_CMAP), cax=cax,
                            orientation="horizontal")
        cbar.set_label(label, color=TEXT_DARK_BG, fontsize=9)
        cbar.ax.tick_params(colors=TEXT_DARK_BG, labelsize=8)
        cbar.outline.set_edgecolor("#2A3A52")
        handles = []

    for idx, color in ((0, START), (-1, END)):
        ax.scatter(*r_km[idx], color=color, s=42, edgecolor="white",
                   linewidth=0.7, depthshade=False, zorder=4)

    # Frame on the orbit and Earth together (LEO stays Earth-centered)
    lo = np.minimum(r_km.min(axis=0), -R)
    hi = np.maximum(r_km.max(axis=0), R)
    half = max(0.54 * (hi - lo).max(), 1.55 * R)
    center = 0.5 * (lo + hi) if half > 1.55 * R else np.zeros(3)

    # Sun direction
    base, tip = 1.12 * R * sun_hat, (1.12 * R + 0.5 * half) * sun_hat
    sun_hidden = _hidden(np.array([base, tip]), view, R).any()
    ax.quiver(*base, *(tip - base), color=SUNLIT, lw=1.2, alpha=0.25 if sun_hidden else 0.9,
              arrow_length_ratio=min(0.15, 0.3 * R / np.linalg.norm(tip - base)), zorder=3)
    ax.text(*(tip * 1.04), "Sun", color=SUNLIT, fontsize=9, zorder=4)

    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(c - half, c + half)
    ax.set_proj_type("ortho")
    try:
        ax.set_box_aspect((1, 1, 1), zoom=1.45)   # zoom needs matplotlib >= 3.6
    except TypeError:
        ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    # Labels
    alt = np.linalg.norm(r_km, axis=1) - R
    fig.text(0.04, 0.04,
             f"Initial orbit   {_orbit_summary(tr.r[0], tr.v[0])}\n"
             f"Altitude        {alt.min():,.0f}–{alt.max():,.0f} km\n"
             f"Span            {_duration(tr.t[-1] - t0)} from {epoch:%Y-%m-%d %H:%M} UTC",
             color=TEXT_DARK_BG, fontsize=9, family="monospace")
    handles += [Line2D([], [], marker="o", ls="", color=START, label="Start"),
                Line2D([], [], marker="o", ls="", color=END, label="End")]
    ax.legend(handles=handles, loc="upper right", frameon=False, labelcolor=TEXT_DARK_BG)
    ax.set_title(title or "Orbit", color="white", fontsize=13, pad=0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.94)
    return fig, ax