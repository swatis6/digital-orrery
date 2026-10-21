'''Simple, static, 2 dimensional plots to get started.
    viz.plot_altitude(traj, reentry_altitude=100e3)
    viz.plot_ground_track(traj, ground_stations=[("GS", 37.94, -75.47)])
    viz.plot_orbit_3d(traj)
    viz.plt.show()
'''

from __future__ import annotations
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from orrery.constants import R_EARTH
from orrery.frames import J2000, ecef_to_eci, ecef_to_latlon, eci_to_ecef
from orrery.propagate import Trajectory
from orrery.sun import in_eclipse, sun_position_eci

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


# Colors
SPACE = "#0B0F1A"
OCEAN_DAY = np.array([0.20, 0.45, 0.75])
OCEAN_NIGHT = np.array([0.05, 0.08, 0.15])
SUNLIT = "#FFB347"
ECLIPSE = "#8EA7FF"
START = "#4ADE80"
END = "#F87171"
MAP_OCEAN = "#DCE8F2"
MAP_LAND = "#F4F1EA"
MAP_SUNLIT = "#E8890C"
MAP_ECLIPSE = "#3B5BA9"
MAP_NIGHT = (0.04, 0.09, 0.28)


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


def _coastlines() -> list[np.ndarray]:
    if not HAS_CARTOPY:
        return []
    try:
        lines = []
        for geom in cfeature.COASTLINE.with_scale("110m").geometries():
            for line in getattr(geom, "geoms", [geom]):
                lines.append(np.asarray(line.coords))
        return lines
    except Exception:
        return []


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
    '''
    epoch = _epoch(traj, epoch)
    tr = _sample(traj, max_points=50_000)
    lat, lon = ecef_to_latlon(eci_to_ecef(tr.r, epoch, tr.t))

    if ax is None:
        fig = plt.figure(figsize=(12, 6.2))
        ax = fig.add_subplot(projection=ccrs.PlateCarree()) if HAS_CARTOPY else fig.add_subplot()
    fig = ax.figure

    if HAS_CARTOPY and hasattr(ax, "projection"):
        ax.set_global()
        ax.add_feature(cfeature.OCEAN, facecolor=MAP_OCEAN, zorder=0)
        ax.add_feature(cfeature.LAND, facecolor=MAP_LAND, zorder=0)
        ax.coastlines(lw=0.5, color="#8A96A3", zorder=1)
        gl = ax.gridlines(draw_labels=True, lw=0.4, color="#9AA5B1", alpha=0.5,
                          xlocs=range(-180, 181, 30), ylocs=range(-90, 91, 30))
        gl.top_labels = gl.right_labels = False
    else:
        ax.set_facecolor(MAP_OCEAN)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_aspect("equal")
        ax.set_xticks(range(-180, 181, 30))
        ax.set_yticks(range(-90, 91, 30))
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}°")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}°")
        ax.grid(True, color="white", lw=0.8, alpha=0.9)
        ax.set_axisbelow(True)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

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
        ax.imshow(img, extent=(-180, 180, -90, 90), origin="lower",
                  interpolation="bilinear", zorder=2)

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
    elev: float = 22.0,
    azim: float | None = None,
    title: str | None = None,
    ax=None,
):
    """Orbit around a sun-lit Earth, oriented and lit as at the start of the run.
    Arcs are colored sunlit vs. eclipse, and arcs behind Earth are hidden for
    the chosen view. matplotlib can't depth-sort 3D, so pick the view with
    elev/azim (default: 45 deg from the Sun) rather than by dragging.
    """
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
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection="3d")
    fig = ax.figure
    fig.patch.set_facecolor(SPACE)
    ax.set_facecolor(SPACE)
    ax.computed_zorder = False#draw in order: Earth, grid, orbit, markers

    # Earth with a soft day/night terminator
    lon = np.linspace(0, 2 * np.pi, 241)
    lat = np.linspace(-np.pi / 2, np.pi / 2, 121)
    LON, LAT = np.meshgrid(lon, lat)
    normals = np.stack([np.cos(LAT) * np.cos(LON), np.cos(LAT) * np.sin(LON), np.sin(LAT)], -1)
    light = normals @ sun_hat
    day = np.clip((light + 0.08) / 0.16, 0.0, 1.0)[..., None]
    brightness = (0.55 + 0.45 * np.clip(light, 0.0, 1.0))[..., None]
    rgb = (1 - day) * OCEAN_NIGHT + day * OCEAN_DAY * brightness
    face = np.concatenate([rgb, np.ones(rgb.shape[:-1] + (1,))], axis=-1)
    ax.plot_surface(*(R * normals[..., i] for i in range(3)), facecolors=face,
                    rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, zorder=1)

    # Lat/lon grid every 30 deg, and coastlines if available (front side only)
    grid = []
    for lo in range(-180, 180, 30):
        la = np.linspace(-90, 90, 91)
        grid.append(np.column_stack((np.full_like(la, lo), la)))
    for la0 in range(-60, 61, 30):
        lo = np.linspace(-180, 180, 181)
        grid.append(np.column_stack((lo, np.full_like(lo, la0))))
    for lines, radius, style in (
        (grid, 1.002, dict(color="white", lw=0.4, alpha=0.18)),
        (_coastlines(), 1.003, dict(color="#E8EEF5", lw=0.6, alpha=0.8)),
    ):
        pts = ecef_to_eci(radius * R * _lonlat_to_unit(lines), epoch, t0)
        if len(pts):
            with np.errstate(invalid="ignore"):
                front = pts @ view > 0
            ax.plot(*_mask(pts, front).T, zorder=2, **style)

    # Orbit colored by lighting, hidden where Earth blocks it
    visible = ~_hidden(r_km, view, R)
    shadow = in_eclipse(tr.r, sun_position_eci(epoch, tr.t))
    ax.plot(*_mask(r_km, visible & _grow(~shadow)).T, color=SUNLIT, lw=1.2, zorder=3)
    ax.plot(*_mask(r_km, visible & _grow(shadow)).T, color=ECLIPSE, lw=1.2, zorder=3)
    for idx, color in ((0, START), (-1, END)):
        if visible[idx]:
            ax.scatter(*r_km[idx], color=color, s=40, edgecolor="white",
                       linewidth=0.6, depthshade=False, zorder=4)

    # Sun direction
    extent = max(np.linalg.norm(r_km, axis=1).max() * 1.08, 1.55 * R)
    base, tip = 1.12 * R * sun_hat, 0.92 * extent * sun_hat
    hidden = _hidden(np.array([base, tip]), view, R).any()
    ax.quiver(*base, *(tip - base), color=SUNLIT, lw=1.2, alpha=0.25 if hidden else 0.9,
              arrow_length_ratio=min(0.15, 0.3 * R / np.linalg.norm(tip - base)), zorder=3)
    ax.text(*(tip * 1.04), "Sun", color=SUNLIT, fontsize=9, zorder=4)

    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(-extent, extent)
    try:
        ax.set_box_aspect((1, 1, 1), zoom=1.45)
    except TypeError:
        ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    alt = np.linalg.norm(r_km, axis=1) - R
    fig.text(0.04, 0.04, f"Altitude {alt.min():,.0f}–{alt.max():,.0f} km\n"
                         f"Epoch {epoch:%Y-%m-%d %H:%M} UTC  ·  {_duration(tr.t[-1] - t0)}",
             color="#C9D3E0", fontsize=9, family="monospace")
    handles = [Line2D([], [], color=SUNLIT, lw=2, label="Sunlit"),
               Line2D([], [], color=ECLIPSE, lw=2, label="Eclipse"),
               Line2D([], [], marker="o", ls="", color=START, label="Start"),
               Line2D([], [], marker="o", ls="", color=END, label="End")]
    ax.legend(handles=handles, loc="upper right", frameon=False, labelcolor="#C9D3E0")
    ax.set_title(title or "Orbit", color="white", fontsize=13, pad=0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.94)
    return fig, ax