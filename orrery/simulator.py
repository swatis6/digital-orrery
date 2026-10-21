'''Runs a simulation from a SimConfig, so examples only need the settings.

from orrery.simulator import SimConfig, CircularOrbit, AtmosphereConfig, OutputConfig, run
traj = run(SimConfig(run_name="my_run", epoch=..., spacecraft=..., orbit=CircularOrbit(200e3)))

Each run saves into results/<run_name>/:
    config.txt --> every setting used, so the run can be repeated
    trajectory.csv --> evenly spaced states (if outputs.csv is True)
    <plot>.png --> one image per plot listed in outputs.plots
**Running again with the same run_name WILL overwrite that folder's files.** '''

from __future__ import annotations
import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from pprint import pformat
import numpy as np

from orrery.constants import DEG2RAD, R_EARTH, SECONDS_PER_DAY
from orrery.dynamics import ExponentialAtmosphere, ForceModel, NRLMSISAtmosphere
from orrery.orbits import circular_velocity
from orrery.propagate import Trajectory, altitude_event, propagate
from orrery.spacecraft import Spacecraft

PLOT_NAMES = ("altitude", "ground_track", "orbit_3d") #plots run() knows how to make

#Config pieces -------------

@dataclass
class CircularOrbit:
    '''Circular orbit. With the defaults (RAAN = 0, argument of latitude = 0) it starts
    at the ascending node on the ECI x-axis.'''
    altitude: float #m above the equatorial radius
    inclination_deg: float = 0.0
    raan_deg: float = 0.0 #where the orbit crosses the equator going north
    arg_latitude_deg: float = 0.0 #starting point along the orbit, from the ascending node

    def initial_state(self, body_radius: float = R_EARTH) -> np.ndarray:
        r = body_radius + self.altitude
        vc = circular_velocity(r)
        u = self.arg_latitude_deg * DEG2RAD
        i = self.inclination_deg * DEG2RAD
        raan = self.raan_deg * DEG2RAD
        r_plane = np.array([r * np.cos(u), r * np.sin(u), 0.0]) #position in orbit plane, x toward ascending node
        v_plane = np.array([-vc * np.sin(u), vc * np.cos(u), 0.0]) #velocity in orbit plane
        tilt = np.array([[1.0, 0.0, 0.0],
                         [0.0, np.cos(i), -np.sin(i)],
                         [0.0, np.sin(i), np.cos(i)]]) #tilt by inclination about x
        turn = np.array([[np.cos(raan), -np.sin(raan), 0.0],
                         [np.sin(raan), np.cos(raan), 0.0],
                         [0.0, 0.0, 1.0]]) #turn by RAAN about z
        rot = turn @ tilt
        return np.concatenate((rot @ r_plane, rot @ v_plane))

    def label(self) -> str:
        return f"{self.altitude / 1e3:.0f} km, {self.inclination_deg}°"


@dataclass
class AtmosphereConfig:
    '''model: "nrlmsis", "exponential", or None (no drag).
    f107, f107a, ap are only used by nrlmsis; set all three to None for real observed/predicted values.'''
    model: str | None = "nrlmsis"
    f107: float | None = 150.0
    f107a: float | None = 150.0
    ap: float | None = 15.0

    def build(self, epoch: datetime):
        if self.model is None:
            return None
        if self.model == "exponential":
            return ExponentialAtmosphere()
        if self.model == "nrlmsis":
            return NRLMSISAtmosphere(epoch=epoch, f107=self.f107, f107a=self.f107a, ap=self.ap)
        raise ValueError(f'unknown atmosphere model "{self.model}" (use "nrlmsis", "exponential" or None)')

    def label(self) -> str:
        return {None: "no drag", "exponential": "exponential atm", "nrlmsis": "NRLMSIS"}.get(self.model, str(self.model))


@dataclass
class OutputConfig:
    '''plots: {plot name: extra keyword arguments for that plot}, e.g. {"orbit_3d": {"color_by": "altitude"}}.
    Plot names: "altitude", "ground_track", "orbit_3d". Pass "title" in the kwargs to override the default title.'''
    plots: dict[str, dict] = field(default_factory=lambda: {name: {} for name in PLOT_NAMES})
    csv: bool = True
    csv_points: int = 5000 #evenly spaced rows in trajectory.csv
    show_plots: bool = False #open plot windows at the end (they're saved either way)
    save: bool = True #False = save nothing, just return the trajectory (used by the tests)
    results_dir: str = "results"


@dataclass
class SimConfig:
    run_name: str #also the folder name under results/
    epoch: datetime #UTC at t=0
    spacecraft: Spacecraft
    orbit: CircularOrbit
    duration_days: float = 60.0
    use_j2: bool = True
    atmosphere: AtmosphereConfig = field(default_factory=AtmosphereConfig)
    stop_altitude: float | None = 100e3 #m; None = run the full duration
    method: str = "DOP853" #or "RK4" (then dt is required)
    dt: float | None = None
    rtol: float = 1e-10 #adaptive only: relative error per step
    atol: float = 1e-6 #adaptive only: absolute error per step (m and m/s)
    outputs: OutputConfig = field(default_factory=OutputConfig)

    def label(self) -> str:
        return f"{self.orbit.label()}" + (", J2" if self.use_j2 else "") + f", {self.atmosphere.label()}"

#Running -------------

def run(config: SimConfig) -> Trajectory:
    '''Build the force model, propagate, print a summary, and save results.'''
    unknown = set(config.outputs.plots) - set(PLOT_NAMES)
    if unknown: #check before a long run, not after
        raise ValueError(f"unknown plot(s) {sorted(unknown)}; choose from {PLOT_NAMES}")

    model = ForceModel(use_j2=config.use_j2, atmosphere=config.atmosphere.build(config.epoch),
                       spacecraft=config.spacecraft)
    events = [altitude_event(config.stop_altitude)] if config.stop_altitude is not None else []
    tolerances = {} if config.method.upper() == "RK4" else {"rtol": config.rtol, "atol": config.atol}
    traj = propagate(model, config.orbit.initial_state(), (0.0, config.duration_days * SECONDS_PER_DAY),
                     method=config.method, dt=config.dt, events=events, epoch=config.epoch, **tolerances)

    _print_summary(config, traj)
    if config.outputs.save:
        folder = Path(config.outputs.results_dir) / config.run_name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.txt").write_text(pformat(config) + "\n", encoding="utf-8")
        if config.outputs.csv:
            _save_csv(traj, folder / "trajectory.csv", config.outputs.csv_points)
        _save_plots(config, traj, folder)
        print(f"Results saved to {folder}")
    return traj


def _print_summary(config: SimConfig, traj: Trajectory) -> None:
    name, days = config.spacecraft.name, traj.t[-1] / SECONDS_PER_DAY
    years = f" ({days / 365.25:.2f} years)" if days > 365.25 else "" #adds years for long runs
    if traj.stopped_by_event:
        print(f"{name} reached {config.stop_altitude / 1e3:.0f} km after {days:.2f} days{years}.")
    else:
        print(f"{name} still in orbit after {days:.0f} days{years}.")


def _save_csv(traj: Trajectory, path: Path, n_points: int) -> None:
    '''Evenly spaced rows: time since epoch, UTC time, ECI state, altitude.'''
    s = traj.resample(n_points)
    alt = np.linalg.norm(s.r, axis=1) - R_EARTH
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "utc", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps", "altitude_m"])
        for t, y, h in zip(s.t, s.y, alt):
            utc = (traj.epoch + timedelta(seconds=float(t))).isoformat() if traj.epoch else ""
            w.writerow([f"{t:.3f}", utc, *(f"{v:.6f}" for v in y), f"{h:.3f}"])


def _save_plots(config: SimConfig, traj: Trajectory, folder: Path) -> None:
    if not config.outputs.plots:
        return
    import matplotlib.pyplot as plt #imported here so runs without plots don't need cartopy
    from orrery.visualization import basic_plots as viz

    name, label = config.spacecraft.name, config.label()
    makers = {
        "altitude": (viz.plot_altitude, f"{name} orbital decay · {label}",
                     {"reentry_altitude": config.stop_altitude}),
        "ground_track": (viz.plot_ground_track, f"{name} ground track · {label}", {}),
        "orbit_3d": (viz.plot_orbit_3d, f"{name} orbit · {label}", {}),
    }
    figures = []
    for plot_name, options in config.outputs.plots.items():
        func, title, defaults = makers[plot_name]
        kwargs = {**defaults, "title": title, **options} #options override defaults
        fig, _ = func(traj, **kwargs)
        fig.savefig(folder / f"{plot_name}.png", dpi=150, bbox_inches="tight")
        figures.append(fig)

    if config.outputs.show_plots:
        plt.show()
    for fig in figures:
        plt.close(fig)