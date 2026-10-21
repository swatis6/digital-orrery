'''Config template: every setting the simulator currently accepts.
Copy/paste, change settings, and run.'''

from datetime import datetime, timezone
from orrery.simulator import AtmosphereConfig, CircularOrbit, OutputConfig, SimConfig, run
from orrery.spacecraft import Spacecraft

#satellite inertia matrix, comment out if not needed
# TESTERSAT_INERTIA = np.array([
#     [ 0, 0,  0],
#     [ 0,  0,  0],
#     [ 0,  0,  0],
# ])

CONFIG = SimConfig(
    # Run -------------
    run_name="my_run", #folder name under results/ -- same name WILL overwrite
    epoch=datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc), #UTC time at the start
    duration_days=60.0, #max run length (default 60)

    # Spacecraft -------------
    spacecraft=Spacecraft(
        name="TesterSat", #printouts + plot titles
        mass=4.0, #kg
        cd=2.2, #drag coefficient (2.2 typical for CubeSats)
        area=0.03, #m^2 facing the flow (3U long side ~0.03, end ~0.01)
        inertia=None, #TESTERSAT_INERTIA when needed, 3x3 array in kg m^2
    ),

    # Starting orbit -------------
    orbit=CircularOrbit(
        altitude=200e3, #m above the equator
        inclination_deg=0.0, #0 = equatorial, 51.6 = ISS orbit, 97 = SSO
        raan_deg=0.0, #deg, where orbit crosses the equator going north; sets beta angle/eclipses
        arg_latitude_deg=0.0, #deg, starting point along the orbit from the ascending node
    ),

    # Forces -------------
    use_j2=True, #is Earth oblate?
    atmosphere=AtmosphereConfig(
        model="nrlmsis", #"nrlmsis" / "exponential" / None (no drag)
        f107=150.0, #nrlmsis only: solar flux, ~70 solar min, ~150 moderate, ~200+ solar max
        f107a=150.0, #nrlmsis only: 81-day average solar flux
        ap=15.0, #nrlmsis only: geomagnetic index, 4 is quiet, 15+ is active, 100+ is a storm
        #set f107, f107a and ap all to None to use real data, which needs internet
    ),

    # Stopping -------------
    stop_altitude=100e3, #m, stop when altitude drops below this

    # Integrator -------------
    method="DOP853", #"DOP853" (adaptive) / "RK4" (fixed step, needs dt)
    dt=None, #s, step size only for RK4 (usually 10.0)
    rtol=1e-10, #DOP853 only: relative error per step (default 1e-10)
    atol=1e-6, #DOP853 only: absolute error per step, m and m/s (default 1e-6)
    #looser (1e-9 / 1e-3) = faster long runs, slightly less exact

    # Outputs -------------
    outputs=OutputConfig(
        # Delete line to skip plot, use plots={} for none.
        # Any option you left out uses default.
        plots={
            "altitude": {
                # "title": "Altitude", # default from the settings above
                # "reentry_altitude": 100e3, # m, dashed line (default: stop_altitude)
            },
            "ground_track": {
                # "title": "Ground Track",
                # "ground_stations": [("TesterGS", 37.94, -75.47)],  # (name, lat deg, lon deg)
                # "show_night": True, #shade the night side (default True)
                # "night_at": 0.0, #s after epoch for the night shading (default: start)
            },
            "orbit_3d": {
                # "title": "3D Orbit",
                # "color_by": "altitude", #"lighting" / "altitude" / or "time"
                # "elev": 22.0, #camera elevation, deg
                # "azim": None, #camera azimuth, deg (default 45 deg from the Sun)
            },
        },
        csv=True, #save trajectory.csv
        csv_points=5000, #rows in the CSV, evenly spaced in time (default 5000)
        show_plots=True, #open plot windows at the end; they're saved either way
        save=True, #False = save nothing, just print the summary
        results_dir="results", #folder to save in for all runs
    ),
)

if __name__ == "__main__":
    run(CONFIG)