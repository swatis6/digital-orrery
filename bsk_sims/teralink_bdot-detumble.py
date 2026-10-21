
import csv
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from Basilisk import __path__ as bsk_path
from Basilisk.architecture import messaging, sysModel
from Basilisk.simulation import (MtbEffector, magneticFieldCenteredDipole, magneticFieldWMM,
                                 magnetometer, spacecraft)
from Basilisk.fswAlgorithms import tamComm
from Basilisk.utilities import (SimulationBaseClass, macros, orbitalMotion, simIncludeGravBody,
                                simSetPlanetEnvironment, unitTestSupport)

# Run -------------
RUN_NAME = "TeraLink-1_BSKdetumble" #folder name under results/ -- same name WILL overwrite
EPOCH = "2027 July 01, 12:00:00.0 (UTC)" #deploy date, time of day not known yet
DURATION_ORBITS = 4.0 #run length in orbits
SHOW_PLOTS = True #open plot windows at the end; they're saved either way

# Spacecraft -------------
MASS = 11.0 #kg
INERTIA = np.array([
    [ 6.3197e-2, -5.3206e-3,  1.3419e-4],
    [-5.3206e-3,  2.1643e-2,  7.8170e-4],
    [ 1.3419e-4,  7.8170e-4,  7.6194e-2],
]) #kg m^2 about CoM, physics convention; must be in the same body frame as the torque rods
INITIAL_RATES_DEG = [10.0, -10.0, 10.0] #deg/s body rates at deploy (ADCS-1: <=10 deg/s per axis)
INITIAL_MRP = [0.1, 0.2, -0.3] #starting attitude (MRP); detumble barely depends on it

# Orbit -------------
ALTITUDE = 416e3 #m
INCLINATION_DEG = 51.6
RAAN_DEG = 0.0 #set from ISS state at deploy when known
ARG_LATITUDE_DEG = 0.0 #starting point along the orbit from the ascending node

# Magnetic field -------------
FIELD_MODEL = "dipole" # "dipole" (centered dipole)

# Magnetorquers (3x CubeTorquer CR0004) -------------
MAX_DIPOLE = 0.4 #A m^2 per rod; rods aligned with body x, y, z

# Magnetometer (CubeMag) -------------
MAG_NOISE_T = 0.0 #T, 1-sigma noise per axis; 0 = ideal (CubeMag spec TBD)

# Controller -------------
CONTROL_PERIOD = 1.0 #s, read magnetometer + command rods once per period
GAIN = None #N m s; None = Avanzini & Giulietti near-optimal gain from orbit + inertia
DYNAMICS_STEP = 0.1 #s, Basilisk integration step

# Reporting -------------
PAYLOAD_PLANE_AXES = (0, 2) #CAD frame: boresight on a 2U face = long axis = y (smallest inertia), so plane is x+z
RATE_REQUIREMENT_DEG = 0.1 #deg/s, ADCS-1
RESULTS_DIR = "results"


class BDotController(sysModel.SysModel):
    '''Normalized B-dot (Avanzini & Giulietti): m = -(k / |B|) * d(B/|B|)/dt.
    d/dt is a finite difference between consecutive magnetometer readings.
    If any axis would exceed the rod limit, the whole dipole is scaled down (keeps its direction).'''

    def __init__(self):
        super(BDotController, self).__init__()
        self.gain = 0.0 #N m s
        self.max_dipole = 0.4 #A m^2
        self.tamInMsg = messaging.TAMSensorBodyMsgReader()
        self.mtbCmdOutMsg = messaging.MTBCmdMsg()
        self._prev_b_hat = None
        self._prev_time = None

    def Reset(self, CurrentSimNanos):
        self._prev_b_hat = None
        self._prev_time = None

    def UpdateState(self, CurrentSimNanos):
        b_body = np.array(self.tamInMsg().tam_B, dtype=float)
        b_norm = np.linalg.norm(b_body)
        dipole = np.zeros(3)
        if b_norm > 0.0:
            b_hat = b_body / b_norm
            if self._prev_b_hat is not None:
                dt = (CurrentSimNanos - self._prev_time) * 1e-9
                b_hat_dot = (b_hat - self._prev_b_hat) / dt
                dipole = -(self.gain / b_norm) * b_hat_dot
                worst = np.max(np.abs(dipole)) / self.max_dipole
                if worst > 1.0:
                    dipole = dipole / worst
            self._prev_b_hat = b_hat
            self._prev_time = CurrentSimNanos

        payload = messaging.MTBCmdMsgPayload()
        n_slots = len(payload.mtbDipoleCmds) #fixed-size array in the message
        payload.mtbDipoleCmds = dipole.tolist() + [0.0] * (n_slots - 3)
        self.mtbCmdOutMsg.write(payload, CurrentSimNanos, self.moduleID)


def bdot_gain(inertia, n, inclination_rad):
    '''k = 2 n (1 + sin xi) J_min, with the geomagnetic inclination xi approximated by the orbit inclination'''
    j_min = np.min(np.linalg.eigvalsh(inertia))
    return 2.0 * n * (1.0 + np.sin(inclination_rad)) * j_min


def settled_time(t, ok):
    '''first time after which ok stays True to the end; None if it never settles'''
    if not ok[-1]:
        return None
    bad = np.flatnonzero(~ok)
    return t[0] if bad.size == 0 else t[bad[-1] + 1]


def run():
    sim = SimulationBaseClass.SimBaseClass()
    process = sim.CreateNewProcess("process")
    dyn_task, fsw_task = "dynTask", "fswTask"
    process.addTask(sim.CreateNewTask(dyn_task, macros.sec2nano(DYNAMICS_STEP)))
    process.addTask(sim.CreateNewTask(fsw_task, macros.sec2nano(CONTROL_PERIOD)))

    # Spacecraft
    sc = spacecraft.Spacecraft()
    sc.ModelTag = "TeraLink-1"
    sc.hub.mHub = MASS
    sc.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    sc.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(INERTIA.flatten().tolist())
    sim.AddModelToTask(dyn_task, sc)

    grav = simIncludeGravBody.gravBodyFactory()
    earth = grav.createEarth()
    earth.isCentralBody = True
    grav.addBodiesTo(sc)

    oe = orbitalMotion.ClassicElements()
    oe.a = earth.radEquator + ALTITUDE
    oe.e = 0.0
    oe.i = INCLINATION_DEG * macros.D2R
    oe.Omega = RAAN_DEG * macros.D2R
    oe.omega = 0.0
    oe.f = ARG_LATITUDE_DEG * macros.D2R
    r_n, v_n = orbitalMotion.elem2rv(earth.mu, oe)
    sc.hub.r_CN_NInit = r_n
    sc.hub.v_CN_NInit = v_n
    sc.hub.sigma_BNInit = [[s] for s in INITIAL_MRP]
    sc.hub.omega_BN_BInit = [[w * macros.D2R] for w in INITIAL_RATES_DEG]

    n = np.sqrt(earth.mu / oe.a**3) #orbit rate, rad/s
    period = 2.0 * np.pi / n
    duration = DURATION_ORBITS * period

    # Magnetic field
    if FIELD_MODEL == "wmm":
        mag = magneticFieldWMM.MagneticFieldWMM()
        data_path = os.path.join(bsk_path[0], "supportData", "MagneticField") + os.sep
        if not os.path.exists(os.path.join(data_path, "WMM.COF")):
            raise FileNotFoundError(f"WMM coefficient file not found in {data_path}; "
                                    'set FIELD_MODEL = "dipole" or check the Basilisk install')
        mag.dataPath = data_path
        mag.epochInMsg.subscribeTo(unitTestSupport.timeStringToGregorianUTCMsg(EPOCH))
    elif FIELD_MODEL == "dipole":
        mag = magneticFieldCenteredDipole.MagneticFieldCenteredDipole()
        simSetPlanetEnvironment.centeredDipoleMagField(mag, "earth")
    else:
        raise ValueError('FIELD_MODEL must be "wmm" or "dipole"')
    mag.ModelTag = "magField"
    mag.addSpacecraftToModel(sc.scStateOutMsg)
    sim.AddModelToTask(dyn_task, mag)

    # Torque rods
    mtb_config = messaging.MTBArrayConfigMsgPayload()
    mtb_config.numMTB = 3
    mtb_config.GtMatrix_B = [1.0, 0.0, 0.0,
                             0.0, 1.0, 0.0,
                             0.0, 0.0, 1.0] #row-major, one column per rod
    mtb_config.maxMtbDipoles = [MAX_DIPOLE] * 3
    mtb_config_msg = messaging.MTBArrayConfigMsg().write(mtb_config)

    mtb = MtbEffector.MtbEffector()
    mtb.ModelTag = "CR0004x3"
    sc.addDynamicEffector(mtb)
    sim.AddModelToTask(dyn_task, mtb)

    # Magnetometer -> body frame -> B-dot
    tam = magnetometer.Magnetometer()
    tam.ModelTag = "CubeMag"
    tam.scaleFactor = 1.0
    tam.senNoiseStd = [MAG_NOISE_T] * 3
    sim.AddModelToTask(fsw_task, tam)

    tam_comm = tamComm.tamComm()
    tam_comm.ModelTag = "tamComm"
    tam_comm.dcm_BS = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] #sensor axes = body axes
    sim.AddModelToTask(fsw_task, tam_comm)

    gain = GAIN if GAIN is not None else bdot_gain(INERTIA, n, oe.i)
    bdot = BDotController()
    bdot.ModelTag = "bdot"
    bdot.gain = gain
    bdot.max_dipole = MAX_DIPOLE
    sim.AddModelToTask(fsw_task, bdot)

    # Messages
    tam.stateInMsg.subscribeTo(sc.scStateOutMsg)
    tam.magInMsg.subscribeTo(mag.envOutMsgs[0])
    tam_comm.tamInMsg.subscribeTo(tam.tamDataOutMsg)
    bdot.tamInMsg.subscribeTo(tam_comm.tamOutMsg)
    mtb.mtbCmdInMsg.subscribeTo(bdot.mtbCmdOutMsg)
    mtb.mtbParamsInMsg.subscribeTo(mtb_config_msg)
    mtb.magInMsg.subscribeTo(mag.envOutMsgs[0])

    # Logging (once per control period)
    log_step = macros.sec2nano(CONTROL_PERIOD)
    state_log = sc.scStateOutMsg.recorder(log_step)
    cmd_log = bdot.mtbCmdOutMsg.recorder(log_step)
    field_log = tam_comm.tamOutMsg.recorder(log_step)
    for rec in (state_log, cmd_log, field_log):
        sim.AddModelToTask(fsw_task, rec)

    print(f"Orbit period {period / 60:.1f} min, running {DURATION_ORBITS} orbits "
          f"({duration / 3600:.1f} h), field model: {FIELD_MODEL}, gain k = {gain:.3e} N m s")
    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(duration))
    sim.ExecuteSimulation()

    t = state_log.times() * macros.NANO2SEC
    omega_deg = np.degrees(np.array(state_log.omega_BN_B))
    dipole = np.array(cmd_log.mtbDipoleCmds)[:, :3]
    field_nt = np.array(field_log.tam_B) * 1e9
    report(t, omega_deg, dipole, field_nt, n, period, gain)


def report(t, omega_deg, dipole, field_nt, n, period, gain):
    rate = np.linalg.norm(omega_deg, axis=1)
    plane = np.abs(omega_deg[:, list(PAYLOAD_PLANE_AXES)])
    t_orbits = t / period
    last_orbit = t >= t[-1] - period
    saturated = np.any(np.isclose(np.abs(dipole), MAX_DIPOLE, rtol=1e-6), axis=1)
    duty = np.mean(np.abs(dipole), axis=0) / MAX_DIPOLE

    t_1deg = settled_time(t, rate < 1.0)
    t_req = settled_time(t, np.all(plane < RATE_REQUIREMENT_DEG, axis=1))
    axis_names = "xyz"
    plane_names = "+".join(axis_names[a] for a in PAYLOAD_PLANE_AXES)

    def when(x):
        return "not reached" if x is None else f"{x / 60:.1f} min ({x / period:.2f} orbits)"

    lines = [
        f"Initial |rate|: {rate[0]:.2f} deg/s",
        f"Stays below 1 deg/s total after: {when(t_1deg)}",
        f"Stays below {RATE_REQUIREMENT_DEG} deg/s on {plane_names} (ADCS-1) after: {when(t_req)}",
        "Final-orbit mean |rate| per axis (deg/s): "
        + ", ".join(f"{axis_names[i]} {np.mean(np.abs(omega_deg[last_orbit, i])):.3f}" for i in range(3)),
        f"Final-orbit mean |rate| total: {np.mean(rate[last_orbit]):.3f} deg/s "
        f"(pure B-dot typically settles near 2x orbit rate = {2 * np.degrees(n):.3f} deg/s)",
        f"Time with a rod at its limit: {np.mean(saturated):.0%}",
        "Mean rod use (|m| / max): " + ", ".join(f"{axis_names[i]} {duty[i]:.0%}" for i in range(3)),
        f"Gain k: {gain:.3e} N m s, field model: {FIELD_MODEL}",
    ]
    print("\n".join(lines))

    folder = Path(RESULTS_DIR) / RUN_NAME
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(folder / "detumble.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "wx_dps", "wy_dps", "wz_dps", "rate_dps", "mx_Am2", "my_Am2", "mz_Am2",
                    "Bx_nT", "By_nT", "Bz_nT"])
        for row in zip(t, *omega_deg.T, rate, *dipole.T, *field_nt.T):
            w.writerow([f"{v:.6g}" for v in row])

    fig1, ax = plt.subplots(figsize=(10, 4.5))
    for i in range(3):
        ax.plot(t_orbits, omega_deg[:, i], lw=0.8, label=f"ω{axis_names[i]}")
    ax.plot(t_orbits, rate, color="black", lw=1.0, label="|ω|")
    ax.axhline(RATE_REQUIREMENT_DEG, color="gray", ls="--", lw=0.8)
    ax.axhline(-RATE_REQUIREMENT_DEG, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Time (orbits)")
    ax.set_ylabel("Body rate (deg/s)")
    ax.set_title("TeraLink-1 B-dot detumble", loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig1.tight_layout()
    fig1.savefig(folder / "rates.png", dpi=150)

    fig2, ax = plt.subplots(figsize=(10, 4.5))
    ax.semilogy(t_orbits, rate, color="black", lw=1.0, label="|ω|")
    ax.axhline(RATE_REQUIREMENT_DEG, color="gray", ls="--", lw=0.8, label=f"{RATE_REQUIREMENT_DEG} deg/s")
    ax.axhline(2 * np.degrees(n), color="tab:orange", ls=":", lw=1.0, label="2× orbit rate")
    ax.set_xlabel("Time (orbits)")
    ax.set_ylabel("|ω| (deg/s, log)")
    ax.set_title("Total body rate", loc="left")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")
    fig2.tight_layout()
    fig2.savefig(folder / "rate_total.png", dpi=150)

    fig3, ax = plt.subplots(figsize=(10, 4.5))
    for i in range(3):
        ax.plot(t_orbits, dipole[:, i], lw=0.6, label=f"m{axis_names[i]}")
    ax.axhline(MAX_DIPOLE, color="gray", ls="--", lw=0.8)
    ax.axhline(-MAX_DIPOLE, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Time (orbits)")
    ax.set_ylabel("Commanded dipole (A m²)")
    ax.set_title("Torque rod commands", loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig3.tight_layout()
    fig3.savefig(folder / "dipoles.png", dpi=150)

    print(f"Results saved to {folder}")
    if SHOW_PLOTS:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    run()