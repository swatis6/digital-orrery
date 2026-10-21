'''Integrators that take an equations of motion function f(t, y) and return
a Trajectory so that the results look the same no matter which method makes them

traj = propagate(force_model, y0, (0,86400), "RK4", dt=10) --> would be a fixed step RK4 integrator
traj = propagate(force_model, y0, (0,86400)) --> would be an adaptive DOP853 integrator'''

from __future__ import annotations
from asyncio import events
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Sequence
import numpy as np
from scipy.integrate import solve_ivp
from orrery.constants import R_EARTH

#Trajectory class -------------

EOM = Callable[[float, np.ndarray], np.ndarray] #eom function that takes in time + state vector and returns state deriv vector
Event = Callable[[float, np.ndarray], float] #event function that takes in time + state vector and returns a float value -- if the float value is 0, the event is triggered.

@dataclass
class Trajectory:
    t: np.ndarray # sec since epoch - shape (N,)
    y: np.ndarray # states [r, v] - shape (N, 6)
    stopped_by_event: bool = False
    message: str = ""
    epoch: datetime | None = None #utc time at t=0 needed for ground tracks and sun and etc
    interpolant: Callable | None = None #callable function that takes in time and returns the state vector at that time for adaptive runs only

    @property
    def r(self) -> np.ndarray:
        return self.y[:, :3] #returns only position vector from state vector
    @property
    def v(self) -> np.ndarray:
        return self.y[:, 3:] #returns only velocity vector from state vector

    def at(self, times: np.ndarray) -> np.ndarray:
        '''returns state vector at given times within run. uses integrator output when it's
        available (adaptive), otherwise just uses linear interpolation between stored points.'''
        times = np.atleast_1d(np.asarray(times, dtype=float)) #ensures that time is a 1d array of floats 
        if self.interpolant is not None: #if the interpolant is not None, then use it 
            return self.interpolant(times).T #returns the state vector at the given times
        return np.column_stack([np.interp(times, self.t, self.y[:,i]) for i in range(6)]) #otherwise, use linear interpolation to get the state vector at the times

    def resample(self, n_points: int) -> "Trajectory": 
        '''returns a copy of trajectory with n_points evenly spaced in time for smoother plots and data'''
        times = np.linspace(self.t[0], self.t[-1], n_points) #creates an array of n_points evenly spaced in time between the start and end of the trajectory    
        return Trajectory(times, self.at(times), self.stopped_by_event, self.message, self.epoch, self.interpolant) #returns a new trajectory with the resampled times and state vectors

#Fixed-Step RK4 Integrator -----------

def rk4_step(f: EOM, t: float, y: np.ndarray, h: float) -> np.ndarray:
    '''4th order Runge-Kutta integrator. f is the equations of motion function, t is the current time, 
    y is the current state vector, and h is the time step. returns the next state vector.'''
    k1 = f(t,y)
    k2 = f(t+h/2, y+h*k1/2)
    k3 = f(t+h/2, y+h*k2/2)
    k4 = f(t+h, y+h*k3)
    return y + h*(k1 + 2*k2 + 2*k3 + k4)/6

def _crossed(g_prev: float, g: float, direction: float) -> bool:
    '''helper function to determine if an event has been crossed. 
    g_prev is the previous value of the event function, g is the current value, 
    and direction is the direction of crossing. If direction is negative, only falling 
    crossings are considered. If direction is positive, only rising crossings are 
    considered. If direction is zero, both falling and rising crossings are considered.'''
    falling = g_prev > 0.0 >= g
    rising = g_prev < 0.0 <= g
    if direction < 0:
        return falling
    if direction > 0:
        return rising
    return falling or rising

def propagate_rk4(f: EOM, y0: np.ndarray, t_span: tuple[float, float], #(start time, end time) in seconds since epoch
                dt: float, events: Sequence[Event] | None = None, epoch: datetime | None = None) -> Trajectory:
    '''propagates the state vector using a fixed-step RK4 integrator. f is the equations of motion function,
    y0 is the initial state vector, t_span is a tuple of (start time, end time) in seconds since epoch, 
    dt is the time step in seconds, events is a list of event functions, and epoch is the datetime of the 
    start of the propagation. returns a Trajectory object.'''
    t0, tf = t_span
    y = np.asarray(y0, dtype=float) 
    t = t0
    ts, ys = [t], [y] #initialize lists to store time and state vectors
    g_prev = [ev(t, y) for ev in events] #initialize list to store previous event function values

    n_steps = int(np.ceil((tf-t0)/dt)) #calculate the number of steps needed to reach the final time
    for k in range(n_steps):
        t_next = min(t0+(k+1)*dt, tf) #calculate the next time step, ensuring it does not exceed the final time
        y= rk4_step(f, t, y, t_next - t) #calculate the next state vector using the RK4 integrator
        t = t_next #update current time
        ts.append(t) 
        ys.append(y)
    
    for i, ev in enumerate(events):
            g = ev(t, y) #calculate current value of event function
            if getattr(ev, "terminal", False) and _crossed(g_prev[i], g, getattr(ev, "direction", 0)): #if event terminal/crossed, stop run
                name=getattr(ev, "__name__", "event") #get the name of the event function
                return Trajectory(np.array(ts), np.array(ys), True, f"{name} triggered at t={t:.1f} s", epoch) #return traj with event triggered msg
    g_prev[i] = g #update previous value of event function

    return Trajectory(np.array(ts), np.array(ys), False, "Run completed", epoch) #return traj with run completed msg    

#DOP853 Adaptive Integrator (using scipy) -----------

def propagate_adaptive(
        f:EOM,
        y0: np.ndarray,
        t_span: tuple[float, float],
        method: str = "DOP853",
        rtol: float = 1e-10, #relative tolerance 
        atol: float = 1e-6, #absolute tolerance
        events: Sequence[Event] = (),
        max_step: float = np.inf,
        t_eval: np.ndarray | None = None,
        epoch: datetime | None = None,
) -> Trajectory:
    '''adaptive integration with scipy integrate solve_ivp. default set to DOP853
    (8th order dormand prince).'''
    sol = solve_ivp(f, t_span, np.asarray(y0, dtype=float), method=method, rtol=rtol,
                    atol=atol, events=list(events) or None, max_step=max_step, t_eval=t_eval,
                    dense_output=True)
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return Trajectory(sol.t, sol.y.T, sol.status==1, sol.message, epoch, sol.sol)

#Entrypoint for propagation -----------

def propagate(
        f: EOM,
        y0: np.ndarray,
        t_span: tuple[float,float],
        method: str = "DOP853",
        dt: float | None = None,
        events: Sequence[Event] = (),
        epoch: datetime | None = None,
        **kwargs, #keyword arguments
) -> Trajectory:
    '''Propagate with rk4 (which requires dt) or any solve_ivp method name
    epoch is UTC datetime at t=0, required for ground tracks, sun position, and mag field.'''
    if method.upper() == "RK4":
        if dt is None:
            raise ValueError("RK4 needs a step size: pass dt=...")
        return propagate_rk4(f, y0, t_span, dt, events, epoch)
    return propagate_adaptive(f, y0, t_span, method=method, events=events,
                              epoch=epoch, **kwargs)

#Common Events -----------
def altitude_event(min_altitude: float, body_radius: float = R_EARTH) -> Event:
    """event that stops propagation when altitude falls below min_altitude."""
    def altitude_below(t: float, y: np.ndarray) -> float:
        return np.linalg.norm(y[:3]) - body_radius - min_altitude

    altitude_below.terminal = True
    altitude_below.direction = -1
    return altitude_below