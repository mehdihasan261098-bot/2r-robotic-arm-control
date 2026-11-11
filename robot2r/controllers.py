"""
Joint-space controllers (THEORY.md §7).

Every controller implements the :class:`Controller` interface:

* ``torque(t, q, qd, ref, z)`` returns the commanded joint torque;
* ``n_states`` / ``state_derivative(...)`` / ``initial_state()`` describe
  internal states (e.g. the PID integrator). The simulator appends them to the
  plant state so that ``solve_ivp`` integrates them consistently.

Controllers are therefore *pure functions* of time and state, which is
required for correct use with adaptive ODE solvers.

Switching controllers is a one-liner, see :func:`make_controller`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

from .robot import RobotArm2R

Gain = Union[float, Sequence[float], np.ndarray]


@dataclass(frozen=True)
class Reference:
    """Desired joint position, velocity and acceleration at one instant."""

    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray


def _gain(k: Gain, n: int = 2) -> np.ndarray:
    """Broadcast a scalar or vector gain to a length-``n`` array (diagonal gain matrix)."""
    k = np.broadcast_to(np.asarray(k, dtype=float), (n,)).copy()
    if np.any(k < 0):
        raise ValueError("gains must be non-negative")
    return k


class Controller(ABC):
    """Abstract base class for joint-space controllers."""

    #: Human-readable name used in plots and tables.
    name: str = "controller"
    #: Number of internal states integrated by the simulator.
    n_states: int = 0

    @abstractmethod
    def torque(
        self, t: float, q: np.ndarray, qd: np.ndarray, ref: Reference, z: np.ndarray
    ) -> np.ndarray:
        """Commanded joint torque."""

    def state_derivative(
        self, t: float, q: np.ndarray, qd: np.ndarray, ref: Reference, z: np.ndarray
    ) -> np.ndarray:
        """Time derivative of the internal states (empty by default)."""
        return np.zeros(0)

    def initial_state(self) -> np.ndarray:
        """Initial value of the internal states."""
        return np.zeros(self.n_states)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class ZeroTorqueController(Controller):
    """Applies no torque. Used for passive (free-swinging) simulations."""

    name = "Passive"

    def torque(self, t, q, qd, ref, z):
        return np.zeros(2)


class PIDController(Controller):
    """Decentralised (independent-joint) PID controller, THEORY.md §7.1.

    ``tau = Kp e + Ki int(e) + Kd e' [+ G_hat(q)]``

    Parameters
    ----------
    kp, ki, kd : float or array_like, shape (2,)
        Diagonal gains.
    integral_limit : float or array_like, optional
        Anti-windup bound on ``|int(e)|`` per joint [rad s]. Integration stops
        when the bound is reached and the error would push further out
        (conditional integration).
    gravity_model : RobotArm2R, optional
        If given, adds model-based gravity compensation ``G_hat(q)`` ("PID+G").
    name : str, optional
    """

    n_states = 2

    def __init__(
        self,
        kp: Gain,
        ki: Gain,
        kd: Gain,
        *,
        integral_limit: Optional[Gain] = None,
        gravity_model: Optional[RobotArm2R] = None,
        name: Optional[str] = None,
    ) -> None:
        self.kp, self.ki, self.kd = _gain(kp), _gain(ki), _gain(kd)
        self.integral_limit = None if integral_limit is None else _gain(integral_limit)
        self.gravity_model = gravity_model
        self.name = name or ("PID+G" if gravity_model is not None else "PID")

    def torque(self, t, q, qd, ref, z):
        e = ref.q - q
        ed = ref.qd - qd
        tau = self.kp * e + self.ki * z + self.kd * ed
        if self.gravity_model is not None:
            tau = tau + self.gravity_model.gravity_vector(q)
        return tau

    def state_derivative(self, t, q, qd, ref, z):
        e = ref.q - q
        if self.integral_limit is None:
            return e
        # Conditional integration: freeze a saturated integrator unless the error
        # drives it back towards zero.
        saturated = (np.abs(z) >= self.integral_limit) & (np.sign(e) == np.sign(z))
        return np.where(saturated, 0.0, e)


class ComputedTorqueController(Controller):
    """Computed-torque (inverse-dynamics / feedback-linearising) controller, THEORY.md §7.2.

    ``tau = M_hat(q) (qdd_d + Kd e' + Kp e) + C_hat(q, q') q' + G_hat(q)``

    With a perfect model the closed-loop error obeys ``e'' + Kd e' + Kp e = 0``.

    Parameters
    ----------
    model : RobotArm2R
        The controller's *internal* model. Pass a robot with different
        parameters than the plant to study model mismatch.
    kp, kd : float or array_like, shape (2,)
        Outer-loop PD gains (units 1/s^2 and 1/s).
    ki : float or array_like, shape (2,), optional
        Outer-loop integral gain (1/s^3). When non-zero the outer loop becomes
        ``v = qdd_d + Kd e' + Kp e + Ki int(e)`` ("CTC+I"), which removes the
        steady-state error caused by constant model errors such as an
        unmodelled payload.
    name : str, optional
    """

    def __init__(self, model: RobotArm2R, kp: Gain, kd: Gain, ki: Gain = 0.0, *,
                 name: Optional[str] = None) -> None:
        self.model = model
        self.kp, self.kd, self.ki = _gain(kp), _gain(kd), _gain(ki)
        self.n_states = 2 if np.any(self.ki > 0) else 0
        self.name = name or ("CTC+I" if self.n_states else "CTC")

    @classmethod
    def from_bandwidth(
        cls, model: RobotArm2R, wn: float = 10.0, zeta: float = 1.0, integral: bool = False, **kwargs
    ) -> "ComputedTorqueController":
        """Pole placement for the linearised error dynamics.

        * ``integral=False``: ``Kp = wn^2``, ``Kd = 2 zeta wn`` → poles of ``s^2 + 2 zeta wn s + wn^2``.
        * ``integral=True``: triple pole at ``-wn``, i.e. ``(s + wn)^3`` →
          ``Kd = 3 wn``, ``Kp = 3 wn^2``, ``Ki = wn^3`` (``zeta`` is ignored).
        """
        if integral:
            return cls(model, kp=3.0 * wn**2, kd=3.0 * wn, ki=wn**3, **kwargs)
        return cls(model, kp=wn**2, kd=2.0 * zeta * wn, **kwargs)

    def torque(self, t, q, qd, ref, z):
        e = ref.q - q
        ed = ref.qd - qd
        v = ref.qdd + self.kd * ed + self.kp * e          # outer loop: desired acceleration
        if self.n_states:
            v = v + self.ki * z
        M, C, G = self.model.dynamics(q, qd)                # inner loop: linearisation
        return M @ v + C @ qd + G + self.model.friction_torque(qd)

    def state_derivative(self, t, q, qd, ref, z):
        return (ref.q - q) if self.n_states else np.zeros(0)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
#: Default gains used throughout the examples. PID gains were chosen to give a
#: closed-loop bandwidth comparable to the CTC outer loop (wn = 10 rad/s) at a
#: mid-workspace configuration; see README "Gain selection".
DEFAULT_PID_GAINS = dict(kp=(200.0, 100.0), ki=(100.0, 50.0), kd=(30.0, 10.0), integral_limit=(1.0, 1.0))
DEFAULT_CTC_GAINS = dict(wn=10.0, zeta=1.0)


def make_controller(kind: str, model: RobotArm2R, **overrides) -> Controller:
    """Create a controller by name.

    Parameters
    ----------
    kind : {"pid", "pidg", "ctc", "ctci", "passive"}
        ``"pidg"`` is PID with model-based gravity compensation, ``"ctci"`` is
        computed-torque control with an integral term in the outer loop.
    model : RobotArm2R
        Model used by model-based controllers (CTC, PID+G).
    **overrides
        Keyword arguments replacing the defaults, e.g. ``kp=(300, 150)`` or ``wn=15``.
    """
    kind = kind.lower()
    if kind == "pid":
        return PIDController(**{**DEFAULT_PID_GAINS, **overrides})
    if kind in ("pidg", "pid+g"):
        return PIDController(**{**DEFAULT_PID_GAINS, **overrides}, gravity_model=model)
    if kind in ("ctc", "ctci", "ctc+i"):
        if "kp" in overrides or "kd" in overrides:  # explicit gains
            return ComputedTorqueController(model, **overrides)
        params = {**DEFAULT_CTC_GAINS, "integral": kind != "ctc", **overrides}
        return ComputedTorqueController.from_bandwidth(model, **params)
    if kind in ("passive", "none", "zero"):
        return ZeroTorqueController()
    raise ValueError(f"Unknown controller '{kind}'. Choose from pid, pidg, ctc, ctci, passive.")
