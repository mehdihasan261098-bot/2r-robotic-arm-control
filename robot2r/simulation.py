"""
Closed-loop simulation engine built on :func:`scipy.integrate.solve_ivp` (THEORY.md §8).

The augmented state is ``x = [q (2), qd (2), z (n_z)]`` where ``z`` are the
controller's internal states. The plant (``robot``) and the controller's
internal model may differ, which enables robustness studies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from .controllers import Controller, Reference
from .robot import RobotArm2R

_trapz = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2.x / 1.x


@dataclass
class SimulationResult:
    """Time histories of a closed-loop simulation plus derived quantities.

    Arrays are sampled on the uniform output grid ``t`` (shape ``(N,)``);
    joint quantities have shape ``(N, 2)``.
    """

    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    q_ref: np.ndarray
    qd_ref: np.ndarray
    qdd_ref: np.ndarray
    tau: np.ndarray          # torque applied to the plant (after saturation)
    tau_cmd: np.ndarray      # torque commanded by the controller
    z: np.ndarray            # controller internal states, shape (N, n_z)
    robot: RobotArm2R = field(repr=False)
    controller_name: str = "controller"
    solver_stats: Dict[str, int] = field(default_factory=dict)

    # ----------------------------------------------------------------- derived
    @property
    def e(self) -> np.ndarray:
        """Joint position error ``q_ref - q`` [rad]."""
        return self.q_ref - self.q

    @property
    def ed(self) -> np.ndarray:
        """Joint velocity error [rad/s]."""
        return self.qd_ref - self.qd

    @property
    def ee(self) -> np.ndarray:
        """Actual end-effector position, shape (N, 2)."""
        return self.robot.forward_kinematics(self.q)

    @property
    def ee_ref(self) -> np.ndarray:
        """Desired end-effector position, shape (N, 2)."""
        return self.robot.forward_kinematics(self.q_ref)

    @property
    def ee_error(self) -> np.ndarray:
        """Euclidean end-effector position error [m], shape (N,)."""
        return np.linalg.norm(self.ee_ref - self.ee, axis=1)

    @property
    def energy(self) -> np.ndarray:
        """Total mechanical energy ``T + V`` of the plant [J]."""
        return np.array([self.robot.total_energy(q, qd) for q, qd in zip(self.q, self.qd)])

    def metrics(self, t_start: Optional[float] = None) -> Dict[str, object]:
        """Performance metrics (THEORY.md §8).

        Parameters
        ----------
        t_start : float, optional
            Ignore samples before this time (e.g. to exclude an initial transient).
        """
        m = np.ones_like(self.t, dtype=bool) if t_start is None else self.t >= t_start
        t, e, tau, qd = self.t[m], self.e[m], self.tau[m], self.qd[m]
        ee_err = self.ee_error[m]
        return {
            "controller": self.controller_name,
            "rmse_joint_rad": np.sqrt(np.mean(e**2, axis=0)),
            "max_abs_error_rad": np.max(np.abs(e), axis=0),
            "final_error_rad": np.abs(e[-1]),
            "rmse_ee_m": float(np.sqrt(np.mean(ee_err**2))),
            "max_ee_error_m": float(np.max(ee_err)),
            "rms_torque_Nm": np.sqrt(np.mean(tau**2, axis=0)),
            "peak_torque_Nm": np.max(np.abs(tau), axis=0),
            "control_effort": float(_trapz(np.sum(tau**2, axis=1), t)),
            "mechanical_work_J": float(_trapz(np.abs(np.sum(tau * qd, axis=1)), t)),
        }


class Simulator:
    """Simulate a robot under feedback control while tracking a trajectory.

    Parameters
    ----------
    robot : RobotArm2R
        The *plant* (true dynamics).
    controller : Controller
        Any object implementing :class:`~robot2r.controllers.Controller`.
    trajectory : object, optional
        Anything with ``evaluate(t) -> (q, qd, qdd)`` and ``t0``/``tf``
        attributes (e.g. :class:`~robot2r.trajectory.QuinticTrajectory`).
        If omitted, the reference is the initial configuration (regulation).
    torque_limit : float or array_like, optional
        Symmetric actuator saturation per joint [N m].
    method, rtol, atol, max_step
        Passed to :func:`scipy.integrate.solve_ivp`.

    Examples
    --------
    >>> robot = RobotArm2R()
    >>> traj = QuinticTrajectory([0, 0], [1, -1], 0, 2)
    >>> sim = Simulator(robot, make_controller("ctc", robot), traj)
    >>> res = sim.run(t_final=3.0)
    """

    def __init__(
        self,
        robot: RobotArm2R,
        controller: Controller,
        trajectory=None,
        *,
        torque_limit: Optional[Sequence[float] | float] = None,
        method: str = "RK45",
        rtol: float = 1e-8,
        atol: float = 1e-10,
        max_step: float = 0.01,
    ) -> None:
        self.robot = robot
        self.controller = controller
        self.trajectory = trajectory
        self.torque_limit = (
            None if torque_limit is None
            else np.broadcast_to(np.asarray(torque_limit, dtype=float), (2,)).copy()
        )
        self.method, self.rtol, self.atol, self.max_step = method, rtol, atol, max_step
        self._hold: Optional[np.ndarray] = None  # regulation set-point when no trajectory

    # ----------------------------------------------------------------- helpers
    def reference(self, t: float) -> Reference:
        """Desired state at time ``t``."""
        if self.trajectory is None:
            z = np.zeros(2)
            return Reference(self._hold, z, z)
        q, qd, qdd = self.trajectory.evaluate(t)
        return Reference(q, qd, qdd)

    def saturate(self, tau: np.ndarray) -> np.ndarray:
        """Apply the actuator torque limits (identity if none)."""
        if self.torque_limit is None:
            return tau
        return np.clip(tau, -self.torque_limit, self.torque_limit)

    def _rhs(self, t: float, x: np.ndarray) -> np.ndarray:
        """Right-hand side of the augmented closed-loop ODE."""
        q, qd, z = x[0:2], x[2:4], x[4:]
        ref = self.reference(t)
        tau = self.saturate(self.controller.torque(t, q, qd, ref, z))
        qdd = self.robot.forward_dynamics(q, qd, tau)
        dz = self.controller.state_derivative(t, q, qd, ref, z)
        return np.concatenate([qd, qdd, dz])

    # --------------------------------------------------------------------- run
    def run(
        self,
        t_final: Optional[float] = None,
        *,
        q0: Optional[Sequence[float]] = None,
        qd0: Optional[Sequence[float]] = None,
        t0: Optional[float] = None,
        dt: float = 0.005,
    ) -> SimulationResult:
        """Integrate the closed loop and return a :class:`SimulationResult`.

        Parameters
        ----------
        t_final : float, optional
            End time. Defaults to the trajectory end time.
        q0, qd0 : array_like, optional
            Initial joint state. Defaults to the trajectory's initial state
            (zero initial tracking error).
        t0 : float, optional
            Start time (default: trajectory start or 0).
        dt : float
            Sampling period of the output grid [s].
        """
        traj = self.trajectory
        t0 = (traj.t0 if traj is not None else 0.0) if t0 is None else float(t0)
        if t_final is None:
            if traj is None:
                raise ValueError("t_final is required when no trajectory is given")
            t_final = traj.tf
        if q0 is None:
            if traj is None:
                raise ValueError("q0 is required when no trajectory is given")
            q0 = traj.evaluate(t0)[0]
        if qd0 is None:
            qd0 = traj.evaluate(t0)[1] if traj is not None else np.zeros(2)
        q0, qd0 = np.asarray(q0, dtype=float), np.asarray(qd0, dtype=float)
        self._hold = q0.copy()

        x0 = np.concatenate([q0, qd0, self.controller.initial_state()])
        n_steps = int(round((t_final - t0) / dt))
        t_eval = t0 + dt * np.arange(n_steps + 1)

        sol = solve_ivp(
            self._rhs, (t0, t_eval[-1]), x0, method=self.method, t_eval=t_eval,
            rtol=self.rtol, atol=self.atol, max_step=self.max_step,
        )
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        X = sol.y.T
        q, qd, z = X[:, 0:2], X[:, 2:4], X[:, 4:]

        # Re-evaluate reference and torque on the output grid (controller is stateless
        # apart from z, so this reproduces exactly what the plant received).
        q_ref = np.empty_like(q)
        qd_ref, qdd_ref = np.empty_like(q), np.empty_like(q)
        tau_cmd = np.empty_like(q)
        for k, tk in enumerate(sol.t):
            ref = self.reference(tk)
            q_ref[k], qd_ref[k], qdd_ref[k] = ref.q, ref.qd, ref.qdd
            tau_cmd[k] = self.controller.torque(tk, q[k], qd[k], ref, z[k])
        tau = np.array([self.saturate(u) for u in tau_cmd])

        return SimulationResult(
            t=sol.t, q=q, qd=qd, q_ref=q_ref, qd_ref=qd_ref, qdd_ref=qdd_ref,
            tau=tau, tau_cmd=tau_cmd, z=z, robot=self.robot,
            controller_name=self.controller.name,
            solver_stats={"nfev": int(sol.nfev), "status": int(sol.status)},
        )


def format_metrics_table(results: Sequence[SimulationResult], t_start: Optional[float] = None) -> str:
    """Markdown table comparing the metrics of several simulations."""
    rows = [r.metrics(t_start) for r in results]
    deg = 180.0 / np.pi
    header = (
        "| Controller | RMSE θ1 [deg] | RMSE θ2 [deg] | Max \\|e\\| θ1 [deg] | Max \\|e\\| θ2 [deg] "
        "| EE RMSE [mm] | EE max [mm] | RMS τ1 [N·m] | RMS τ2 [N·m] | Peak τ1 [N·m] "
        "| Peak τ2 [N·m] | Effort ∫τᵀτ dt |"
    )
    sep = "|" + "---|" * 13
    lines = [header, sep]
    for m in rows:
        # Errors span many orders of magnitude (CTC vs PID): use significant figures.
        lines.append(
            f"| {m['controller']} "
            f"| {m['rmse_joint_rad'][0]*deg:.3g} | {m['rmse_joint_rad'][1]*deg:.3g} "
            f"| {m['max_abs_error_rad'][0]*deg:.3g} | {m['max_abs_error_rad'][1]*deg:.3g} "
            f"| {m['rmse_ee_m']*1e3:.3g} | {m['max_ee_error_m']*1e3:.3g} "
            f"| {m['rms_torque_Nm'][0]:.3f} | {m['rms_torque_Nm'][1]:.3f} "
            f"| {m['peak_torque_Nm'][0]:.3f} | {m['peak_torque_Nm'][1]:.3f} "
            f"| {m['control_effort']:.1f} |"
        )
    return "\n".join(lines)
