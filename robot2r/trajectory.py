"""
Joint-space trajectory generation with quintic polynomials (THEORY.md §6).

All trajectory classes share one interface::

    q, qd, qdd = traj.evaluate(t)   # t: scalar -> arrays of shape (n,)
                                    # t: (N,)   -> arrays of shape (N, n)

and expose ``t0`` / ``tf``. Outside ``[t0, tf]`` the trajectory holds its end
position with zero velocity and acceleration (a regulation set-point).
"""
from __future__ import annotations

import warnings
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .robot import Elbow, RobotArm2R, SingularityWarning

ArrayOrStr = Union[np.ndarray, Sequence[Sequence[float]], str]


class QuinticTrajectory:
    """Single quintic segment between two joint-space states.

    Solves the 6x6 boundary-value system for each joint independently, so
    arbitrary boundary velocities and accelerations are supported.

    Parameters
    ----------
    q0, qf : array_like, shape (n,)
        Initial and final positions.
    t0, tf : float
        Start and end times, ``tf > t0``.
    v0, vf, a0, af : array_like, shape (n,), optional
        Boundary velocities and accelerations (default zero: rest-to-rest).
    """

    def __init__(
        self,
        q0: Sequence[float],
        qf: Sequence[float],
        t0: float,
        tf: float,
        v0: Optional[Sequence[float]] = None,
        vf: Optional[Sequence[float]] = None,
        a0: Optional[Sequence[float]] = None,
        af: Optional[Sequence[float]] = None,
    ) -> None:
        if tf <= t0:
            raise ValueError(f"tf ({tf}) must be greater than t0 ({t0})")
        self.q0 = np.atleast_1d(np.asarray(q0, dtype=float))
        self.qf = np.atleast_1d(np.asarray(qf, dtype=float))
        if self.q0.shape != self.qf.shape:
            raise ValueError("q0 and qf must have the same shape")
        n = self.q0.size
        z = np.zeros(n)
        self.v0 = z if v0 is None else np.asarray(v0, dtype=float)
        self.vf = z if vf is None else np.asarray(vf, dtype=float)
        self.a0 = z if a0 is None else np.asarray(a0, dtype=float)
        self.af = z if af is None else np.asarray(af, dtype=float)
        self.t0, self.tf = float(t0), float(tf)
        self.n_joints = n

        T = self.tf - self.t0
        A = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 2, 0, 0, 0],
                [1, T, T**2, T**3, T**4, T**5],
                [0, 1, 2 * T, 3 * T**2, 4 * T**3, 5 * T**4],
                [0, 0, 2, 6 * T, 12 * T**2, 20 * T**3],
            ],
            dtype=float,
        )
        b = np.vstack([self.q0, self.v0, self.a0, self.qf, self.vf, self.af])  # (6, n)
        #: Polynomial coefficients a_0..a_5 (rows) for each joint (columns).
        self.coeffs = np.linalg.solve(A, b)

    @property
    def duration(self) -> float:
        return self.tf - self.t0

    def evaluate(self, t: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Desired position, velocity and acceleration at time(s) ``t``."""
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        tau = np.clip(t_arr - self.t0, 0.0, self.duration)[:, None]  # (N, 1)
        a = self.coeffs
        k = np.arange(6)
        P = tau ** k                                            # (N, 6) : tau^k
        dP = np.where(k > 0, k * tau ** np.maximum(k - 1, 0), 0.0)
        ddP = np.where(k > 1, k * (k - 1) * tau ** np.maximum(k - 2, 0), 0.0)
        q, qd, qdd = P @ a, dP @ a, ddP @ a

        # Hold the boundary positions outside the active interval.
        before = (t_arr < self.t0)[:, None]
        after = (t_arr > self.tf)[:, None]
        q = np.where(before, self.q0, np.where(after, self.qf, q))
        qd = np.where(before | after, 0.0, qd)
        qdd = np.where(before | after, 0.0, qdd)

        if np.ndim(t) == 0:
            return q[0], qd[0], qdd[0]
        return q, qd, qdd

    __call__ = evaluate


class PiecewiseQuinticTrajectory:
    """C^2 trajectory through joint-space via points, one quintic per segment.

    Parameters
    ----------
    waypoints : array_like, shape (K, n)
        Joint positions at the via points (K >= 2).
    times : array_like, shape (K,)
        Strictly increasing arrival times.
    via_velocities : {"heuristic", "zero"} or array_like, shape (K, n)
        ``"heuristic"`` — Craig's rule: average of adjacent segment slopes when
        they have the same sign, else zero (THEORY.md §6).
        ``"zero"`` — stop at every via point.
        An explicit array is used as given (first/last rows should be zero for rest-to-rest).

    Via-point accelerations are zero, so acceleration is continuous.
    """

    def __init__(
        self,
        waypoints: Sequence[Sequence[float]],
        times: Sequence[float],
        via_velocities: ArrayOrStr = "heuristic",
    ) -> None:
        W = np.asarray(waypoints, dtype=float)
        T = np.asarray(times, dtype=float)
        if W.ndim != 2 or W.shape[0] < 2:
            raise ValueError("waypoints must have shape (K, n) with K >= 2")
        if T.shape != (W.shape[0],) or np.any(np.diff(T) <= 0):
            raise ValueError("times must be strictly increasing with one entry per waypoint")

        if isinstance(via_velocities, str):
            V = self._via_velocities(W, T, via_velocities)
        else:
            V = np.asarray(via_velocities, dtype=float)
            if V.shape != W.shape:
                raise ValueError("via_velocities must have the same shape as waypoints")

        self.waypoints, self.times, self.via_velocities = W, T, V
        self.segments = [
            QuinticTrajectory(W[k], W[k + 1], T[k], T[k + 1], v0=V[k], vf=V[k + 1])
            for k in range(len(T) - 1)
        ]
        self.t0, self.tf = float(T[0]), float(T[-1])
        self.n_joints = W.shape[1]

    @staticmethod
    def _via_velocities(W: np.ndarray, T: np.ndarray, mode: str) -> np.ndarray:
        V = np.zeros_like(W)
        if mode == "zero":
            return V
        if mode != "heuristic":
            raise ValueError("via_velocities must be 'heuristic', 'zero' or an array")
        slopes = np.diff(W, axis=0) / np.diff(T)[:, None]   # (K-1, n)
        for k in range(1, len(T) - 1):
            d_prev, d_next = slopes[k - 1], slopes[k]
            same_sign = np.sign(d_prev) == np.sign(d_next)
            V[k] = np.where(same_sign, 0.5 * (d_prev + d_next), 0.0)
        return V

    def evaluate(self, t: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Desired position, velocity and acceleration at time(s) ``t``."""
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        # Segment index for every sample (clamped to the first / last segment).
        idx = np.clip(np.searchsorted(self.times, t_arr, side="right") - 1, 0, len(self.segments) - 1)
        q = np.empty((t_arr.size, self.n_joints))
        qd, qdd = np.empty_like(q), np.empty_like(q)
        for k in np.unique(idx):
            mask = idx == k
            q[mask], qd[mask], qdd[mask] = self.segments[k].evaluate(t_arr[mask])
        if np.ndim(t) == 0:
            return q[0], qd[0], qdd[0]
        return q, qd, qdd

    __call__ = evaluate


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def joint_trajectory_from_cartesian(
    robot: RobotArm2R,
    points: Sequence[Sequence[float]],
    times: Sequence[float],
    elbow: Elbow = "up",
    via_velocities: ArrayOrStr = "heuristic",
) -> PiecewiseQuinticTrajectory:
    """Build a joint-space trajectory through Cartesian via points.

    Each point is mapped through the analytical IK on a single elbow branch;
    ``theta1`` is unwrapped so the joint path is continuous. Note that the
    end-effector path *between* via points is not a straight line, because
    interpolation happens in joint space.
    """
    Q = np.array([robot.inverse_kinematics(p, elbow=elbow) for p in points])
    Q[:, 0] = np.unwrap(Q[:, 0])
    return PiecewiseQuinticTrajectory(Q, times, via_velocities=via_velocities)


def check_trajectory_singularities(
    robot: RobotArm2R,
    trajectory,
    *,
    tol: float = 0.05,
    n_samples: int = 2000,
    warn: bool = True,
) -> float:
    """Scan a joint trajectory for near-singular configurations.

    Returns the minimum singularity index ``|sin(theta2)|`` along the path and
    issues a :class:`SingularityWarning` if it falls below ``tol``.
    """
    ts = np.linspace(trajectory.t0, trajectory.tf, n_samples)
    q, _, _ = trajectory.evaluate(ts)
    idx = np.abs(np.sin(q[:, 1]))
    k = int(np.argmin(idx))
    if warn and idx[k] < tol:
        warnings.warn(
            f"Trajectory passes near a kinematic singularity at t = {ts[k]:.3f} s "
            f"(|sin theta2| = {idx[k]:.3e} < {tol}). Joint-space control is unaffected, "
            "but Cartesian quantities (J^-1) are ill-conditioned there.",
            SingularityWarning,
            stacklevel=2,
        )
    return float(idx[k])
