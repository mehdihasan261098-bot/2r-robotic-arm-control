"""
Kinematic and dynamic model of a planar 2R serial manipulator.

Conventions (see ``docs/THEORY.md`` §1)
---------------------------------------
* ``q = [theta1, theta2]``: ``theta1`` is measured from the world x-axis,
  ``theta2`` is the angle of link 2 *relative to* link 1.
* The world y-axis points up; gravity acts along -y.
* Angles and torques are positive counter-clockwise.

Every method accepts plain sequences or NumPy arrays. Kinematic functions
(:meth:`RobotArm2R.joint_positions`, :meth:`RobotArm2R.forward_kinematics`)
are vectorised over leading dimensions, i.e. ``q`` may have shape ``(2,)`` or
``(N, 2)``. Dynamics functions operate on a single configuration.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Dict, Literal, Optional, Sequence, Tuple

import numpy as np

Elbow = Literal["up", "down"]


class WorkspaceError(ValueError):
    """Raised when a Cartesian target lies outside the reachable workspace."""


class SingularityWarning(UserWarning):
    """Issued when an operation is evaluated at or near a kinematic singularity."""


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RobotParams:
    """Physical parameters of the 2R arm.

    Links are uniform slender rods: COM at mid-length and
    ``I = m l^2 / 12`` about the COM. An optional point-mass ``payload`` at the
    tip of link 2 and viscous joint friction are provided for robustness
    studies; both are zero in the base model.

    Attributes
    ----------
    l1, l2 : float
        Link lengths [m].
    m1, m2 : float
        Link masses [kg].
    g : float
        Gravitational acceleration [m/s^2].
    payload : float
        Point mass rigidly attached at the end-effector [kg].
    viscous_friction : tuple of float
        Viscous friction coefficients ``(b1, b2)`` [N m s/rad].
    """

    l1: float = 1.0
    l2: float = 0.8
    m1: float = 1.0
    m2: float = 0.8
    g: float = 9.81
    payload: float = 0.0
    viscous_friction: Tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        for name in ("l1", "l2", "m1", "m2"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.payload < 0.0 or self.g < 0.0:
            raise ValueError("payload and g must be non-negative")

    # Link 1 -----------------------------------------------------------------
    @property
    def lc1(self) -> float:
        """Distance from joint 1 to the COM of link 1 [m]."""
        return 0.5 * self.l1

    @property
    def I1(self) -> float:
        """Inertia of link 1 about its COM [kg m^2]."""
        return self.m1 * self.l1**2 / 12.0

    # Link 2 (rod + optional tip payload, treated as one rigid body) -------------
    @property
    def m2_eff(self) -> float:
        """Total mass of the distal body (link 2 + payload) [kg]."""
        return self.m2 + self.payload

    @property
    def lc2(self) -> float:
        """Distance from joint 2 to the COM of the distal body [m]."""
        return (self.m2 * 0.5 * self.l2 + self.payload * self.l2) / self.m2_eff

    @property
    def I2(self) -> float:
        """Inertia of the distal body about its own COM (parallel-axis theorem) [kg m^2]."""
        lc = self.lc2
        return (
            self.m2 * self.l2**2 / 12.0
            + self.m2 * (0.5 * self.l2 - lc) ** 2
            + self.payload * (self.l2 - lc) ** 2
        )

    def with_payload(self, payload: float) -> "RobotParams":
        """Return a copy with a different tip payload."""
        return replace(self, payload=payload)


# --------------------------------------------------------------------------- #
# Robot model
# --------------------------------------------------------------------------- #
class RobotArm2R:
    """Planar two-link revolute (2R) robot arm.

    Provides forward/inverse kinematics, the geometric Jacobian with singularity
    analysis, the Lagrangian dynamic model ``M(q) q'' + C(q, q') q' + G(q) = tau``
    and a simple Matplotlib drawing routine.

    Parameters
    ----------
    params : RobotParams, optional
        Physical parameters. Defaults to the project robot
        (l1 = 1.0, l2 = 0.8, m1 = 1.0, m2 = 0.8, g = 9.81).
    singularity_tol : float
        Threshold on ``|sin(theta2)|`` below which a configuration is
        reported as singular.

    Examples
    --------
    >>> robot = RobotArm2R()
    >>> q = robot.inverse_kinematics([1.2, 0.6], elbow="up")
    >>> np.allclose(robot.forward_kinematics(q), [1.2, 0.6])
    True
    """

    def __init__(self, params: Optional[RobotParams] = None, singularity_tol: float = 1e-3) -> None:
        self.params = params if params is not None else RobotParams()
        self.singularity_tol = float(singularity_tol)

        p = self.params
        self.l1, self.l2, self.g = p.l1, p.l2, p.g
        m1, m2 = p.m1, p.m2_eff
        lc1, lc2, I1, I2 = p.lc1, p.lc2, p.I1, p.I2

        # Lumped inertial constants (THEORY.md §5.4)
        self.alpha = m1 * lc1**2 + I1 + m2 * (p.l1**2 + lc2**2) + I2
        self.beta = m2 * p.l1 * lc2
        self.delta = m2 * lc2**2 + I2
        # Gravity constants (THEORY.md §5.8)
        self.g1 = (m1 * lc1 + m2 * p.l1) * p.g
        self.g2 = m2 * lc2 * p.g
        self.friction = np.asarray(p.viscous_friction, dtype=float)

        self.reach_max = p.l1 + p.l2
        self.reach_min = abs(p.l1 - p.l2)

    def __repr__(self) -> str:
        p = self.params
        return (
            f"RobotArm2R(l1={p.l1}, l2={p.l2}, m1={p.m1}, m2={p.m2}, "
            f"payload={p.payload}, g={p.g})"
        )

    # ======================================================================= #
    # Kinematics
    # ======================================================================= #
    def joint_positions(self, q: Sequence[float] | np.ndarray) -> np.ndarray:
        """Positions of base, elbow and end-effector.

        Parameters
        ----------
        q : array_like, shape (..., 2)

        Returns
        -------
        np.ndarray, shape (..., 3, 2)
            ``[..., 0, :]`` base, ``[..., 1, :]`` elbow, ``[..., 2, :]`` end-effector.
        """
        q = np.asarray(q, dtype=float)
        t1, t12 = q[..., 0], q[..., 0] + q[..., 1]
        elbow = np.stack([self.l1 * np.cos(t1), self.l1 * np.sin(t1)], axis=-1)
        ee = elbow + np.stack([self.l2 * np.cos(t12), self.l2 * np.sin(t12)], axis=-1)
        base = np.zeros_like(elbow)
        return np.stack([base, elbow, ee], axis=-2)

    def forward_kinematics(
        self, q: Sequence[float] | np.ndarray, return_orientation: bool = False
    ) -> np.ndarray:
        """End-effector pose (THEORY.md §2).

        Parameters
        ----------
        q : array_like, shape (..., 2)
        return_orientation : bool
            If True, append the EE orientation ``phi = theta1 + theta2``.

        Returns
        -------
        np.ndarray, shape (..., 2) or (..., 3)
            ``[x, y]`` or ``[x, y, phi]``.
        """
        q = np.asarray(q, dtype=float)
        ee = self.joint_positions(q)[..., 2, :]
        if return_orientation:
            phi = q[..., 0] + q[..., 1]
            return np.concatenate([ee, phi[..., None]], axis=-1)
        return ee

    def in_workspace(self, p: Sequence[float], margin: float = 0.0) -> bool:
        """True if the point lies in the reachable annulus shrunk by ``margin``."""
        r = float(np.hypot(p[0], p[1]))
        return self.reach_min + margin <= r <= self.reach_max - margin

    def inverse_kinematics(
        self,
        p: Sequence[float],
        elbow: Elbow = "up",
        *,
        boundary_tol: float = 1e-9,
    ) -> np.ndarray:
        """Analytical inverse kinematics (THEORY.md §3).

        Parameters
        ----------
        p : array_like, shape (2,)
            Target end-effector position ``[x, y]``.
        elbow : {"up", "down"}
            Solution branch. ``"up"`` returns ``theta2 <= 0``, ``"down"`` returns ``theta2 >= 0``.
        boundary_tol : float
            Targets with ``1 < |cos(theta2)| <= 1 + boundary_tol`` are treated as
            lying on the boundary (round-off) instead of raising.

        Returns
        -------
        np.ndarray, shape (2,)
            Joint angles; ``theta1`` is wrapped to ``(-pi, pi]``.

        Raises
        ------
        WorkspaceError
            If the target is outside the reachable workspace.

        Warns
        -----
        SingularityWarning
            If the target is (numerically) on the workspace boundary, where the
            two branches coincide and the Jacobian is singular.
        """
        if elbow not in ("up", "down"):
            raise ValueError("elbow must be 'up' or 'down'")
        x, y = float(p[0]), float(p[1])
        l1, l2 = self.l1, self.l2
        r2 = x * x + y * y

        c2 = (r2 - l1**2 - l2**2) / (2.0 * l1 * l2)
        if abs(c2) > 1.0 + boundary_tol:
            raise WorkspaceError(
                f"Target ({x:.4f}, {y:.4f}) at r = {np.sqrt(r2):.4f} m is outside the "
                f"workspace [{self.reach_min:.3f}, {self.reach_max:.3f}] m"
            )
        c2 = float(np.clip(c2, -1.0, 1.0))
        s2_mag = np.sqrt(max(0.0, 1.0 - c2 * c2))
        if s2_mag < self.singularity_tol:
            warnings.warn(
                f"IK target ({x:.4f}, {y:.4f}) is on the workspace boundary: "
                "elbow-up and elbow-down solutions coincide (singular configuration).",
                SingularityWarning,
                stacklevel=2,
            )
        s2 = -s2_mag if elbow == "up" else s2_mag

        theta2 = np.arctan2(s2, c2)
        theta1 = np.arctan2(y, x) - np.arctan2(l2 * s2, l1 + l2 * c2)
        theta1 = _wrap_to_pi(theta1)
        return np.array([theta1, theta2])

    def inverse_kinematics_all(self, p: Sequence[float]) -> Dict[str, np.ndarray]:
        """Both IK branches as ``{"up": q_up, "down": q_down}``."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SingularityWarning)
            return {b: self.inverse_kinematics(p, elbow=b) for b in ("up", "down")}

    # ----------------------------------------------------------------------- #
    # Jacobian & singularities
    # ----------------------------------------------------------------------- #
    def jacobian(self, q: Sequence[float]) -> np.ndarray:
        """Positional Jacobian ``J_v = d[x, y]/dq`` (2x2), THEORY.md §4.1."""
        t1, t12 = q[0], q[0] + q[1]
        s1, c1, s12, c12 = np.sin(t1), np.cos(t1), np.sin(t12), np.cos(t12)
        l1, l2 = self.l1, self.l2
        return np.array(
            [
                [-l1 * s1 - l2 * s12, -l2 * s12],
                [l1 * c1 + l2 * c12, l2 * c12],
            ]
        )

    def geometric_jacobian(self, q: Sequence[float]) -> np.ndarray:
        """Full 6x2 geometric Jacobian mapping ``q'`` to ``[v; omega]``."""
        J = np.zeros((6, 2))
        J[0:2, :] = self.jacobian(q)
        J[5, :] = 1.0  # both joint axes are parallel to world z
        return J

    def jacobian_dot(self, q: Sequence[float], qd: Sequence[float]) -> np.ndarray:
        """Time derivative of the positional Jacobian, ``dJ_v/dt``."""
        t1, t12 = q[0], q[0] + q[1]
        w1, w12 = qd[0], qd[0] + qd[1]
        s1, c1, s12, c12 = np.sin(t1), np.cos(t1), np.sin(t12), np.cos(t12)
        l1, l2 = self.l1, self.l2
        return np.array(
            [
                [-l1 * c1 * w1 - l2 * c12 * w12, -l2 * c12 * w12],
                [-l1 * s1 * w1 - l2 * s12 * w12, -l2 * s12 * w12],
            ]
        )

    def ee_velocity(self, q: Sequence[float], qd: Sequence[float]) -> np.ndarray:
        """End-effector linear velocity ``J_v(q) q'``."""
        return self.jacobian(q) @ np.asarray(qd, dtype=float)

    def manipulability(self, q: Sequence[float]) -> float:
        """Yoshikawa manipulability ``sqrt(det(J J^T)) = l1 l2 |sin(theta2)|``."""
        return float(self.l1 * self.l2 * abs(np.sin(q[1])))

    def singularity_index(self, q: Sequence[float]) -> float:
        """Normalised distance to singularity, ``|sin(theta2)|`` in [0, 1]."""
        return float(abs(np.sin(q[1])))

    def is_singular(self, q: Sequence[float], tol: Optional[float] = None) -> bool:
        """True if ``|sin(theta2)| < tol`` (arm fully stretched or folded)."""
        tol = self.singularity_tol if tol is None else tol
        return self.singularity_index(q) < tol

    def jacobian_condition_number(self, q: Sequence[float]) -> float:
        """2-norm condition number of ``J_v`` (``inf`` at a singularity)."""
        if self.singularity_index(q) == 0.0:
            return float("inf")
        return float(np.linalg.cond(self.jacobian(q)))

    def inverse_velocity(
        self,
        q: Sequence[float],
        xdot: Sequence[float],
        *,
        damping_threshold: float = 0.05,
        max_damping: float = 0.05,
    ) -> np.ndarray:
        """Joint velocity producing EE velocity ``xdot`` (singularity-robust).

        Uses the exact inverse away from singularities and a damped
        least-squares (DLS) inverse with smoothly increasing damping when
        ``|sin(theta2)| < damping_threshold`` (THEORY.md §4.2).
        """
        J = self.jacobian(q)
        xdot = np.asarray(xdot, dtype=float)
        s = self.singularity_index(q)
        if s >= damping_threshold:
            return np.linalg.solve(J, xdot)
        lam2 = max_damping**2 * (1.0 - (s / damping_threshold) ** 2)
        warnings.warn(
            f"inverse_velocity near singularity (|sin theta2| = {s:.2e}); using DLS with "
            f"lambda = {np.sqrt(lam2):.3g}",
            SingularityWarning,
            stacklevel=2,
        )
        return J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(2), xdot)

    # ======================================================================= #
    # Dynamics  (THEORY.md §5)
    # ======================================================================= #
    def mass_matrix(self, q: Sequence[float]) -> np.ndarray:
        """Inertia matrix ``M(q)`` (symmetric positive definite, 2x2)."""
        c2 = np.cos(q[1])
        m11 = self.alpha + 2.0 * self.beta * c2
        m12 = self.delta + self.beta * c2
        return np.array([[m11, m12], [m12, self.delta]])

    def coriolis_matrix(self, q: Sequence[float], qd: Sequence[float]) -> np.ndarray:
        """Coriolis/centrifugal matrix ``C(q, q')`` from Christoffel symbols.

        With this choice ``M' - 2C`` is skew-symmetric (THEORY.md §5.7).
        """
        h = -self.beta * np.sin(q[1])
        return np.array([[h * qd[1], h * (qd[0] + qd[1])], [-h * qd[0], 0.0]])

    def gravity_vector(self, q: Sequence[float]) -> np.ndarray:
        """Gravity torque vector ``G(q) = dV/dq``."""
        c1, c12 = np.cos(q[0]), np.cos(q[0] + q[1])
        return np.array([self.g1 * c1 + self.g2 * c12, self.g2 * c12])

    def friction_torque(self, qd: Sequence[float]) -> np.ndarray:
        """Viscous friction torque ``B q'`` (zero in the base model)."""
        return self.friction * np.asarray(qd, dtype=float)

    def dynamics(
        self, q: Sequence[float], qd: Sequence[float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convenience accessor returning ``(M, C, G)``."""
        return self.mass_matrix(q), self.coriolis_matrix(q, qd), self.gravity_vector(q)

    def inverse_dynamics(
        self, q: Sequence[float], qd: Sequence[float], qdd: Sequence[float]
    ) -> np.ndarray:
        """Joint torques required for a given motion: ``M q'' + C q' + G + B q'``."""
        q, qd, qdd = (np.asarray(v, dtype=float) for v in (q, qd, qdd))
        M, C, G = self.dynamics(q, qd)
        return M @ qdd + C @ qd + G + self.friction_torque(qd)

    def forward_dynamics(
        self, q: Sequence[float], qd: Sequence[float], tau: Sequence[float]
    ) -> np.ndarray:
        """Joint accelerations ``q'' = M^{-1}(tau - C q' - G - B q')``."""
        q, qd, tau = (np.asarray(v, dtype=float) for v in (q, qd, tau))
        M, C, G = self.dynamics(q, qd)
        return np.linalg.solve(M, tau - C @ qd - G - self.friction_torque(qd))

    # ----------------------------------------------------------------------- #
    # Energy
    # ----------------------------------------------------------------------- #
    def kinetic_energy(self, q: Sequence[float], qd: Sequence[float]) -> float:
        """``T = 0.5 q'^T M(q) q'``."""
        qd = np.asarray(qd, dtype=float)
        return float(0.5 * qd @ self.mass_matrix(q) @ qd)

    def potential_energy(self, q: Sequence[float]) -> float:
        """``V`` with zero potential at the base height (THEORY.md §5.3).

        ``V = g1 sin(theta1) + g2 sin(theta1 + theta2)`` uses the same constants
        as ``G``, which is exactly ``dV/dq``.
        """
        return float(self.g1 * np.sin(q[0]) + self.g2 * np.sin(q[0] + q[1]))

    def total_energy(self, q: Sequence[float], qd: Sequence[float]) -> float:
        """Mechanical energy ``T + V``."""
        return self.kinetic_energy(q, qd) + self.potential_energy(q)

    # ======================================================================= #
    # Visualisation
    # ======================================================================= #
    def plot(
        self,
        q: Sequence[float],
        ax=None,
        *,
        color: str = "C0",
        alpha: float = 1.0,
        linewidth: float = 4.0,
        label: Optional[str] = None,
        show_workspace: bool = False,
        show_frame: bool = True,
    ):
        """Draw the arm in configuration ``q``.

        Parameters
        ----------
        q : array_like, shape (2,)
        ax : matplotlib Axes, optional
            Created if not given.
        color, alpha, linewidth, label
            Styling of the links.
        show_workspace : bool
            Draw the inner and outer workspace boundaries.
        show_frame : bool
            Set equal aspect ratio, limits and axis labels.

        Returns
        -------
        ax : matplotlib Axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        pts = self.joint_positions(q)
        ax.plot(pts[:, 0], pts[:, 1], "-", color=color, lw=linewidth, alpha=alpha,
                solid_capstyle="round", label=label, zorder=3)
        ax.plot(pts[:2, 0], pts[:2, 1], "o", color="k", ms=7, alpha=alpha, zorder=4)
        ax.plot(pts[2, 0], pts[2, 1], "o", mfc="w", mec=color, ms=7, mew=2, alpha=alpha, zorder=4)
        if show_workspace:
            self.plot_workspace(ax)
        if show_frame:
            self._format_axes(ax)
        return ax

    def plot_workspace(self, ax, **kwargs) -> None:
        """Shade the reachable annulus and draw its boundaries."""
        th = np.linspace(0.0, 2.0 * np.pi, 361)
        style = dict(color="0.6", lw=1.0, ls="--")
        style.update(kwargs)
        for r in (self.reach_min, self.reach_max):
            ax.plot(r * np.cos(th), r * np.sin(th), **style)
        outer = np.c_[self.reach_max * np.cos(th), self.reach_max * np.sin(th)]
        inner = np.c_[self.reach_min * np.cos(th[::-1]), self.reach_min * np.sin(th[::-1])]
        ring = np.vstack([outer, inner])
        ax.fill(ring[:, 0], ring[:, 1], color="0.92", zorder=0)

    def _format_axes(self, ax) -> None:
        lim = 1.1 * self.reach_max
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.axhline(0.0, color="0.85", lw=0.8, zorder=0)
        ax.axvline(0.0, color="0.85", lw=0.8, zorder=0)


def _wrap_to_pi(angle: float) -> float:
    """Wrap an angle to the interval ``(-pi, pi]``."""
    a = (angle + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.pi if a == -np.pi else a)
