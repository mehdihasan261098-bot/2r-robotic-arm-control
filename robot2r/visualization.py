"""
Plotting and animation utilities for :class:`~robot2r.simulation.SimulationResult`.

All functions return the Matplotlib figure (or animation) so callers decide
whether to show or save it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .simulation import SimulationResult

DEG = 180.0 / np.pi
#: Consistent colour per controller name across all figures.
CONTROLLER_COLORS = {
    "PID": "#d62728", "PID+G": "#ff7f0e", "CTC": "#1f77b4", "CTC+I": "#9467bd",
    "CTC (exact)": "#17becf", "Passive": "#2ca02c",
}


def _color(name: str, i: int = 0) -> str:
    """Colour of the longest known controller name that prefixes ``name``."""
    for key in sorted(CONTROLLER_COLORS, key=len, reverse=True):
        if name.startswith(key):
            return CONTROLLER_COLORS[key]
    return f"C{i}"


# --------------------------------------------------------------------------- #
# Single-run figures
# --------------------------------------------------------------------------- #
def plot_tracking(result: SimulationResult, title: Optional[str] = None) -> plt.Figure:
    """Joint positions vs reference, tracking errors, and torques (3x2 grid)."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    t, c = result.t, _color(result.controller_name)
    for j in range(2):
        ax = axes[0, j]
        ax.plot(t, result.q_ref[:, j] * DEG, "k--", lw=1.2, label="reference")
        ax.plot(t, result.q[:, j] * DEG, color=c, lw=1.6, label="actual")
        ax.set_ylabel(f"θ{j+1} [deg]")
        ax.legend(loc="best", fontsize=8)

        axes[1, j].plot(t, result.e[:, j] * DEG, color=c, lw=1.4)
        axes[1, j].axhline(0.0, color="0.6", lw=0.8)
        axes[1, j].set_ylabel(f"e{j+1} [deg]")

        axes[2, j].plot(t, result.tau[:, j], color=c, lw=1.4)
        axes[2, j].set_ylabel(f"τ{j+1} [N·m]")
        axes[2, j].set_xlabel("time [s]")
        axes[0, j].set_title(f"Joint {j+1}")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.suptitle(title or f"Trajectory tracking — {result.controller_name}")
    fig.tight_layout()
    return fig


def plot_task_space(
    results: Sequence[SimulationResult],
    n_snapshots: int = 6,
    title: str = "End-effector path",
) -> plt.Figure:
    """End-effector paths in the workspace with arm snapshots of the first result."""
    robot = results[0].robot
    fig, ax = plt.subplots(figsize=(7, 7))
    robot.plot_workspace(ax)
    ref = results[0].ee_ref
    ax.plot(ref[:, 0], ref[:, 1], "k--", lw=1.5, label="desired path", zorder=2)
    for i, r in enumerate(results):
        ax.plot(r.ee[:, 0], r.ee[:, 1], color=_color(r.controller_name, i), lw=1.6,
                label=f"{r.controller_name} actual", zorder=2)
    idx = np.linspace(0, len(results[0].t) - 1, n_snapshots).astype(int)
    for k, i in enumerate(idx):
        robot.plot(results[0].q[i], ax, color="0.35", alpha=0.25 + 0.75 * k / max(1, n_snapshots - 1),
                   linewidth=3, show_frame=False)
    ax.plot(*ref[0], "s", color="g", ms=9, label="start", zorder=5)
    ax.plot(*ref[-1], "*", color="m", ms=14, label="goal", zorder=5)
    robot._format_axes(ax)
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_energy(result: SimulationResult) -> plt.Figure:
    """Energy components and relative drift (for the passive validation case)."""
    robot = result.robot
    T = np.array([robot.kinetic_energy(q, qd) for q, qd in zip(result.q, result.qd)])
    V = np.array([robot.potential_energy(q) for q in result.q])
    E = T + V
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(result.t, T, label="kinetic T")
    ax1.plot(result.t, V, label="potential V")
    ax1.plot(result.t, E, "k", lw=2, label="total E")
    ax1.set_ylabel("energy [J]")
    ax1.legend()
    scale = max(abs(E[0]), np.ptp(V), 1e-12)
    ax2.plot(result.t, (E - E[0]) / scale, "k")
    ax2.set_ylabel("(E − E₀) / scale")
    ax2.set_xlabel("time [s]")
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.suptitle("Passive motion (τ = 0): energy conservation check")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Comparison figure
# --------------------------------------------------------------------------- #
def plot_comparison(results: Sequence[SimulationResult], title: str = "PID vs Computed-Torque Control") -> plt.Figure:
    """Side-by-side comparison: errors, torques, EE error and RMSE bar chart."""
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 3)
    ax_e = [fig.add_subplot(gs[0, j]) for j in range(2)]
    ax_t = [fig.add_subplot(gs[1, j], sharex=ax_e[0]) for j in range(2)]
    ax_ee = fig.add_subplot(gs[2, 0:2], sharex=ax_e[0])
    ax_bar = fig.add_subplot(gs[0:2, 2])
    ax_tab = fig.add_subplot(gs[2, 2])

    for i, r in enumerate(results):
        c = _color(r.controller_name, i)
        for j in range(2):
            ax_e[j].plot(r.t, r.e[:, j] * DEG, color=c, lw=1.4, label=r.controller_name)
            ax_t[j].plot(r.t, r.tau[:, j], color=c, lw=1.4, label=r.controller_name)
        # Small floor only to keep exact zeros (t = 0) off the log axis.
        ax_ee.semilogy(r.t, np.maximum(r.ee_error * 1e3, 1e-10), color=c, lw=1.4, label=r.controller_name)

    for j in range(2):
        ax_e[j].set_title(f"Tracking error, joint {j+1}")
        ax_e[j].set_ylabel("e [deg]")
        ax_t[j].set_title(f"Control torque, joint {j+1}")
        ax_t[j].set_ylabel("τ [N·m]")
    ax_ee.set_title("End-effector position error (log scale)")
    ax_ee.set_ylabel("‖p_d − p‖ [mm]")
    ax_ee.set_xlabel("time [s]")
    for ax in (*ax_e, *ax_t, ax_ee):
        ax.grid(alpha=0.3, which="both")
    # One shared legend instead of one per panel (keeps the curves visible).
    handles, labels = ax_e[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(results), bbox_to_anchor=(0.5, 0.955),
               frameon=False, fontsize=10)

    # RMSE bar chart (log scale: CTC is usually orders of magnitude smaller)
    names = [r.controller_name for r in results]
    rmse = np.array([r.metrics()["rmse_joint_rad"] * DEG for r in results])
    x = np.arange(2)
    w = 0.8 / len(results)
    for i, (n, v) in enumerate(zip(names, rmse)):
        bars = ax_bar.bar(x + (i - (len(results) - 1) / 2) * w, v, w, color=_color(n, i), label=n)
        ax_bar.bar_label(bars, fmt="%.2g", fontsize=8)
    ax_bar.set_xticks(x, ["θ1", "θ2"])
    ax_bar.set_yscale("log")
    ax_bar.set_ylabel("RMSE [deg] (log)")
    ax_bar.set_title("Joint RMSE")
    ax_bar.grid(alpha=0.3, axis="y", which="both")

    # Compact metrics table
    ax_tab.axis("off")
    cell = []
    for r in results:
        m = r.metrics()
        cell.append([
            r.controller_name.replace(" (", "\n("),
            f"{m['rmse_ee_m'] * 1e3:.3g}",
            f"{m['rms_torque_Nm'][0]:.2f} / {m['rms_torque_Nm'][1]:.2f}",
            f"{m['control_effort']:.0f}",
        ])
    tab = ax_tab.table(cellText=cell, colLabels=["", "EE RMSE\n[mm]", "RMS τ1/τ2\n[N·m]", "∫τᵀτ dt"],
                       loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1.0, 2.0 if len(results) <= 3 else 2.4)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# --------------------------------------------------------------------------- #
# Animation
# --------------------------------------------------------------------------- #
def animate(
    results: SimulationResult | Sequence[SimulationResult],
    *,
    fps: int = 30,
    speed: float = 1.0,
    trail: bool = True,
    save_path: Optional[str | Path] = None,
    titles: Optional[Sequence[str]] = None,
) -> FuncAnimation:
    """Animate one or several simulations side by side.

    Parameters
    ----------
    results : SimulationResult or sequence of them
        Runs to animate; each gets its own panel. Runs must share the time grid span.
    fps : int
        Frames per second of the animation.
    speed : float
        Playback speed relative to real time.
    trail : bool
        Draw the path traced by the end-effector.
    save_path : str or Path, optional
        ``.gif`` (Pillow, always available) or ``.mp4`` (needs ffmpeg).
    titles : sequence of str, optional
        Panel titles (default: controller names).

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Keep a reference to it while the figure is displayed.
    """
    if isinstance(results, SimulationResult):
        results = [results]
    n = len(results)
    robot = results[0].robot
    t0, tf = results[0].t[0], results[0].t[-1]
    frame_t = np.arange(t0, tf + 1e-9, speed / fps)

    def interp(arr: np.ndarray, t: np.ndarray, ts: np.ndarray) -> np.ndarray:
        return np.column_stack([np.interp(ts, t, arr[:, j]) for j in range(arr.shape[1])])

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.8), squeeze=False)
    axes = axes[0]
    artists = []
    for i, (ax, r) in enumerate(zip(axes, results)):
        c = _color(r.controller_name, i)
        robot.plot_workspace(ax)
        robot._format_axes(ax)
        ax.plot(r.ee_ref[:, 0], r.ee_ref[:, 1], "k--", lw=1.0, alpha=0.6, label="desired")
        ghost, = ax.plot([], [], "-", color="0.5", lw=3, alpha=0.35, solid_capstyle="round", label="reference arm")
        arm, = ax.plot([], [], "-o", color=c, lw=5, ms=7, mfc="k", mec="k", solid_capstyle="round")
        tip, = ax.plot([], [], "o", mfc="w", mec=c, mew=2, ms=8)
        path, = ax.plot([], [], "-", color=c, lw=1.5, label="actual")
        text = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", family="monospace", fontsize=9)
        ax.set_title(titles[i] if titles else r.controller_name)
        ax.legend(loc="lower left", fontsize=8)
        pts = robot.joint_positions(interp(r.q, r.t, frame_t))         # (F, 3, 2)
        ref_pts = robot.joint_positions(interp(r.q_ref, r.t, frame_t))
        err = np.interp(frame_t, r.t, r.ee_error)
        artists.append((arm, tip, path, ghost, text, pts, ref_pts, err))

    def update(k: int):
        out = []
        for arm, tip, path, ghost, text, pts, ref_pts, err in artists:
            arm.set_data(pts[k, :, 0], pts[k, :, 1])
            tip.set_data([pts[k, 2, 0]], [pts[k, 2, 1]])
            ghost.set_data(ref_pts[k, :, 0], ref_pts[k, :, 1])
            if trail:
                path.set_data(pts[: k + 1, 2, 0], pts[: k + 1, 2, 1])
            text.set_text(f"t = {frame_t[k]:5.2f} s\n|e_EE| = {err[k] * 1e3:7.3f} mm")
            out += [arm, tip, path, ghost, text]
        return out

    anim = FuncAnimation(fig, update, frames=len(frame_t), interval=1000.0 / fps, blit=False)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.suffix.lower() == ".gif":
            anim.save(str(save_path), writer=PillowWriter(fps=fps))
        else:
            anim.save(str(save_path), writer="ffmpeg", fps=fps)
    return anim
