#!/usr/bin/env python
"""
Run the simulation cases of the 2R planar arm project.

Examples
--------
    python main.py --case all --no-show          # everything, save figures to ./results
    python main.py --case compare --animate       # PID vs CTC with side-by-side animation
    python main.py --case p2p --controller pid    # point-to-point move with PID
    python main.py --case robust --payload 0.5    # model mismatch study
    python main.py --case pid --q0-offset 3 -3    # start 3 deg off the reference

Cases
-----
passive   Zero torque from rest: energy conservation check of the dynamic model.
p2p       Point-to-point motion between two Cartesian points (IK + quintic).
pid       Trajectory tracking with independent-joint PID.
ctc       Trajectory tracking with computed-torque control.
compare   PID vs CTC on the same trajectory: errors, torques, RMSE table.
robust    Same, but the plant carries an unmodelled tip payload.
all       All of the above.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

# Cartesian via points of the tracking task (a closed loop through the workspace).
TRACK_POINTS = [(1.4, 0.2), (0.9, 0.9), (0.2, 1.3), (-0.5, 1.0), (0.3, 0.6), (1.4, 0.2)]
TRACK_TIMES = [0.0, 1.5, 3.0, 4.5, 6.0, 7.5]
TRACK_HOLD = 1.0        # seconds of regulation after the trajectory ends

P2P_START, P2P_GOAL = (1.5, 0.3), (-0.6, 1.1)
P2P_DURATION, P2P_HOLD = 2.5, 1.0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", default="compare",
                   choices=["passive", "p2p", "pid", "ctc", "compare", "robust", "all"])
    p.add_argument("--controller", default="ctc", choices=["pid", "pidg", "ctc", "ctci"],
                   help="controller for the p2p case (default: ctc)")
    p.add_argument("--elbow", default="up", choices=["up", "down"], help="IK branch")
    p.add_argument("--q0-offset", nargs=2, type=float, default=[0.0, 0.0], metavar=("D1", "D2"),
                   help="initial joint offset from the reference [deg]")
    p.add_argument("--torque-limit", type=float, default=None, help="actuator saturation [N m]")
    p.add_argument("--payload", type=float, default=0.5, help="unmodelled tip payload for 'robust' [kg]")
    p.add_argument("--with-pidg", action="store_true", help="also run PID + gravity compensation in 'compare'")
    p.add_argument("--animate", action="store_true", help="create (and save) arm animations")
    p.add_argument("--save-dir", default="results", help="output directory for figures and tables")
    p.add_argument("--no-show", action="store_true", help="do not open figure windows")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Case implementations (return lists of SimulationResult)
# --------------------------------------------------------------------------- #
def tracking_trajectory(robot, elbow):
    from robot2r import check_trajectory_singularities, joint_trajectory_from_cartesian

    traj = joint_trajectory_from_cartesian(robot, TRACK_POINTS, TRACK_TIMES, elbow=elbow)
    s_min = check_trajectory_singularities(robot, traj)
    print(f"  tracking trajectory: {len(TRACK_POINTS)} via points, {traj.tf:.1f} s, "
          f"min |sin θ2| = {s_min:.3f}")
    return traj


def run_tracking(robot_plant, model, kinds, args, label_suffix=""):
    """Simulate the tracking task for each controller kind; plant and model may differ."""
    from robot2r import Simulator, make_controller

    traj = tracking_trajectory(model, args.elbow)
    q0 = traj.evaluate(traj.t0)[0] + np.deg2rad(args.q0_offset)
    results = []
    for kind in kinds:
        ctrl = make_controller(kind, model)
        if kind != "pid":  # plain PID uses no model, so the suffix does not apply
            ctrl.name += label_suffix
        sim = Simulator(robot_plant, ctrl, traj, torque_limit=args.torque_limit)
        res = sim.run(t_final=traj.tf + TRACK_HOLD, q0=q0)
        print(f"  {ctrl.name:<22s} solved ({res.solver_stats['nfev']} RHS evaluations)")
        results.append(res)
    return results


def case_passive(args, out):
    from robot2r import RobotArm2R, Simulator, ZeroTorqueController
    from robot2r.visualization import animate, plot_energy

    print("\n[passive] free swing from rest, τ = 0")
    robot = RobotArm2R()
    sim = Simulator(robot, ZeroTorqueController(), rtol=1e-11, atol=1e-12, max_step=0.005)
    res = sim.run(t_final=5.0, q0=[0.3, 0.6], dt=0.002)
    E = res.energy
    scale = max(abs(E[0]), np.ptp([robot.potential_energy(q) for q in res.q]))
    print(f"  E0 = {E[0]:.6f} J, max |E - E0| = {np.max(np.abs(E - E[0])):.3e} J "
          f"(relative {np.max(np.abs(E - E[0])) / scale:.2e})")
    out.save(plot_energy(res), "passive_energy")
    if args.animate:
        out.keep(animate(res, save_path=out.dir / "passive.gif", titles=["Passive swing (τ = 0)"]))
    return [res]


def case_p2p(args, out):
    from robot2r import QuinticTrajectory, RobotArm2R, Simulator, check_trajectory_singularities, make_controller
    from robot2r.visualization import animate, plot_task_space, plot_tracking

    print(f"\n[p2p] {P2P_START} -> {P2P_GOAL} in {P2P_DURATION} s, controller = {args.controller}")
    robot = RobotArm2R()
    q_start = robot.inverse_kinematics(P2P_START, elbow=args.elbow)
    q_goal = robot.inverse_kinematics(P2P_GOAL, elbow=args.elbow)
    print(f"  IK ({args.elbow}): q_start = {np.rad2deg(q_start).round(2)} deg, "
          f"q_goal = {np.rad2deg(q_goal).round(2)} deg")
    sols = robot.inverse_kinematics_all(P2P_GOAL)
    print(f"  both IK branches at goal: up = {np.rad2deg(sols['up']).round(2)}, "
          f"down = {np.rad2deg(sols['down']).round(2)} deg")

    traj = QuinticTrajectory(q_start, q_goal, 0.0, P2P_DURATION)
    check_trajectory_singularities(robot, traj)
    dq = q_goal - q_start
    print(f"  quintic peak joint speeds  = {np.abs(15 / 8 * dq / P2P_DURATION).round(3)} rad/s")

    ctrl = make_controller(args.controller, robot)
    sim = Simulator(robot, ctrl, traj, torque_limit=args.torque_limit)
    res = sim.run(t_final=P2P_DURATION + P2P_HOLD, q0=q_start + np.deg2rad(args.q0_offset))
    m = res.metrics()
    print(f"  final EE error = {res.ee_error[-1] * 1e3:.3g} mm, EE RMSE = {m['rmse_ee_m'] * 1e3:.3g} mm")
    out.save(plot_task_space([res], title=f"Point-to-point motion ({ctrl.name})"), "p2p_task_space")
    out.save(plot_tracking(res, title=f"Point-to-point motion — {ctrl.name}"), "p2p_tracking")
    if args.animate:
        out.keep(animate(res, save_path=out.dir / "p2p.gif"))
    return [res]


def case_single(kind, args, out):
    from robot2r import RobotArm2R
    from robot2r.visualization import animate, plot_tracking

    print(f"\n[{kind}] trajectory tracking")
    robot = RobotArm2R()
    (res,) = run_tracking(robot, robot, [kind], args)
    out.save(plot_tracking(res), f"{kind}_tracking")
    if args.animate:
        out.keep(animate(res, save_path=out.dir / f"{kind}_tracking.gif"))
    return [res]


def case_compare(args, out):
    from robot2r import RobotArm2R, format_metrics_table
    from robot2r.visualization import animate, plot_comparison, plot_task_space

    print("\n[compare] PID vs computed-torque control, identical trajectory")
    robot = RobotArm2R()
    kinds = ["pid", "pidg", "ctc"] if args.with_pidg else ["pid", "ctc"]
    results = run_tracking(robot, robot, kinds, args)
    table = format_metrics_table(results)
    print(table)
    out.table("compare", "PID vs CTC (nominal model)", table)
    out.save(plot_comparison(results), "compare_errors_torques")
    out.save(plot_task_space(results, title="End-effector path: PID vs CTC"), "compare_task_space")
    if args.animate:
        out.keep(animate(results, save_path=out.dir / "compare.gif"))
    return results


def case_robust(args, out):
    from robot2r import RobotArm2R, RobotParams, format_metrics_table
    from robot2r.visualization import plot_comparison

    print(f"\n[robust] plant carries an unmodelled {args.payload} kg tip payload")
    nominal = RobotArm2R()
    plant = RobotArm2R(RobotParams(payload=args.payload))
    results = run_tracking(plant, nominal, ["pid", "pidg", "ctc", "ctci"], args, label_suffix=" (nominal)")
    # Reference: CTC that knows the payload (perfect model).
    results += run_tracking(plant, plant, ["ctc"], args, label_suffix=" (exact)")
    table = format_metrics_table(results)
    print(table)
    out.table("robust", f"Model mismatch: unmodelled payload {args.payload} kg", table)
    out.save(plot_comparison(results, title=f"Robustness to an unmodelled {args.payload} kg payload"),
             "robust_comparison")
    return results


# --------------------------------------------------------------------------- #
# Output handling
# --------------------------------------------------------------------------- #
class Output:
    """Saves figures/tables and keeps animations alive until plt.show()."""

    def __init__(self, directory: str):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._anims = []
        self._tables = []

    def save(self, fig, name: str) -> None:
        path = self.dir / f"{name}.png"
        fig.savefig(path, dpi=130)
        print(f"  saved {path}")

    def keep(self, anim) -> None:
        self._anims.append(anim)

    def table(self, key: str, title: str, md: str) -> None:
        self._tables.append(f"## {title}\n\n{md}\n")

    def write_tables(self) -> None:
        if self._tables:
            path = self.dir / "metrics.md"
            path.write_text("# Simulation metrics\n\n" + "\n".join(self._tables), encoding="utf-8")
            print(f"\nMetrics written to {path}")


def main(argv=None) -> int:
    args = parse_args(argv)
    # Windows consoles default to a legacy code page that cannot print θ, τ, ...
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Make sure the package is importable when run from any directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    warnings.simplefilter("always")  # surface singularity warnings once per occurrence

    out = Output(args.save_dir)
    cases = {
        "passive": lambda: case_passive(args, out),
        "p2p": lambda: case_p2p(args, out),
        "pid": lambda: case_single("pid", args, out),
        "ctc": lambda: case_single("ctc", args, out),
        "compare": lambda: case_compare(args, out),
        "robust": lambda: case_robust(args, out),
    }
    selected = list(cases) if args.case == "all" else [args.case]
    for name in selected:
        cases[name]()
    out.write_tables()

    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
