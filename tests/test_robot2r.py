"""
Verification tests for the 2R arm model, trajectories, controllers and simulator.

Run with:  python -m pytest -q
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot2r import (  # noqa: E402
    PiecewiseQuinticTrajectory,
    QuinticTrajectory,
    RobotArm2R,
    RobotParams,
    Simulator,
    SingularityWarning,
    WorkspaceError,
    ZeroTorqueController,
    make_controller,
)

RNG = np.random.default_rng(42)


@pytest.fixture(scope="module")
def robot():
    return RobotArm2R()


# ------------------------------------------------------------------ kinematics
def test_parameters_match_specification(robot):
    p = robot.params
    assert (p.l1, p.l2, p.m1, p.m2, p.g) == (1.0, 0.8, 1.0, 0.8, 9.81)
    assert np.isclose(p.I1, 1.0 / 12.0) and np.isclose(p.I2, 0.8 * 0.64 / 12.0)
    assert np.isclose(robot.alpha, 1.304) and np.isclose(robot.beta, 0.32)
    assert np.isclose(robot.delta, 0.128 + 0.8 * 0.64 / 12.0)


def test_forward_kinematics_known_poses(robot):
    assert np.allclose(robot.forward_kinematics([0, 0]), [1.8, 0.0])
    assert np.allclose(robot.forward_kinematics([np.pi / 2, 0]), [0.0, 1.8])
    assert np.allclose(robot.forward_kinematics([0, np.pi / 2]), [1.0, 0.8])
    assert robot.forward_kinematics(RNG.uniform(-3, 3, (10, 2))).shape == (10, 2)


@pytest.mark.parametrize("elbow", ["up", "down"])
def test_ik_fk_roundtrip(robot, elbow):
    for _ in range(500):
        r = RNG.uniform(robot.reach_min + 1e-3, robot.reach_max - 1e-3)
        phi = RNG.uniform(-np.pi, np.pi)
        p = r * np.array([np.cos(phi), np.sin(phi)])
        q = robot.inverse_kinematics(p, elbow=elbow)
        assert np.allclose(robot.forward_kinematics(q), p, atol=1e-10)
        assert (q[1] <= 0) if elbow == "up" else (q[1] >= 0)


def test_ik_unreachable_and_boundary(robot):
    with pytest.raises(WorkspaceError):
        robot.inverse_kinematics([2.0, 0.0])
    with pytest.raises(WorkspaceError):
        robot.inverse_kinematics([0.1, 0.0])
    with pytest.warns(SingularityWarning):
        q = robot.inverse_kinematics([1.8, 0.0])
    assert np.allclose(q, [0.0, 0.0], atol=1e-6)


# -------------------------------------------------------------------- Jacobian
def test_jacobian_matches_finite_differences(robot):
    h = 1e-6
    for _ in range(50):
        q = RNG.uniform(-np.pi, np.pi, 2)
        J_fd = np.column_stack([
            (robot.forward_kinematics(q + h * e) - robot.forward_kinematics(q - h * e)) / (2 * h)
            for e in np.eye(2)
        ])
        assert np.allclose(robot.jacobian(q), J_fd, atol=1e-8)


def test_jacobian_dot_matches_finite_differences(robot):
    h = 1e-6
    q, qd = RNG.uniform(-np.pi, np.pi, 2), RNG.uniform(-2, 2, 2)
    Jd_fd = (robot.jacobian(q + h * qd) - robot.jacobian(q - h * qd)) / (2 * h)
    assert np.allclose(robot.jacobian_dot(q, qd), Jd_fd, atol=1e-7)


def test_singularity_detection(robot):
    for q2 in (0.0, np.pi, -np.pi):
        q = [0.4, q2]
        assert robot.is_singular(q)
        assert np.isclose(np.linalg.det(robot.jacobian(q)), 0.0, atol=1e-12)
    q = [0.4, 1.0]
    assert np.isclose(np.linalg.det(robot.jacobian(q)), robot.l1 * robot.l2 * np.sin(1.0))
    assert np.isclose(robot.manipulability([0, np.pi / 2]), 0.8)


def test_dls_inverse_velocity_is_bounded_near_singularity(robot):
    q = [0.2, 1e-6]
    with pytest.warns(SingularityWarning):
        qd = robot.inverse_velocity(q, [0.1, 0.0])
    assert np.all(np.isfinite(qd)) and np.linalg.norm(qd) < 100.0
    q = [0.2, 1.0]  # regular configuration: exact inverse
    xdot = np.array([0.1, -0.2])
    assert np.allclose(robot.jacobian(q) @ robot.inverse_velocity(q, xdot), xdot)


# -------------------------------------------------------------------- dynamics
def test_mass_matrix_symmetric_positive_definite(robot):
    for q2 in np.linspace(-np.pi, np.pi, 73):
        M = robot.mass_matrix([0.0, q2])
        assert np.allclose(M, M.T)
        assert np.all(np.linalg.eigvalsh(M) > 0)
        assert np.linalg.det(M) >= 0.0910 - 1e-4


def test_skew_symmetry_of_Mdot_minus_2C(robot):
    h = 1e-6
    for _ in range(50):
        q, qd = RNG.uniform(-np.pi, np.pi, 2), RNG.uniform(-3, 3, 2)
        Mdot = (robot.mass_matrix(q + h * qd) - robot.mass_matrix(q - h * qd)) / (2 * h)
        N = Mdot - 2 * robot.coriolis_matrix(q, qd)
        assert np.allclose(N, -N.T, atol=1e-7)


def test_gravity_is_gradient_of_potential(robot):
    h = 1e-6
    for _ in range(50):
        q = RNG.uniform(-np.pi, np.pi, 2)
        dV = np.array([(robot.potential_energy(q + h * e) - robot.potential_energy(q - h * e)) / (2 * h)
                       for e in np.eye(2)])
        assert np.allclose(robot.gravity_vector(q), dV, atol=1e-7)


def test_inverse_and_forward_dynamics_are_consistent(robot):
    for _ in range(50):
        q, qd, qdd = RNG.uniform(-3, 3, 2), RNG.uniform(-3, 3, 2), RNG.uniform(-5, 5, 2)
        tau = robot.inverse_dynamics(q, qd, qdd)
        assert np.allclose(robot.forward_dynamics(q, qd, tau), qdd)


def test_symbolic_derivation_matches_closed_form(robot):
    sympy = pytest.importorskip("sympy")  # noqa: F841
    from robot2r.symbolic import verify_against_numeric

    err = verify_against_numeric(robot, n_samples=50)
    assert max(err.values()) < 1e-12, err


def test_payload_model_reduces_to_base_model():
    a, b = RobotArm2R(), RobotArm2R(RobotParams(payload=0.0))
    assert np.isclose(a.alpha, b.alpha) and np.isclose(a.g1, b.g1)
    heavy = RobotArm2R(RobotParams(payload=0.5))
    assert heavy.params.lc2 > a.params.lc2 and heavy.g2 > a.g2


# ---------------------------------------------------------------- trajectories
def test_quintic_boundary_conditions():
    tr = QuinticTrajectory([0.0, 1.0], [1.0, -1.0], 1.0, 3.0, v0=[0.5, 0], af=[0, 1.0])
    q, qd, qdd = tr.evaluate(1.0)
    assert np.allclose(q, [0, 1]) and np.allclose(qd, [0.5, 0]) and np.allclose(qdd, 0)
    q, qd, qdd = tr.evaluate(3.0)
    assert np.allclose(q, [1, -1]) and np.allclose(qd, 0) and np.allclose(qdd, [0, 1.0])


def test_quintic_rest_to_rest_closed_form():
    tr = QuinticTrajectory([0.0], [2.0], 0.0, 2.0)
    ts = np.linspace(0, 2, 101)
    s = ts / 2.0
    q, qd, _ = tr.evaluate(ts)
    assert np.allclose(q[:, 0], 2.0 * (10 * s**3 - 15 * s**4 + 6 * s**5))
    assert np.isclose(qd[:, 0].max(), 15 / 8 * 2.0 / 2.0)


def test_piecewise_trajectory_is_c2_continuous():
    W = [[0, 0], [1, 0.5], [2, -0.5], [1, 0]]
    tr = PiecewiseQuinticTrajectory(W, [0, 1, 2.5, 4])
    eps = 1e-9
    for tk in (1.0, 2.5):
        a, b = tr.evaluate(tk - eps), tr.evaluate(tk + eps)
        for x, y in zip(a, b):
            assert np.allclose(x, y, atol=1e-6)
    assert np.allclose(tr.evaluate(2.5)[0], W[2])


# --------------------------------------------------------------- simulation
def test_passive_simulation_conserves_energy(robot):
    sim = Simulator(robot, ZeroTorqueController(), rtol=1e-11, atol=1e-12)
    res = sim.run(t_final=2.0, q0=[0.3, 0.6])
    E = res.energy
    assert np.max(np.abs(E - E[0])) < 1e-6


def test_ctc_with_perfect_model_tracks_exactly(robot):
    traj = QuinticTrajectory([0.2, -0.5], [1.5, -1.8], 0.0, 1.5)
    res = Simulator(robot, make_controller("ctc", robot), traj).run(t_final=2.0)
    assert np.max(np.abs(res.e)) < 1e-6


def test_ctc_error_dynamics_match_linear_design(robot):
    """With an initial error, CTC error must follow e(t) = e0 (1 + wn t) exp(-wn t)."""
    wn = 10.0
    traj = QuinticTrajectory([0.5, -1.0], [0.5, -1.0], 0.0, 1.0)  # constant reference
    e0 = np.array([0.05, -0.03])
    res = Simulator(robot, make_controller("ctc", robot, wn=wn), traj).run(
        t_final=1.0, q0=traj.q0 - e0)
    expected = np.outer((1 + wn * res.t) * np.exp(-wn * res.t), e0)
    assert np.allclose(res.e, expected, atol=1e-6)


def test_pid_regulates_against_gravity(robot):
    q_ref = np.array([0.0, 0.0])  # horizontal, stretched: maximum gravity load
    traj = QuinticTrajectory(q_ref, q_ref, 0.0, 0.1)
    res = Simulator(robot, make_controller("pid", robot), traj).run(t_final=12.0)
    assert np.all(np.abs(res.e[-1]) < 1e-3)  # integral action removes steady-state error


def test_ctc_integral_removes_payload_steady_state_error(robot):
    """With an unmodelled payload, CTC (PD outer loop) keeps a static error; CTC+I removes it."""
    plant = RobotArm2R(RobotParams(payload=0.5))
    q_ref = np.array([0.3, -0.8])
    traj = QuinticTrajectory(q_ref, q_ref, 0.0, 0.1)
    e_pd = Simulator(plant, make_controller("ctc", robot), traj).run(t_final=4.0).e[-1]
    e_pid = Simulator(plant, make_controller("ctci", robot), traj).run(t_final=4.0).e[-1]
    assert np.max(np.abs(e_pd)) > 1e-2
    assert np.max(np.abs(e_pid)) < 1e-4


def test_torque_saturation_is_respected(robot):
    traj = QuinticTrajectory([0, 0], [1.5, -1.5], 0.0, 0.5)
    res = Simulator(robot, make_controller("ctc", robot), traj, torque_limit=5.0).run(t_final=1.0)
    assert np.max(np.abs(res.tau)) <= 5.0 + 1e-12
