"""
robot2r — Modeling, simulation and control of a 2R planar robotic arm.

Modules
-------
robot          RobotArm2R: kinematics, Jacobian, Lagrangian dynamics, drawing
trajectory     Quintic and piecewise-quintic joint-space trajectories
controllers    PID, PID+G, computed-torque control, controller factory
simulation     solve_ivp-based closed-loop simulator and metrics
visualization  Tracking / comparison plots and arm animation
symbolic       SymPy derivation and verification of M, C, G
"""
from .controllers import (
    ComputedTorqueController,
    Controller,
    PIDController,
    Reference,
    ZeroTorqueController,
    make_controller,
)
from .robot import RobotArm2R, RobotParams, SingularityWarning, WorkspaceError
from .simulation import SimulationResult, Simulator, format_metrics_table
from .trajectory import (
    PiecewiseQuinticTrajectory,
    QuinticTrajectory,
    check_trajectory_singularities,
    joint_trajectory_from_cartesian,
)

__all__ = [
    "RobotArm2R", "RobotParams", "WorkspaceError", "SingularityWarning",
    "QuinticTrajectory", "PiecewiseQuinticTrajectory",
    "joint_trajectory_from_cartesian", "check_trajectory_singularities",
    "Controller", "Reference", "PIDController", "ComputedTorqueController",
    "ZeroTorqueController", "make_controller",
    "Simulator", "SimulationResult", "format_metrics_table",
]
__version__ = "1.0.0"
