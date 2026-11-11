"""
Symbolic derivation of the 2R dynamics with SymPy (THEORY.md §5).

The derivation is *independent* of the hand-written closed forms in
:mod:`robot2r.robot`: it builds the Lagrangian from COM positions, applies the
Euler–Lagrange equations, and extracts ``M``, ``C`` (Christoffel symbols) and
``G``. :func:`verify_against_numeric` then compares both implementations.

Run ``python -m robot2r.symbolic`` to print the equations in LaTeX and run the check.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import sympy as sp

from .robot import RobotArm2R


@dataclass
class SymbolicModel:
    """Symbolic dynamic model and lambdified numeric functions."""

    M: sp.Matrix
    C: sp.Matrix
    G: sp.Matrix
    T: sp.Expr
    V: sp.Expr
    eom: sp.Matrix                     # Euler–Lagrange left-hand side (= tau)
    symbols: Dict[str, sp.Symbol]
    M_func: Callable
    C_func: Callable
    G_func: Callable


def derive_dynamics() -> SymbolicModel:
    """Derive ``M``, ``C``, ``G`` of the 2R arm from its Lagrangian."""
    t = sp.symbols("t", real=True)
    l1, l2, lc1, lc2, m1, m2, I1, I2, g = sp.symbols(
        "l_1 l_2 l_c1 l_c2 m_1 m_2 I_1 I_2 g", positive=True
    )
    th1, th2 = sp.Function("theta_1")(t), sp.Function("theta_2")(t)
    th = sp.Matrix([th1, th2])
    thd = th.diff(t)
    thdd = thd.diff(t)

    # COM positions (§5.1)
    pc1 = sp.Matrix([lc1 * sp.cos(th1), lc1 * sp.sin(th1)])
    pc2 = sp.Matrix([l1 * sp.cos(th1) + lc2 * sp.cos(th1 + th2),
                     l1 * sp.sin(th1) + lc2 * sp.sin(th1 + th2)])
    vc1, vc2 = pc1.diff(t), pc2.diff(t)
    w1, w2 = thd[0], thd[0] + thd[1]

    # Energies (§5.2, §5.3)
    T = sp.Rational(1, 2) * (m1 * vc1.dot(vc1) + I1 * w1**2 + m2 * vc2.dot(vc2) + I2 * w2**2)
    V = m1 * g * pc1[1] + m2 * g * pc2[1]
    L = sp.simplify(T - V)

    # Euler–Lagrange equations (§5.6)
    eom = sp.Matrix([
        sp.diff(sp.diff(L, thd[i]), t) - sp.diff(L, th[i]) for i in range(2)
    ]).applyfunc(sp.simplify)

    # Replace time functions by plain symbols for extraction / lambdify.
    q1, q2, dq1, dq2, ddq1, ddq2 = sp.symbols("q1 q2 dq1 dq2 ddq1 ddq2", real=True)
    subs = {thdd[0]: ddq1, thdd[1]: ddq2, thd[0]: dq1, thd[1]: dq2, th1: q1, th2: q2}
    eom_s = eom.subs(subs)
    T_s = sp.simplify(T.subs(subs))
    V_s = sp.simplify(V.subs(subs))
    q, dq = sp.Matrix([q1, q2]), sp.Matrix([dq1, dq2])

    # Inertia matrix: coefficients of the accelerations.
    M = sp.Matrix(2, 2, lambda i, j: sp.simplify(sp.diff(eom_s[i], [ddq1, ddq2][j])))
    # Gravity vector: dV/dq.
    G = sp.Matrix([sp.diff(V_s, qi) for qi in q]).applyfunc(sp.simplify)
    # Coriolis matrix from Christoffel symbols of the first kind (§5.6–5.7).
    C = sp.zeros(2, 2)
    for i in range(2):
        for j in range(2):
            C[i, j] = sp.simplify(sum(
                sp.Rational(1, 2) * (sp.diff(M[i, j], q[k]) + sp.diff(M[i, k], q[j]) - sp.diff(M[j, k], q[i])) * dq[k]
                for k in range(2)
            ))

    # Consistency: the Euler–Lagrange expression must equal M q'' + C q' + G.
    residual = (eom_s - (M * sp.Matrix([ddq1, ddq2]) + C * dq + G)).applyfunc(sp.simplify)
    assert residual == sp.zeros(2, 1), f"Lagrangian and M/C/G forms disagree: {residual}"

    params = (l1, l2, lc1, lc2, m1, m2, I1, I2, g)
    syms = dict(zip(["l1", "l2", "lc1", "lc2", "m1", "m2", "I1", "I2", "g"], params))
    syms.update(q1=q1, q2=q2, dq1=dq1, dq2=dq2)
    return SymbolicModel(
        M=M, C=C, G=G, T=T_s, V=V_s, eom=eom_s, symbols=syms,
        M_func=sp.lambdify((q1, q2, *params), M, "numpy"),
        C_func=sp.lambdify((q1, q2, dq1, dq2, *params), C, "numpy"),
        G_func=sp.lambdify((q1, q2, *params), G, "numpy"),
    )


def verify_against_numeric(robot: RobotArm2R, model: SymbolicModel | None = None,
                           n_samples: int = 200, seed: int = 0) -> Dict[str, float]:
    """Max absolute difference between symbolic and closed-form ``M``, ``C``, ``G``."""
    model = model or derive_dynamics()
    p = robot.params
    pv = (p.l1, p.l2, p.lc1, p.lc2, p.m1, p.m2_eff, p.I1, p.I2, p.g)
    rng = np.random.default_rng(seed)
    err = {"M": 0.0, "C": 0.0, "G": 0.0}
    for _ in range(n_samples):
        q = rng.uniform(-np.pi, np.pi, 2)
        qd = rng.uniform(-5.0, 5.0, 2)
        err["M"] = max(err["M"], np.max(np.abs(np.array(model.M_func(*q, *pv), float) - robot.mass_matrix(q))))
        err["C"] = max(err["C"], np.max(np.abs(np.array(model.C_func(*q, *qd, *pv), float) - robot.coriolis_matrix(q, qd))))
        err["G"] = max(err["G"], np.max(np.abs(np.array(model.G_func(*q, *pv), float).ravel() - robot.gravity_vector(q))))
    return err


if __name__ == "__main__":  # pragma: no cover
    sm = derive_dynamics()
    print("Inertia matrix M(q):")
    sp.pprint(sm.M)
    print("\nCoriolis matrix C(q, dq):")
    sp.pprint(sm.C)
    print("\nGravity vector G(q):")
    sp.pprint(sm.G)
    print("\nLaTeX:")
    for name in ("M", "C", "G"):
        print(f"{name} = {sp.latex(getattr(sm, name))}\n")
    print("Max |symbolic - closed form| over 200 random states:",
          verify_against_numeric(RobotArm2R(), sm))
