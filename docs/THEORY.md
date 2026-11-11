# Modeling, Simulation, and Control of a 2R Planar Robotic Arm — Theory

This document derives every model used in the code. Section numbers are cited in
the docstrings of the Python package (`robot2r/`).

---

## 1. System description and conventions

| Symbol | Meaning | Value |
|---|---|---|
| $l_1, l_2$ | link lengths | 1.0 m, 0.8 m |
| $m_1, m_2$ | link masses | 1.0 kg, 0.8 kg |
| $l_{c1}, l_{c2}$ | distance from joint $i$ to the centre of mass (COM) of link $i$ | $l_1/2 = 0.5$ m, $l_2/2 = 0.4$ m |
| $I_1, I_2$ | moment of inertia of link $i$ about its COM (slender rod) | $\tfrac{1}{12}m_1l_1^2 = 0.08\overline{3}$, $\tfrac{1}{12}m_2l_2^2 = 0.042\overline{6}$ kg·m² |
| $g$ | gravitational acceleration | 9.81 m/s² |

**Conventions.**

* Generalised coordinates $\boldsymbol\theta = [\theta_1,\ \theta_2]^\top$. $\theta_1$ is the angle of link 1 measured
  from the world $x$-axis; $\theta_2$ is the angle of link 2 *relative to link 1*.
* The world $y$-axis points up, so gravity is $\mathbf g = [0,\ -g]^\top$.
* Angles and torques are positive counter-clockwise. No joint friction (base model).
* Shorthand: $c_i = \cos\theta_i$, $s_i = \sin\theta_i$, $c_{12} = \cos(\theta_1+\theta_2)$, $s_{12} = \sin(\theta_1+\theta_2)$.

---

## 2. Forward kinematics

Using homogeneous transforms with standard DH parameters ($a_i = l_i$, $\alpha_i = 0$, $d_i = 0$):

$$
{}^{0}T_{1} = \begin{bmatrix} c_1 & -s_1 & 0 & l_1c_1\\ s_1 & c_1 & 0 & l_1s_1\\ 0&0&1&0\\0&0&0&1\end{bmatrix},\qquad
{}^{1}T_{2} = \begin{bmatrix} c_2 & -s_2 & 0 & l_2c_2\\ s_2 & c_2 & 0 & l_2s_2\\ 0&0&1&0\\0&0&0&1\end{bmatrix}
$$

$$
{}^{0}T_{2} = {}^{0}T_{1}\,{}^{1}T_{2} =
\begin{bmatrix} c_{12} & -s_{12} & 0 & l_1c_1 + l_2c_{12}\\ s_{12} & c_{12} & 0 & l_1s_1 + l_2s_{12}\\ 0&0&1&0\\0&0&0&1\end{bmatrix}
$$

so the elbow and end-effector (EE) positions and the EE orientation are

$$
\mathbf p_e = \begin{bmatrix} l_1c_1 \\ l_1s_1\end{bmatrix},\qquad
\mathbf p = \begin{bmatrix} x\\ y\end{bmatrix} = \begin{bmatrix} l_1c_1 + l_2c_{12}\\ l_1s_1 + l_2s_{12}\end{bmatrix},\qquad
\phi = \theta_1 + \theta_2 .
$$

The **reachable workspace** is the annulus $|l_1 - l_2| \le \|\mathbf p\| \le l_1 + l_2$, i.e. $0.2\ \text{m} \le r \le 1.8\ \text{m}$.

---

## 3. Analytical inverse kinematics

Given a target $\mathbf p = [x, y]^\top$, let $r^2 = x^2 + y^2$.

**Step 1 — elbow angle (law of cosines).** Squaring and adding the FK equations:

$$
x^2 + y^2 = l_1^2 + l_2^2 + 2l_1l_2c_2
\quad\Longrightarrow\quad
c_2 = \frac{x^2 + y^2 - l_1^2 - l_2^2}{2l_1l_2}.
$$

A solution exists iff $|c_2| \le 1$ (equivalently the target lies in the annulus of §2). Then

$$
s_2 = \sigma\sqrt{1 - c_2^2},\qquad \sigma \in \{+1, -1\},\qquad
\theta_2 = \operatorname{atan2}(s_2,\ c_2).
$$

The `atan2` form is used instead of $\arccos$ because it is numerically well conditioned
over the whole range and returns the correct quadrant.

**Step 2 — shoulder angle.** Write the EE position in a frame rotated by $\theta_1$:
$\mathbf p = R(\theta_1)\,[\,l_1 + l_2c_2,\ l_2s_2\,]^\top$. The vector $[l_1 + l_2c_2,\ l_2s_2]^\top$ makes the angle
$\operatorname{atan2}(l_2s_2,\ l_1 + l_2c_2)$ with link 1, hence

$$
\theta_1 = \operatorname{atan2}(y,\ x) - \operatorname{atan2}\!\big(l_2s_2,\ l_1 + l_2c_2\big).
$$

**Two branches.**

| Branch | Sign | Geometry |
|---|---|---|
| **Elbow-up** | $\sigma = -1 \Rightarrow \theta_2 < 0$ | the elbow lies on the left of (above, for targets with $x>0$) the base-to-target line |
| **Elbow-down** | $\sigma = +1 \Rightarrow \theta_2 > 0$ | the elbow lies on the right of (below) the base-to-target line |

**Degenerate cases.**

* $|c_2| > 1$: the target is unreachable → `WorkspaceError`.
* $|c_2| = 1$ ($s_2 = 0$): the target is on the workspace boundary; the two branches merge into one
  (fully stretched $\theta_2 = 0$, or fully folded $\theta_2 = \pi$). This is a kinematic singularity (§4).
* $r = 0$: only possible if $l_1 = l_2$; then $\theta_1$ is arbitrary. Cannot happen for this robot.

---

## 4. Jacobian and singularities

### 4.1 Geometric Jacobian

For revolute joints, column $i$ of the geometric Jacobian is
$\mathbf J_i = \begin{bmatrix}\mathbf z_{i-1}\times(\mathbf o_n - \mathbf o_{i-1})\\ \mathbf z_{i-1}\end{bmatrix}$ with
$\mathbf z_0 = \mathbf z_1 = [0,0,1]^\top$, $\mathbf o_0 = \mathbf 0$, $\mathbf o_1 = [l_1c_1, l_1s_1, 0]^\top$, $\mathbf o_2 = [x, y, 0]^\top$:

$$
\mathbf J(\boldsymbol\theta) =
\begin{bmatrix}
-l_1s_1 - l_2s_{12} & -l_2s_{12}\\
\ \ \,l_1c_1 + l_2c_{12} & \ \ \,l_2c_{12}\\
0 & 0\\ 0&0\\ 0&0\\ 1 & 1
\end{bmatrix} \in \mathbb R^{6\times 2},
\qquad
\begin{bmatrix}\dot{\mathbf p}\\ \boldsymbol\omega\end{bmatrix} = \mathbf J(\boldsymbol\theta)\,\dot{\boldsymbol\theta}.
$$

The rows relevant to planar positioning form the square **positional Jacobian**

$$
\mathbf J_v(\boldsymbol\theta) = \frac{\partial \mathbf p}{\partial\boldsymbol\theta} =
\begin{bmatrix}
-l_1s_1 - l_2s_{12} & -l_2s_{12}\\
l_1c_1 + l_2c_{12} & l_2c_{12}
\end{bmatrix},
\qquad
\dot{\mathbf J}_v =
\begin{bmatrix}
-l_1c_1\dot\theta_1 - l_2c_{12}(\dot\theta_1+\dot\theta_2) & -l_2c_{12}(\dot\theta_1+\dot\theta_2)\\
-l_1s_1\dot\theta_1 - l_2s_{12}(\dot\theta_1+\dot\theta_2) & -l_2s_{12}(\dot\theta_1+\dot\theta_2)
\end{bmatrix}.
$$

The EE acceleration is $\ddot{\mathbf p} = \mathbf J_v\ddot{\boldsymbol\theta} + \dot{\mathbf J}_v\dot{\boldsymbol\theta}$.

### 4.2 Singularities

$$
\det \mathbf J_v = (-l_1s_1 - l_2s_{12})\,l_2c_{12} - (l_1c_1 + l_2c_{12})(-l_2s_{12})
= l_1l_2\,(s_{12}c_1 - c_{12}s_1) = l_1l_2\sin\theta_2 .
$$

$\mathbf J_v$ is singular iff $\sin\theta_2 = 0$, i.e. $\theta_2 = 0$ (arm fully stretched, outer workspace
boundary) or $\theta_2 = \pi$ (arm folded, inner boundary). At these configurations the EE cannot move
along the line of the arm, and inverse velocity $\dot{\boldsymbol\theta} = \mathbf J_v^{-1}\dot{\mathbf p}$ blows up.

Measures used by the code:

* **Yoshikawa manipulability** $w(\boldsymbol\theta) = \sqrt{\det(\mathbf J_v\mathbf J_v^\top)} = l_1l_2|\sin\theta_2|$, maximised at $\theta_2 = \pm\pi/2$.
* **Normalised singularity index** $|\sin\theta_2| \in [0,1]$; a configuration is flagged singular when it is below a tolerance.
* **Damped least squares (DLS)** inverse used near singularities:
  $\mathbf J_v^{\#} = \mathbf J_v^\top(\mathbf J_v\mathbf J_v^\top + \lambda^2\mathbf I)^{-1}$, with the damping
  switched on smoothly as the index drops below a threshold $\varepsilon$:
  $\lambda^2 = \lambda_{\max}^2\,\big(1 - (s_2/\varepsilon)^2\big)$ for $|s_2| < \varepsilon$, else $0$.

Note that the **inertia matrix is never singular** (§5.5), so the dynamics have no singularities;
only the kinematic mappings do.

---

## 5. Dynamic model (Lagrangian mechanics)

### 5.1 COM positions and velocities

$$
\mathbf p_{c1} = \begin{bmatrix} l_{c1}c_1\\ l_{c1}s_1\end{bmatrix},\qquad
\mathbf p_{c2} = \begin{bmatrix} l_1c_1 + l_{c2}c_{12}\\ l_1s_1 + l_{c2}s_{12}\end{bmatrix}
$$

$$
\dot{\mathbf p}_{c1} = l_{c1}\dot\theta_1\begin{bmatrix} -s_1\\ c_1\end{bmatrix},\qquad
\dot{\mathbf p}_{c2} = \begin{bmatrix} -l_1s_1\dot\theta_1 - l_{c2}s_{12}(\dot\theta_1+\dot\theta_2)\\ \ \ \,l_1c_1\dot\theta_1 + l_{c2}c_{12}(\dot\theta_1+\dot\theta_2)\end{bmatrix}
$$

$$
\|\dot{\mathbf p}_{c1}\|^2 = l_{c1}^2\dot\theta_1^2,\qquad
\|\dot{\mathbf p}_{c2}\|^2 = l_1^2\dot\theta_1^2 + l_{c2}^2(\dot\theta_1+\dot\theta_2)^2 + 2l_1l_{c2}c_2\,\dot\theta_1(\dot\theta_1+\dot\theta_2),
$$

where the cross term used $c_1c_{12} + s_1s_{12} = \cos\theta_2$. Angular velocities: $\omega_1 = \dot\theta_1$, $\omega_2 = \dot\theta_1 + \dot\theta_2$.

### 5.2 Kinetic energy

$$
T = \tfrac12 m_1\|\dot{\mathbf p}_{c1}\|^2 + \tfrac12 I_1\omega_1^2 + \tfrac12 m_2\|\dot{\mathbf p}_{c2}\|^2 + \tfrac12 I_2\omega_2^2
$$

$$
T = \tfrac12\big(m_1l_{c1}^2 + I_1 + m_2l_1^2\big)\dot\theta_1^2
+ \tfrac12\big(m_2l_{c2}^2 + I_2\big)(\dot\theta_1+\dot\theta_2)^2
+ m_2l_1l_{c2}c_2\,\dot\theta_1(\dot\theta_1+\dot\theta_2).
$$

This is a quadratic form $T = \tfrac12\dot{\boldsymbol\theta}^\top\mathbf M(\boldsymbol\theta)\dot{\boldsymbol\theta}$ (§5.4).

### 5.3 Potential energy

With the zero of potential at the base height ($y = 0$):

$$
V = m_1g\,l_{c1}s_1 + m_2g\,(l_1s_1 + l_{c2}s_{12}).
$$

### 5.4 Inertia matrix

Reading the coefficients of $T = \tfrac12\sum_{ij}M_{ij}\dot\theta_i\dot\theta_j$ and introducing three
configuration-independent constants

$$
\alpha = m_1l_{c1}^2 + I_1 + m_2\,(l_1^2 + l_{c2}^2) + I_2,\qquad
\beta = m_2l_1l_{c2},\qquad
\delta = m_2l_{c2}^2 + I_2,
$$

$$
\boxed{\ \mathbf M(\boldsymbol\theta) = \begin{bmatrix} \alpha + 2\beta c_2 & \delta + \beta c_2\\ \delta + \beta c_2 & \delta\end{bmatrix}\ }
$$

$\mathbf M$ depends only on $\theta_2$ (the shoulder angle cannot change the arm's shape).

### 5.5 Properties of M

* Symmetric by construction.
* Positive definite: $T > 0$ for every $\dot{\boldsymbol\theta} \ne \mathbf 0$ because each body contributes a non-negative term and
  the rotational terms alone are positive. Explicitly,
  $\det\mathbf M = \alpha\delta - \delta^2 - \beta^2c_2^2 \ge \alpha\delta - \delta^2 - \beta^2 > 0$ for physical parameters
  (numerically $\det\mathbf M \ge 0.0910$ for this robot). Hence $\mathbf M^{-1}$ always exists and forward dynamics is
  well defined everywhere.

### 5.6 Euler–Lagrange equations

With $\mathcal L = T - V$, for $i = 1,2$:

$$
\frac{d}{dt}\frac{\partial\mathcal L}{\partial\dot\theta_i} - \frac{\partial\mathcal L}{\partial\theta_i} = \tau_i .
$$

Because $T = \tfrac12\dot{\boldsymbol\theta}^\top\mathbf M\dot{\boldsymbol\theta}$, the general result is

$$
\sum_j M_{ij}\ddot\theta_j + \sum_{j,k} c_{ijk}\,\dot\theta_j\dot\theta_k + \frac{\partial V}{\partial\theta_i} = \tau_i,
\qquad
c_{ijk} = \tfrac12\Big(\frac{\partial M_{ij}}{\partial\theta_k} + \frac{\partial M_{ik}}{\partial\theta_j} - \frac{\partial M_{jk}}{\partial\theta_i}\Big),
$$

where $c_{ijk}$ are the **Christoffel symbols of the first kind**.

### 5.7 Coriolis / centrifugal matrix

The only non-zero partial derivatives of $\mathbf M$ are with respect to $\theta_2$:

$$
\frac{\partial M_{11}}{\partial\theta_2} = -2\beta s_2,\qquad
\frac{\partial M_{12}}{\partial\theta_2} = \frac{\partial M_{21}}{\partial\theta_2} = -\beta s_2,\qquad
\frac{\partial M_{22}}{\partial\theta_2} = 0.
$$

Define $h(\theta_2) = -\beta s_2 = -m_2l_1l_{c2}\sin\theta_2$. The Christoffel symbols are

| $c_{ijk}$ | value | $c_{ijk}$ | value |
|---|---|---|---|
| $c_{111}$ | $0$ | $c_{211}$ | $\partial M_{21}/\partial\theta_1 - \tfrac12\partial M_{11}/\partial\theta_2 = -h$ |
| $c_{112} = c_{121}$ | $\tfrac12\partial M_{11}/\partial\theta_2 = h$ | $c_{212} = c_{221}$ | $\tfrac12\partial M_{22}/\partial\theta_1 = 0$ |
| $c_{122}$ | $\partial M_{12}/\partial\theta_2 - \tfrac12\partial M_{22}/\partial\theta_1 = h$ | $c_{222}$ | $0$ |

Choosing $C_{ij} = \sum_k c_{ijk}\dot\theta_k$ gives

$$
\boxed{\ \mathbf C(\boldsymbol\theta,\dot{\boldsymbol\theta}) = \begin{bmatrix} h\dot\theta_2 & h(\dot\theta_1 + \dot\theta_2)\\ -h\dot\theta_1 & 0\end{bmatrix},
\qquad h = -m_2l_1l_{c2}\sin\theta_2\ }
$$

so that

$$
\mathbf C\dot{\boldsymbol\theta} = \begin{bmatrix} -m_2l_1l_{c2}s_2\,(2\dot\theta_1\dot\theta_2 + \dot\theta_2^2)\\ \ \ \,m_2l_1l_{c2}s_2\,\dot\theta_1^2\end{bmatrix}.
$$

The $\dot\theta_1\dot\theta_2$ term is the **Coriolis** torque, the $\dot\theta_i^2$ terms are **centrifugal**.

**Passivity (skew-symmetry) property.** With the Christoffel choice of $\mathbf C$,

$$
\dot{\mathbf M} - 2\mathbf C =
\begin{bmatrix} 2h\dot\theta_2 & h\dot\theta_2\\ h\dot\theta_2 & 0\end{bmatrix}
- \begin{bmatrix} 2h\dot\theta_2 & 2h(\dot\theta_1+\dot\theta_2)\\ -2h\dot\theta_1 & 0\end{bmatrix}
= \begin{bmatrix} 0 & -h(2\dot\theta_1 + \dot\theta_2)\\ h(2\dot\theta_1 + \dot\theta_2) & 0\end{bmatrix},
$$

which is skew-symmetric, so $\mathbf x^\top(\dot{\mathbf M} - 2\mathbf C)\mathbf x = 0$ for all $\mathbf x$. This guarantees
$\dot E = \dot{\boldsymbol\theta}^\top\boldsymbol\tau$ (energy conservation when $\boldsymbol\tau = \mathbf 0$) and is
the basis of Lyapunov proofs for PD+gravity and computed-torque control. Both facts are checked numerically by the test suite.

### 5.8 Gravity vector

$$
\boxed{\ \mathbf G(\boldsymbol\theta) = \frac{\partial V}{\partial\boldsymbol\theta} =
\begin{bmatrix} (m_1l_{c1} + m_2l_1)\,g\,c_1 + m_2l_{c2}\,g\,c_{12}\\ m_2l_{c2}\,g\,c_{12}\end{bmatrix}\ }
$$

### 5.9 Equations of motion

$$
\boxed{\ \mathbf M(\boldsymbol\theta)\ddot{\boldsymbol\theta} + \mathbf C(\boldsymbol\theta,\dot{\boldsymbol\theta})\dot{\boldsymbol\theta} + \mathbf G(\boldsymbol\theta) = \boldsymbol\tau\ }
$$

Written out in full:

$$
\begin{aligned}
\tau_1 &= (\alpha + 2\beta c_2)\ddot\theta_1 + (\delta + \beta c_2)\ddot\theta_2 - \beta s_2(2\dot\theta_1\dot\theta_2 + \dot\theta_2^2) + (m_1l_{c1} + m_2l_1)gc_1 + m_2l_{c2}gc_{12}\\
\tau_2 &= (\delta + \beta c_2)\ddot\theta_1 + \delta\,\ddot\theta_2 + \beta s_2\dot\theta_1^2 + m_2l_{c2}gc_{12}
\end{aligned}
$$

### 5.10 Numerical values for this robot

$l_{c1} = 0.5$, $l_{c2} = 0.4$, $I_1 = 1/12$, $I_2 = 0.04267$:

$$
\alpha = 1.30400,\qquad \beta = 0.32000,\qquad \delta = 0.17067\quad[\text{kg·m}^2]
$$

$$
\mathbf M = \begin{bmatrix} 1.30400 + 0.64\,c_2 & 0.17067 + 0.32\,c_2\\ 0.17067 + 0.32\,c_2 & 0.17067\end{bmatrix},\qquad
\mathbf G = \begin{bmatrix} 12.753\,c_1 + 3.1392\,c_{12}\\ 3.1392\,c_{12}\end{bmatrix}\ \text{N·m},\qquad
h = -0.32\,s_2 .
$$

The worst-case static gravity torque on the shoulder is $12.753 + 3.139 = 15.89$ N·m (arm horizontal and stretched).
The effective shoulder inertia varies by a factor $(1.304 + 0.64)/(1.304 - 0.64) \approx 2.9$ over the workspace —
the main reason a fixed-gain PID behaves differently in different configurations.

### 5.11 State-space form for simulation

With $\mathbf x = [\boldsymbol\theta^\top,\ \dot{\boldsymbol\theta}^\top]^\top$:

$$
\dot{\mathbf x} = \begin{bmatrix}\dot{\boldsymbol\theta}\\ \mathbf M(\boldsymbol\theta)^{-1}\big(\boldsymbol\tau - \mathbf C(\boldsymbol\theta,\dot{\boldsymbol\theta})\dot{\boldsymbol\theta} - \mathbf G(\boldsymbol\theta)\big)\end{bmatrix}.
$$

The code solves $\mathbf M\ddot{\boldsymbol\theta} = \mathbf b$ with `numpy.linalg.solve` rather than forming $\mathbf M^{-1}$.
Controllers with internal states (the PID integrator) append them to $\mathbf x$ so that `solve_ivp`
integrates them consistently with the plant (see §8).

---

## 6. Quintic polynomial trajectory (joint space)

For each joint, $q(t) = \sum_{k=0}^{5} a_k\tau^k$ with $\tau = t - t_0 \in [0, T]$, $T = t_f - t_0$. The six coefficients
are fixed by position, velocity and acceleration at both ends:

$$
\begin{bmatrix}
1&0&0&0&0&0\\ 0&1&0&0&0&0\\ 0&0&2&0&0&0\\
1&T&T^2&T^3&T^4&T^5\\ 0&1&2T&3T^2&4T^3&5T^4\\ 0&0&2&6T&12T^2&20T^3
\end{bmatrix}
\begin{bmatrix}a_0\\a_1\\a_2\\a_3\\a_4\\a_5\end{bmatrix}
=
\begin{bmatrix}q_0\\ \dot q_0\\ \ddot q_0\\ q_f\\ \dot q_f\\ \ddot q_f\end{bmatrix}.
$$

The matrix is invertible for every $T > 0$. For rest-to-rest motion ($\dot q_0 = \ddot q_0 = \dot q_f = \ddot q_f = 0$) the
solution is the familiar minimum-jerk-like profile

$$
q(t) = q_0 + \Delta q\,(10s^3 - 15s^4 + 6s^5),\qquad s = \tau/T,
\qquad \dot q_{\max} = \frac{15}{8}\frac{\Delta q}{T},\qquad |\ddot q|_{\max} = \frac{10}{\sqrt3}\frac{\Delta q}{T^2}.
$$

Position, velocity **and acceleration** are continuous, so the feed-forward torque
$\mathbf M\ddot{\boldsymbol\theta}_d$ is continuous — important for computed-torque control.

**Multi-segment (via-point) trajectories.** Waypoints $\mathbf q_k$ at times $t_k$ are joined by quintic segments.
Interior via-point velocities follow the heuristic of Craig: with segment slopes $d_k = (q_{k+1} - q_k)/(t_{k+1} - t_k)$,
$\dot q_k = \tfrac12(d_{k-1} + d_k)$ if $d_{k-1}$ and $d_k$ have the same sign, else $0$. Via-point accelerations are
set to zero. This gives a $C^2$ trajectory through all waypoints.

---

## 7. Controllers

Tracking errors: $\mathbf e = \boldsymbol\theta_d - \boldsymbol\theta$, $\dot{\mathbf e} = \dot{\boldsymbol\theta}_d - \dot{\boldsymbol\theta}$.

### 7.1 Independent-joint PID

$$
\boldsymbol\tau = \mathbf K_p\mathbf e + \mathbf K_i\!\int_0^t\!\mathbf e\,d\sigma + \mathbf K_d\dot{\mathbf e}
\qquad(\mathbf K_p, \mathbf K_i, \mathbf K_d \text{ diagonal, positive}).
$$

* Each joint is treated as a decoupled SISO plant; the nonlinear terms $\mathbf C\dot{\boldsymbol\theta} + \mathbf G$ and the
  off-diagonal inertia act as **disturbances**.
* The integral term removes the steady-state error caused by gravity at rest, but during motion the
  disturbance is time varying, so a tracking error remains that scales with speed and with the configuration.
* Locally, around a fixed configuration with effective inertia $m_{ii}$, the closed loop of joint $i$ has characteristic
  polynomial $m_{ii}s^3 + K_{d,i}s^2 + K_{p,i}s + K_{i,i}$, stable iff $K_{d,i}K_{p,i} > m_{ii}K_{i,i}$ (Routh–Hurwitz).
  Because $m_{11}$ varies by ~2.9× over the workspace (§5.10) the damping ratio and bandwidth change with configuration.
* Implementation details: the integrator state is part of the ODE state; conditional-integration anti-windup
  clamps $\int\mathbf e$ to a user-set limit. An optional gravity-compensation term
  $\hat{\mathbf G}(\boldsymbol\theta)$ ("PID + G") is available for ablation.

### 7.2 Computed-torque control (feedback linearisation)

$$
\boldsymbol\tau = \hat{\mathbf M}(\boldsymbol\theta)\big(\ddot{\boldsymbol\theta}_d + \mathbf K_d\dot{\mathbf e} + \mathbf K_p\mathbf e\big)
+ \hat{\mathbf C}(\boldsymbol\theta,\dot{\boldsymbol\theta})\dot{\boldsymbol\theta} + \hat{\mathbf G}(\boldsymbol\theta).
$$

With a **perfect model** ($\hat{\mathbf M} = \mathbf M$, etc.), substituting into the equations of motion and cancelling
$\mathbf M$ (which is invertible) yields linear, decoupled error dynamics

$$
\ddot{\mathbf e} + \mathbf K_d\dot{\mathbf e} + \mathbf K_p\mathbf e = \mathbf 0 .
$$

Choosing $\mathbf K_p = \omega_n^2\mathbf I$, $\mathbf K_d = 2\zeta\omega_n\mathbf I$ places both poles of every joint at
$-\zeta\omega_n \pm \omega_n\sqrt{\zeta^2 - 1}$, **independent of configuration**. With $\zeta = 1$ the error decays as
$(1 + \omega_n t)e^{-\omega_n t}$ with no overshoot. The error converges exponentially and, if $\mathbf e(0) = \dot{\mathbf e}(0) = \mathbf 0$,
remains exactly zero.

**Model mismatch.** If the controller uses estimates $\hat{\mathbf M}, \hat{\mathbf C}, \hat{\mathbf G}$, define
$\tilde{\mathbf M} = \hat{\mathbf M} - \mathbf M$, etc. The error dynamics become

$$
\ddot{\mathbf e} + \mathbf K_d\dot{\mathbf e} + \mathbf K_p\mathbf e = \hat{\mathbf M}^{-1}\big(\tilde{\mathbf M}\ddot{\boldsymbol\theta} + \tilde{\mathbf C}\dot{\boldsymbol\theta} + \tilde{\mathbf G}\big) =: \boldsymbol\eta,
$$

a stable linear system driven by the bounded "uncertainty" $\boldsymbol\eta$. The residual error is therefore
$\mathcal O(\|\boldsymbol\eta\|/\omega_n^2)$; higher gains, an integral term, or robust/adaptive extensions reduce it
(§ "Improvements" in the README). The `robust` simulation case quantifies this with an unmodelled tip payload.

### 7.3 Actuator saturation

When a torque limit $\tau_{\max}$ is specified, the plant receives $\operatorname{sat}(\boldsymbol\tau) = \operatorname{clip}(\boldsymbol\tau, -\tau_{\max}, \tau_{\max})$.
Computed-torque control loses its exact linearisation while saturated.

---

## 8. Numerical simulation

* Integrator: `scipy.integrate.solve_ivp` (explicit Runge–Kutta 4(5) by default, tight tolerances
  `rtol = 1e-8`, `atol = 1e-10`, bounded `max_step`).
* Augmented state $\mathbf x = [\boldsymbol\theta,\ \dot{\boldsymbol\theta},\ \mathbf z]$, where $\mathbf z$ holds controller states.
  An adaptive integrator evaluates the right-hand side at trial times that may be rejected; keeping the integral of the
  error *inside* the ODE state (rather than as a Python attribute updated on each call) is what makes the PID simulation correct.
* Torques are re-computed from the stored states on the output grid, so the reported torque is exactly the torque applied by the controller.
* **Model validation:** with $\boldsymbol\tau = \mathbf 0$ the total energy $E = T + V$ must be constant; the `passive` case reports the relative drift.
  The SymPy module (`robot2r/symbolic.py`) re-derives $\mathbf M, \mathbf C, \mathbf G$ from the Lagrangian automatically and checks
  them against the hand-derived closed forms above.

### Performance metrics

For $N$ samples on a uniform grid:

$$
\text{RMSE}_i = \sqrt{\tfrac1N\textstyle\sum_k e_i(t_k)^2},\qquad
\text{RMSE}_{\text{EE}} = \sqrt{\tfrac1N\textstyle\sum_k\|\mathbf p_d(t_k) - \mathbf p(t_k)\|^2},
$$

$$
\text{RMS torque}_i = \sqrt{\tfrac1N\textstyle\sum_k\tau_i(t_k)^2},\qquad
\text{control effort} = \int_0^{t_f}\boldsymbol\tau^\top\boldsymbol\tau\,dt,\qquad
\text{mechanical work} = \int_0^{t_f}|\boldsymbol\tau^\top\dot{\boldsymbol\theta}|\,dt .
$$
