"""
Euler scheme and shooting method for the Euler-Lagrange equations.

Section 2 of the project.

The IS mean-variance problem leads to the Euler-Lagrange (EL) equation:
    eta * q_ddot(t) = lambda * sigma^2 * q(t)
    <=>  q_ddot = kappa^2 * q,   kappa = sqrt(lambda * sigma^2 / eta)

We rewrite this as a first-order system by setting v = -q_dot (selling rate):
    dq/dt = -v
    dv/dt = -kappa^2 * q

This gives the linear system:
    d/dt [q, v] = A * [q, v],   A = [[0, -1], [-kappa^2, 0]]

Boundary conditions: q(0) = Q,  q(T) = 0.

Because only q(0) is known (not v(0)), we cannot integrate directly.
The shooting method searches for the initial selling rate v(0) = v^i such that
the Euler-integrated solution satisfies q(T) = 0.

Authors: Benjamin Fernandes Neres, Dorian Deilhes, Ben Komara
Date: April 2026
"""

import numpy as np


def euler_step(
    q: float,
    v: float,
    kappa: float,
    dt: float,
) -> tuple[float, float]:
    """
    One step of the explicit Euler scheme for the EL ODE system.

    System:  dq/dt = -v
             dv/dt = -kappa^2 * q

    Euler update:
        q_{k+1} = q_k + dt * (-v_k)       = q_k - dt * v_k
        v_{k+1} = v_k + dt * (-kappa^2 * q_k)

    Parameters
    ----------
    q : float   — current inventory.
    v : float   — current selling rate.
    kappa : float — characteristic rate.
    dt : float  — time step.

    Returns
    -------
    q_next, v_next : float
    """
    q_next = q - dt * v
    v_next = v - dt * kappa**2 * q
    return q_next, v_next


def euler_solve(
    Q: float,
    T: float,
    v0: float,
    kappa: float,
    N: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate the EL ODE from t=0 to t=T using the explicit Euler scheme.

    Initial conditions: q(0) = Q,  v(0) = v0.

    Parameters
    ----------
    Q : float
        Initial inventory.
    T : float
        Liquidation horizon.
    v0 : float
        Initial selling rate (free parameter for the shooting method).
    kappa : float
        Characteristic rate = sqrt(lambda * sigma^2 / eta).
    N : int
        Number of Euler steps.

    Returns
    -------
    t_grid : np.ndarray, shape (N+1,)
    q_path : np.ndarray, shape (N+1,)
    v_path : np.ndarray, shape (N+1,)
    """
    dt = T / N
    t_grid = np.linspace(0.0, T, N + 1)
    q_path = np.zeros(N + 1)
    v_path = np.zeros(N + 1)

    q_path[0] = Q
    v_path[0] = v0

    for k in range(N):
        q_path[k + 1], v_path[k + 1] = euler_step(
            q_path[k], v_path[k], kappa, dt
        )

    return t_grid, q_path, v_path


def terminal_inventory(
    Q: float,
    T: float,
    v0: float,
    kappa: float,
    N: int,
) -> float:
    """
    Return the terminal inventory q(T) for a given initial selling rate v0.

    Used by the shooting method to evaluate the boundary condition residual:
        residual(v0) = q(T) - 0 = q(T)

    We want to find v0 such that residual(v0) = 0.

    Parameters
    ----------
    Q, T, v0, kappa, N : see euler_solve.

    Returns
    -------
    float : q(T), the inventory remaining at the end of the horizon.
    """
    _, q_path, _ = euler_solve(Q, T, v0, kappa, N)
    return q_path[-1]


def shooting_method(
    Q: float,
    T: float,
    kappa: float,
    N: int,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the EL boundary value problem using the shooting method.

    Goal: find v0 such that Euler-integrating from (Q, v0) gives q(T) = 0.

    Method: bisection on v0.
    - Low bound: v0 too small -> q(T) > 0 (not enough selling).
    - High bound: v0 too large -> q(T) < 0 (over-selling).

    The theoretical value v0* = Q * kappa * cosh(kappa*T) / sinh(kappa*T)
    is used to bracket the search interval.

    Parameters
    ----------
    Q : float
        Initial inventory.
    T : float
        Liquidation horizon.
    kappa : float
        Characteristic rate.
    N : int
        Number of Euler steps (grid resolution).
    tol : float
        Convergence tolerance on q(T).
    max_iter : int
        Maximum number of bisection iterations.

    Returns
    -------
    t_grid, q_path, v_path : np.ndarray
        Optimal trajectory found by the shooting method.
    """
    # Bracket: the theoretical v0 is a good starting point.
    # We widen the bracket to ensure we straddle q(T) = 0.
    if kappa < 1e-10:
        v0_theory = Q / T   # TWAP rate
    else:
        v0_theory = Q * kappa * np.cosh(kappa * T) / np.sinh(kappa * T)

    # Search bracket: [v_low, v_high]
    v_low = 0.0                 # selling nothing -> q(T) = Q > 0
    v_high = 10.0 * v0_theory  # selling too much -> q(T) < 0

    # Make sure the bracket is valid (straddles zero)
    q_low = terminal_inventory(Q, T, v_low, kappa, N)
    q_high = terminal_inventory(Q, T, v_high, kappa, N)

    if q_low * q_high > 0:
        raise ValueError(
            "Shooting method: bracket does not straddle q(T) = 0. "
            "Try increasing v_high or checking model parameters."
        )

    # Bisection
    for _ in range(max_iter):
        v_mid = (v_low + v_high) / 2.0
        q_mid = terminal_inventory(Q, T, v_mid, kappa, N)

        if abs(q_mid) < tol:
            break

        # Narrow the bracket
        if q_mid * q_low > 0:
            v_low = v_mid
            q_low = q_mid
        else:
            v_high = v_mid
            q_high = q_mid

    return euler_solve(Q, T, v_mid, kappa, N)
