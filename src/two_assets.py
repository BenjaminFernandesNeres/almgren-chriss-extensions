"""
Optimal liquidation of two correlated assets — Section 3.

We now hold two risky assets S1 and S2 with dynamics:
    dS1_t = sigma1 * dW1_t
    dS2_t = sigma2 * dW2_t
where (W1, W2) is a 2D Brownian motion with correlation rho.

Initial positions: q1_0 = Q > 0 (long S1), q2_0 = 0 (no position in S2).
We must liquidate everything: q1_T = 0, q2_T = 0.

Trading S2 (even though we hold none) is useful because its high correlation
with S1 allows us to offset S1 inventory risk at lower liquidity cost
(since eta2 < eta1, S2 is more liquid than S1).

IS payoff (two assets):
    X_T - Q * S1_0

Mean-variance objective:
    E[payoff] = -eta1 * int v1^2 dt - eta2 * int v2^2 dt
    Var[payoff] = sigma1^2 * int q1^2 dt
                + sigma2^2 * int q2^2 dt
                + 2 * rho * sigma1 * sigma2 * int q1 * q2 dt

We solve this with Bellman backward induction on a 2D inventory grid (q1, q2).

Note: q2 can be negative (short position in S2 used to hedge S1 risk).

Authors: Benjamin Fernandes Neres, Dorian Deilhes, Ben Komara
Date: April 2026
"""

import numpy as np


def bellman_two_assets(
    Q: float,
    T: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    eta1: float,
    eta2: float,
    lam: float,
    N: int = 30,
    n_q: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bellman backward induction for the two-asset IS problem on a discrete grid.

    State: (k, i, j) = (time step, index of q1, index of q2)
    Action: (n1, n2) = shares of S1 and S2 sold at step k.

    q1 in [0, Q], q2 in [-Q, 0] (we short S2 to hedge, so q2 <= 0).
    At terminal time: q1 = 0, q2 = 0.

    Running cost at state (q1_i, q2_j) selling (n1, n2):
        c = eta1 * v1^2 * dt + eta2 * v2^2 * dt
          + lam * (sigma1^2 * q1_i^2
                 + sigma2^2 * q2_j^2
                 + 2 * rho * sigma1 * sigma2 * q1_i * q2_j) * dt

    where v1 = n1 / dt, v2 = n2 / dt.

    Warning: the state space is O(n_q^2) and actions are O(n_q^2) per state,
    so total complexity is O(N * n_q^4). Keep n_q small (20-40) for tractability.

    Parameters
    ----------
    Q : float
        Initial inventory of asset S1.
    T : float
        Liquidation horizon.
    sigma1, sigma2 : float
        Volatilities of S1 and S2.
    rho : float
        Correlation between S1 and S2.
    eta1, eta2 : float
        Temporary impact coefficients (eta2 < eta1, S2 more liquid).
    lam : float
        Risk-aversion coefficient.
    N : int
        Number of time steps.
    n_q : int
        Grid size per asset (total grid: (n_q+1)^2 states per time step).

    Returns
    -------
    t_grid : np.ndarray, shape (N+1,)
    q1_grid : np.ndarray, shape (n_q+1,)   — S1 inventory grid [0, Q]
    q2_grid : np.ndarray, shape (n_q+1,)   — S2 inventory grid [-Q, 0]
    q1_path : np.ndarray, shape (N+1,)     — optimal S1 inventory path
    q2_path : np.ndarray, shape (N+1,)     — optimal S2 inventory path
    """
    dt = T / N
    t_grid = np.linspace(0.0, T, N + 1)

    # S1: long, we sell from Q down to 0
    q1_grid = np.linspace(0.0, Q, n_q + 1)
    # S2: we can go short (sell S2 we don't own to hedge), from 0 down to -Q
    q2_grid = np.linspace(-Q, 0.0, n_q + 1)

    # Indices for zero inventory on each grid
    idx_q1_zero = 0       # q1_grid[0] = 0
    idx_q2_zero = n_q     # q2_grid[n_q] = 0

    # Value function V[k, i, j] = cost-to-go at step k, q1=q1_grid[i], q2=q2_grid[j]
    V = np.full((N + 1, n_q + 1, n_q + 1), np.inf)
    # Policy: store next indices (i_next, j_next) after optimal action
    policy_i = np.zeros((N + 1, n_q + 1, n_q + 1), dtype=int)
    policy_j = np.zeros((N + 1, n_q + 1, n_q + 1), dtype=int)

    # Terminal condition: both inventories must be zero
    V[N, idx_q1_zero, idx_q2_zero] = 0.0

    # Backward induction
    for k in range(N - 1, -1, -1):
        for i in range(n_q + 1):          # loop over q1 levels
            for j in range(n_q + 1):      # loop over q2 levels
                q1_i = q1_grid[i]
                q2_j = q2_grid[j]

                best_cost = np.inf
                best_i_next = idx_q1_zero
                best_j_next = idx_q2_zero

                # Try all feasible next states:
                # q1 can only decrease (we sell S1): i_next <= i
                # q2 can only decrease or stay (we sell/short more S2): j_next <= j
                for i_next in range(i + 1):
                    for j_next in range(j + 1):
                        n1 = q1_i - q1_grid[i_next]   # shares of S1 sold
                        n2 = q2_j - q2_grid[j_next]   # shares of S2 sold (can be negative = buying back short)

                        # Only allow selling S2 (going more short or staying flat)
                        # j_next <= j means q2_grid[j_next] <= q2_grid[j] (more negative or same)
                        v1 = n1 / dt
                        v2 = n2 / dt

                        # Running cost: liquidity cost + inventory risk (cross term via rho)
                        liquidity = eta1 * v1**2 * dt + eta2 * v2**2 * dt
                        inventory_risk = lam * (
                            sigma1**2 * q1_i**2
                            + sigma2**2 * q2_j**2
                            + 2.0 * rho * sigma1 * sigma2 * q1_i * q2_j
                        ) * dt

                        total = liquidity + inventory_risk + V[k + 1, i_next, j_next]

                        if total < best_cost:
                            best_cost = total
                            best_i_next = i_next
                            best_j_next = j_next

                V[k, i, j] = best_cost
                policy_i[k, i, j] = best_i_next
                policy_j[k, i, j] = best_j_next

    # Forward pass: extract optimal paths starting from (q1=Q, q2=0)
    q1_path = np.zeros(N + 1)
    q2_path = np.zeros(N + 1)
    q1_path[0] = Q
    q2_path[0] = 0.0

    i_cur = n_q          # start at q1 = Q  (index n_q in q1_grid)
    j_cur = idx_q2_zero  # start at q2 = 0  (index n_q in q2_grid)

    for k in range(N):
        i_next = policy_i[k, i_cur, j_cur]
        j_next = policy_j[k, i_cur, j_cur]
        q1_path[k + 1] = q1_grid[i_next]
        q2_path[k + 1] = q2_grid[j_next]
        i_cur = i_next
        j_cur = j_next

    return t_grid, q1_grid, q2_grid, q1_path, q2_path
