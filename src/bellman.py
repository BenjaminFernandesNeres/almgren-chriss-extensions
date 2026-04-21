"""
Bellman backward induction on a discrete grid.

Objective:
    Numerical dynamic-programming solver for the discrete version of the
    Almgren-Chriss liquidation problem.

We discretise the liquidation problem in time (N steps) and inventory (n_q levels).
At each node (time step k, inventory q), the Bellman equation gives the optimal
number of shares to sell in order to minimise the remaining cost-to-go.

The value function V[k, i] = minimum expected cost from step k with inventory q_i,
up to terminal step N where all inventory must be liquidated (q = 0).

This is the discrete-time analogue of the continuous mean-variance problem solved
analytically in almgren_chriss.py.  When the grid is fine enough, the two solutions
should match — which we verify in the notebook (Section 1, Question 4).

Authors: Benjamin Fernandes Neres, Dorian Deilhes, Ben Komara
Date: April 2026
"""

import numpy as np


def bellman_is(
    Q: float,
    T: float,
    sigma: float,
    eta: float,
    lam: float,
    N: int = 50,
    n_q: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bellman backward induction for the IS order on a discrete (t, q) grid.

    Discrete objective (IS):
        min  sum_{k=0}^{N-1} [ eta * (n_k / dt)^2 * dt + lambda * sigma^2 * q_k^2 * dt ]
        s.t. q_0 = Q,  q_N = 0,  n_k >= 0  (only selling allowed)

    where n_k = shares sold at step k, q_{k+1} = q_k - n_k.

    The running cost at step k with inventory q selling n shares:
        c(q, n) = eta * (n / dt)^2 * dt + lambda * sigma^2 * q^2 * dt

    Terminal condition: V[N, 0] = 0,  V[N, q] = +inf for q > 0.

    At each step k, backward induction:
        V[k, i] = min_{0 <= n <= q_i} [ c(q_i, n) + V[k+1, i - n_idx] ]

    Parameters
    ----------
    Q : float
        Initial inventory.
    T : float
        Liquidation horizon.
    sigma : float
        Asset volatility.
    eta : float
        Temporary market impact coefficient.
    lam : float
        Risk-aversion coefficient.
    N : int
        Number of time steps.
    n_q : int
        Number of inventory grid points (resolution).

    Returns
    -------
    t_grid : np.ndarray, shape (N+1,)
        Time grid from 0 to T.
    q_grid : np.ndarray, shape (n_q+1,)
        Inventory grid from 0 to Q.
    q_path : np.ndarray, shape (N+1,)
        Optimal inventory path starting from Q (extracted by forward pass).
    """
    dt = T / N

    # Inventory grid: 0, Q/n_q, 2Q/n_q, ..., Q
    q_grid = np.linspace(0.0, Q, n_q + 1)
    dq = q_grid[1] - q_grid[0]      # step size on the inventory grid
    t_grid = np.linspace(0.0, T, N + 1)

    # Value function: V[k, i] = cost-to-go at time step k with inventory q_grid[i]
    V = np.full((N + 1, n_q + 1), np.inf)
    # Policy: policy[k, i] = index of optimal inventory after selling at step k
    policy = np.zeros((N + 1, n_q + 1), dtype=int)

    # Terminal condition: must have sold everything
    V[N, 0] = 0.0   # q = 0 at T -> no cost
    # V[N, i] = inf for i > 0 (infeasible: shares left unsold)

    # Backward induction from step N-1 down to 0
    for k in range(N - 1, -1, -1):
        for i in range(n_q + 1):
            q_i = q_grid[i]   # current inventory

            best_cost = np.inf
            best_j = 0        # index of inventory after selling

            # Try all feasible next inventories j <= i  (can only sell, not buy)
            for j in range(i + 1):
                n_sold = q_i - q_grid[j]          # shares sold this step
                v_k = n_sold / dt                  # selling rate

                # Running cost: liquidity cost + inventory (market risk) cost
                running_cost = eta * v_k**2 * dt + lam * sigma**2 * q_i**2 * dt

                total = running_cost + V[k + 1, j]
                if total < best_cost:
                    best_cost = total
                    best_j = j

            V[k, i] = best_cost
            policy[k, i] = best_j

    # Forward pass: extract optimal inventory path starting from Q (index n_q)
    q_path = np.zeros(N + 1)
    q_path[0] = Q
    idx = n_q   # start at full inventory

    for k in range(N):
        idx = policy[k, idx]
        q_path[k + 1] = q_grid[idx]

    return t_grid, q_grid, q_path


def bellman_tc(
    Q: float,
    T: float,
    sigma: float,
    eta: float,
    lam: float,
    N: int = 50,
    n_q: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bellman backward induction for the TC order on a discrete (t, q) grid.

    Same structure as bellman_is, but the running cost uses the TC variance term:
        c_TC(q, n) = eta * (n / dt)^2 * dt + lambda * sigma^2 * (q - Q)^2 * dt

    The (q - Q)^2 term reflects that the TC benchmark is S_T: the risk is driven
    by how far q is from its initial level Q (i.e., how much has NOT been sold yet
    relative to the full position).

    Parameters
    ----------
    Q, T, sigma, eta, lam, N, n_q : see bellman_is.

    Returns
    -------
    t_grid, q_grid, q_path : np.ndarray — same structure as bellman_is.
    """
    dt = T / N

    q_grid = np.linspace(0.0, Q, n_q + 1)
    t_grid = np.linspace(0.0, T, N + 1)

    V = np.full((N + 1, n_q + 1), np.inf)
    policy = np.zeros((N + 1, n_q + 1), dtype=int)

    V[N, 0] = 0.0

    for k in range(N - 1, -1, -1):
        for i in range(n_q + 1):
            q_i = q_grid[i]

            best_cost = np.inf
            best_j = 0

            for j in range(i + 1):
                n_sold = q_i - q_grid[j]
                v_k = n_sold / dt

                # TC running cost: variance term is (q - Q)^2 instead of q^2
                running_cost = eta * v_k**2 * dt + lam * sigma**2 * (q_i - Q)**2 * dt

                total = running_cost + V[k + 1, j]
                if total < best_cost:
                    best_cost = total
                    best_j = j

            V[k, i] = best_cost
            policy[k, i] = best_j

    # Forward pass
    q_path = np.zeros(N + 1)
    q_path[0] = Q
    idx = n_q

    for k in range(N):
        idx = policy[k, idx]
        q_path[k + 1] = q_grid[idx]

    return t_grid, q_grid, q_path
