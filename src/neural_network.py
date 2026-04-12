"""
Neural network approach to optimal liquidation — Section 4 (Bonus).

We parameterise the trading policy as a small feed-forward neural network:
    v_t = pi_theta(t, q_t)

where (t, q_t) is the current state and v_t is the selling rate.

Training objective: minimise the mean-variance criterion directly via
Monte Carlo simulation over price paths.  No RL environment needed —
the objective is differentiable with respect to theta through the
simulated trajectories.

This approach is sometimes called "deep optimal stopping" or
"neural network control" in the literature.

We cover three order types: IS, TC, and POV (Percentage of Volume).
For POV, the benchmark is a fraction phi of a simulated market volume V_t.

Authors: Benjamin Fernandes Neres, Dorian Deilhes, Ben Komara
Date: April 2026
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

class LiquidationPolicy(nn.Module):
    """
    Small feed-forward neural network mapping state (t, q) to selling rate v.

    Architecture: Linear(2, 32) -> ReLU -> Linear(32, 32) -> ReLU -> Linear(32, 1) -> Softplus

    Inputs (normalised):
        t_norm = t / T    in [0, 1]
        q_norm = q / Q    in [0, 1]

    Output:
        v >= 0  (Softplus ensures non-negative selling rate)

    The network is kept intentionally small to avoid overfitting to the
    discrete grid and to remain interpretable.
    """

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Softplus(),   # guarantees v >= 0
        )

    def forward(self, t_norm: torch.Tensor, q_norm: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        t_norm : torch.Tensor, shape (batch,)  — normalised time in [0, 1]
        q_norm : torch.Tensor, shape (batch,)  — normalised inventory in [0, 1]

        Returns
        -------
        v : torch.Tensor, shape (batch,)  — selling rate (non-negative)
        """
        x = torch.stack([t_norm, q_norm], dim=1)   # (batch, 2)
        return self.net(x).squeeze(1)               # (batch,)


# ---------------------------------------------------------------------------
# Simulation and loss
# ---------------------------------------------------------------------------

def simulate_is(
    policy: LiquidationPolicy,
    Q: float,
    T: float,
    sigma: float,
    eta: float,
    lam: float,
    N: int,
    n_paths: int,
) -> torch.Tensor:
    """
    Simulate n_paths liquidation trajectories under the given policy and
    compute the IS mean-variance loss:

        L = E[cost] + lambda * Var[cost]
          = E[ eta * int v^2 dt + lambda * sigma^2 * int q^2 dt ]

    The simulation is fully differentiable with respect to policy parameters.

    Parameters
    ----------
    policy : LiquidationPolicy
    Q, T, sigma, eta, lam : float  — model parameters
    N : int    — number of time steps per path
    n_paths : int  — number of Monte Carlo paths

    Returns
    -------
    loss : torch.Tensor (scalar)  — mean-variance objective to minimise
    """
    dt = T / N
    sqrt_dt = np.sqrt(dt)

    q = torch.full((n_paths,), Q)   # inventory for each path
    cumulative_cost = torch.zeros(n_paths)   # eta * sum v^2 * dt
    cumulative_var  = torch.zeros(n_paths)   # sigma^2 * sum q^2 * dt

    for k in range(N):
        t_k = k * dt
        t_norm = torch.full((n_paths,), t_k / T)
        q_norm = (q / Q).detach().clamp(0.0, 1.0)   # detach q for normalisation

        # Policy: predicted selling rate
        v = policy(t_norm, q_norm)

        # Clip v so we do not sell more than remaining inventory
        v = torch.minimum(v, q / dt)
        v = torch.clamp(v, min=0.0)

        # Accumulate costs
        cumulative_cost = cumulative_cost + eta * v**2 * dt
        cumulative_var  = cumulative_var  + sigma**2 * q**2 * dt

        # Update inventory (deterministic: q decreases by v*dt)
        q = q - v * dt

    # Terminal penalty: large cost if inventory not fully liquidated
    terminal_penalty = 1e4 * q**2

    total_per_path = cumulative_cost + lam * cumulative_var + terminal_penalty

    # Mean-variance loss across paths
    loss = total_per_path.mean()
    return loss


def simulate_tc(
    policy: LiquidationPolicy,
    Q: float,
    T: float,
    sigma: float,
    eta: float,
    lam: float,
    N: int,
    n_paths: int,
) -> torch.Tensor:
    """
    Simulate TC (Target Close) trajectories and compute the TC mean-variance loss:

        L = E[ eta * int v^2 dt + lambda * sigma^2 * int (q - Q)^2 dt ]

    Same structure as IS but with (q - Q)^2 in the variance term.

    Parameters
    ----------
    Same as simulate_is.

    Returns
    -------
    loss : torch.Tensor (scalar)
    """
    dt = T / N

    q = torch.full((n_paths,), Q)
    cumulative_cost = torch.zeros(n_paths)
    cumulative_var  = torch.zeros(n_paths)

    for k in range(N):
        t_norm = torch.full((n_paths,), k * dt / T)
        q_norm = (q / Q).detach().clamp(0.0, 1.0)

        v = policy(t_norm, q_norm)
        v = torch.minimum(v, q / dt)
        v = torch.clamp(v, min=0.0)

        cumulative_cost = cumulative_cost + eta * v**2 * dt
        cumulative_var  = cumulative_var  + sigma**2 * (q - Q)**2 * dt   # TC penalty

        q = q - v * dt

    terminal_penalty = 1e4 * q**2
    total_per_path = cumulative_cost + lam * cumulative_var + terminal_penalty
    return total_per_path.mean()


def simulate_pov(
    policy: LiquidationPolicy,
    Q: float,
    T: float,
    sigma: float,
    eta: float,
    lam: float,
    phi: float,
    avg_volume: float,
    N: int,
    n_paths: int,
) -> torch.Tensor:
    """
    Simulate POV (Percentage of Volume) trajectories.

    In a POV order, the benchmark is phi * V_T where V_T is the total
    market volume traded over [0, T].  We model market volume as:
        dV_t = avg_volume * dt + vol_vol * avg_volume * dW_t^V

    The IS payoff uses the POV benchmark instead of S_0.

    For simplicity, we use the IS cost structure with an additional term
    that penalises deviating from the target participation rate phi:
        penalty = lam * (v_t - phi * v_market_t)^2

    Parameters
    ----------
    phi : float      — target participation rate (e.g. 0.1 = trade 10% of market volume)
    avg_volume : float — average market volume per unit time
    Others: same as simulate_is.

    Returns
    -------
    loss : torch.Tensor (scalar)
    """
    dt = T / N
    vol_vol = 0.3   # volatility of market volume (fixed)

    q = torch.full((n_paths,), Q)
    cumulative_cost = torch.zeros(n_paths)
    cumulative_var  = torch.zeros(n_paths)

    for k in range(N):
        t_norm = torch.full((n_paths,), k * dt / T)
        q_norm = (q / Q).detach().clamp(0.0, 1.0)

        v = policy(t_norm, q_norm)
        v = torch.minimum(v, q / dt)
        v = torch.clamp(v, min=0.0)

        # Simulate market volume this step (log-normal shocks)
        eps = torch.randn(n_paths)
        v_market = avg_volume * (1.0 + vol_vol * eps * np.sqrt(dt))
        v_market = v_market.clamp(min=0.01)   # market volume always positive

        # POV deviation penalty: we want v_t ≈ phi * v_market_t
        pov_deviation = (v - phi * v_market) ** 2

        cumulative_cost = cumulative_cost + eta * v**2 * dt
        cumulative_var  = cumulative_var  + sigma**2 * q**2 * dt + lam * pov_deviation * dt

        q = q - v * dt

    terminal_penalty = 1e4 * q**2
    total_per_path = cumulative_cost + cumulative_var + terminal_penalty
    return total_per_path.mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_policy(
    order_type: str,
    Q: float,
    T: float,
    sigma: float,
    eta: float,
    lam: float,
    N: int = 50,
    n_paths: int = 256,
    n_epochs: int = 400,
    lr: float = 1e-3,
    hidden_size: int = 32,
    pov_phi: float = 0.1,
    pov_volume: float = 2.0,
    seed: int = 42,
) -> tuple["LiquidationPolicy", list[float]]:
    """
    Train a LiquidationPolicy network for a given order type.

    Parameters
    ----------
    order_type : str
        One of 'IS', 'TC', 'POV'.
    Q, T, sigma, eta, lam : float
        Model parameters.
    N : int
        Number of time steps per simulated path.
    n_paths : int
        Number of Monte Carlo paths per gradient step.
    n_epochs : int
        Number of gradient steps.
    lr : float
        Adam learning rate.
    hidden_size : int
        Number of neurons per hidden layer.
    pov_phi : float
        Target participation rate (POV only).
    pov_volume : float
        Average market volume per unit time (POV only).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    policy : LiquidationPolicy  — trained policy network.
    losses : list[float]        — training loss history.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = LiquidationPolicy(hidden_size=hidden_size)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    losses = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        if order_type == 'IS':
            loss = simulate_is(policy, Q, T, sigma, eta, lam, N, n_paths)
        elif order_type == 'TC':
            loss = simulate_tc(policy, Q, T, sigma, eta, lam, N, n_paths)
        elif order_type == 'POV':
            loss = simulate_pov(policy, Q, T, sigma, eta, lam, pov_phi, pov_volume, N, n_paths)
        else:
            raise ValueError(f"Unknown order type: {order_type}. Choose 'IS', 'TC' or 'POV'.")

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 100 == 0:
            print(f'  [{order_type}] Epoch {epoch + 1}/{n_epochs} — loss: {loss.item():.4f}')

    return policy, losses


def extract_strategy(
    policy: LiquidationPolicy,
    Q: float,
    T: float,
    N: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the deterministic inventory path implied by a trained policy.

    Simulate one trajectory by following the policy greedily (no randomness).

    Parameters
    ----------
    policy : LiquidationPolicy
    Q, T : float
    N : int  — number of time steps

    Returns
    -------
    t_grid, q_path, v_path : np.ndarray
    """
    dt = T / N
    t_grid = np.linspace(0.0, T, N + 1)
    q_path = np.zeros(N + 1)
    v_path = np.zeros(N)

    q_path[0] = Q
    q = Q

    with torch.no_grad():
        for k in range(N):
            t_norm = torch.tensor([k * dt / T], dtype=torch.float32)
            q_norm = torch.tensor([q / Q], dtype=torch.float32).clamp(0.0, 1.0)
            v = policy(t_norm, q_norm).item()
            v = min(v, q / dt)   # cannot sell more than what remains
            v = max(v, 0.0)
            v_path[k] = v
            q = q - v * dt
            q_path[k + 1] = max(q, 0.0)

    return t_grid, q_path, v_path
