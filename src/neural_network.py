"""
Neural approximations for Almgren-Chriss liquidation problems.

Objective:
    Learn liquidation policies with neural networks and compare them with the
    analytical and Bellman references developed in the previous sections.

This module contains two families of policies:
1) Static schedule policy (`StaticScheduleNet`):
    - Learns a deterministic liquidation profile on a fixed grid.
    - Uses softmax weights so traded quantities are nonnegative and sum to Q.
    - Therefore terminal inventory is exactly zero by construction.

2) Dynamic POV policy (`POVPolicy`):
    - Learns a state-dependent participation decision under a hard POV cap.
    - Trained through Monte Carlo simulation with a terminal inventory penalty.
    - Also supports a stochastic-volatility extension where the risk term uses
      a simulated volatility path.

All computations use float64 for stable optimization in this small-scale
quantitative setting.

Authors: Benjamin Fernandes Neres, Dorian Deilhes, Ben Komara
Date: April 2026
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

TORCH_DTYPE = torch.float64


class StaticScheduleNet(nn.Module):
    """
    Neural network for a static liquidation schedule on a fixed time grid.

    Input:
        t_k / T, for k = 0, ..., N-1

    Output:
        nonnegative weights w_k summing to 1 via softmax

    Interpretation:
        x_k = Q * w_k   is the quantity sold on interval k
        v_k = x_k / dt  is the selling rate on interval k

    This guarantees exact liquidation:
        sum_k x_k = Q
    """

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(1, hidden_size, dtype=TORCH_DTYPE),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size, dtype=TORCH_DTYPE),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, dtype=TORCH_DTYPE),
        )

    def forward(self, t_norm_grid: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t_norm_grid : torch.Tensor, shape (N,)
            Normalized time grid in [0, 1).

        Returns
        -------
        weights : torch.Tensor, shape (N,)
            Nonnegative weights summing to 1.
        """
        scores = self.score_net(t_norm_grid.unsqueeze(1)).squeeze(1)  # (N,)
        weights = torch.softmax(scores, dim=0)
        return weights


def build_schedule(
    weights: torch.Tensor,
    Q: float,
    T: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Convert schedule weights into a full discrete trading trajectory.

    Parameters
    ----------
    weights : torch.Tensor, shape (N,)
        Nonnegative weights summing to 1.
    Q : float
        Initial inventory.
    T : float
        Time horizon.

    Returns
    -------
    x : torch.Tensor, shape (N,)
        Quantity traded on each interval.
    q_path : torch.Tensor, shape (N+1,)
        Inventory path at grid points, from q_0 = Q to q_N = 0.
    q_mid : torch.Tensor, shape (N,)
        Midpoint inventory on each interval (used for integral approximation).
    dt : float
        Time step.
    """
    device = weights.device
    N = weights.numel()
    dt = T / N

    # Traded quantity per interval. Because weights sum to 1, x sums to Q.
    x = Q * weights  # sum(x) = Q exactly

    # Inventory right after each interval trade.
    q_after = Q - torch.cumsum(x, dim=0)  # shape (N,)
    q_path = torch.cat(
        [torch.tensor([Q], dtype=weights.dtype, device=device), q_after],
        dim=0
    )  # shape (N+1,)

    # Inventory right before each interval trade.
    q_before = q_path[:-1]  # shape (N,)

    # Midpoint inventory on each interval, better for integral approximation
    q_mid = q_before - 0.5 * x

    return x, q_path, q_mid, dt


def loss_is(
    model: StaticScheduleNet,
    Q: float,
    T: float,
    sigma: float,
    eta: float,
    lam: float,
    N: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, dict]:
    """
    Discrete IS objective on a fixed grid.

    Continuous reference objective:
        J = eta * integral(v_t^2 dt) + lam * sigma^2 * integral(q_t^2 dt)

    Discretization used here:
    - v is piecewise constant on each interval,
    - q is approximated at interval midpoints for the risk term.
    """
    t_norm = torch.linspace(0.0, 1.0, N + 1, dtype=TORCH_DTYPE, device=device)[:-1]
    weights = model(t_norm)

    x, q_path, q_mid, dt = build_schedule(weights, Q, T)
    v = x / dt

    impact = eta * torch.sum(v**2) * dt
    risk = lam * sigma**2 * torch.sum(q_mid**2) * dt
    loss = impact + risk

    # Detached diagnostics are returned for logging/plots without autograd graph.
    aux = {
        "weights": weights.detach(),
        "x": x.detach(),
        "v": v.detach(),
        "q_path": q_path.detach(),
        "impact": impact.item(),
        "risk": risk.item(),
    }
    return loss, aux


def loss_tc(
    model: StaticScheduleNet,
    Q: float,
    T: float,
    sigma: float,
    eta: float,
    lam: float,
    N: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, dict]:
    """
    Discrete TC objective on a fixed grid.

    Continuous reference objective:
        J = eta * integral(v_t^2 dt) + lam * sigma^2 * integral((Q - q_t)^2 dt)

    Here, (Q - q) is the amount already sold, evaluated at interval midpoints.
    """
    t_norm = torch.linspace(0.0, 1.0, N + 1, dtype=TORCH_DTYPE, device=device)[:-1]
    weights = model(t_norm)

    x, q_path, q_mid, dt = build_schedule(weights, Q, T)
    v = x / dt
    x_mid = Q - q_mid  # shares already sold at interval midpoint

    impact = eta * torch.sum(v**2) * dt
    risk = lam * sigma**2 * torch.sum(x_mid**2) * dt
    loss = impact + risk

    # Detached diagnostics are returned for logging/plots without autograd graph.
    aux = {
        "weights": weights.detach(),
        "x": x.detach(),
        "v": v.detach(),
        "q_path": q_path.detach(),
        "impact": impact.item(),
        "risk": risk.item(),
    }
    return loss, aux


def train_static_policy(
    order_type: str,
    Q: float,
    T: float,
    sigma: float,
    eta: float,
    lam: float,
    N: int = 100,
    n_epochs: int = 3000,
    lr: float = 5e-3,
    hidden_size: int = 64,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[StaticScheduleNet, list[float]]:
    """
    Train a static schedule policy for IS or TC.

    Training details:
    - Optimizer: AdamW
    - Gradient clipping: max norm = 1.0
    - Best checkpoint: keeps parameters with lowest observed loss

    Returns
    -------
    model : StaticScheduleNet
        The best model found during training.
    losses : list[float]
        Loss value at each epoch.
    """
    if order_type not in {"IS", "TC"}:
        raise ValueError("order_type must be 'IS' or 'TC'.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = StaticScheduleNet(hidden_size=hidden_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)

    losses = []
    best_state = None
    best_loss = float("inf")

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        if order_type == "IS":
            loss, _ = loss_is(model, Q, T, sigma, eta, lam, N, device=device)
        else:
            loss, _ = loss_tc(model, Q, T, sigma, eta, lam, N, device=device)

        loss.backward()
        # Prevent occasional exploding gradients during early epochs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 500 == 0:
            print(f"[{order_type}] Epoch {epoch + 1}/{n_epochs} — loss = {loss_value:.8f}")

    if best_state is not None:
        # Restore best (not last) parameters for robust downstream evaluation.
        model.load_state_dict(best_state)

    return model, losses


@torch.no_grad()
def extract_static_strategy(
    model: StaticScheduleNet,
    Q: float,
    T: float,
    N: int,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Export a trained static policy to numpy arrays for analysis/plotting.

    Returns
    -------
    t_grid : np.ndarray, shape (N+1,)
        Grid points from 0 to T.
    q_np : np.ndarray, shape (N+1,)
        Inventory at each grid point.
    v_np : np.ndarray, shape (N,)
        Piecewise-constant selling rates per interval.
    """
    model.eval()

    t_norm = torch.linspace(0.0, 1.0, N + 1, dtype=TORCH_DTYPE, device=device)[:-1]
    weights = model(t_norm)
    x, q_path, _, dt = build_schedule(weights, Q, T)
    v = x / dt

    t_grid = np.linspace(0.0, T, N + 1)
    q_np = q_path.cpu().numpy()
    v_np = v.cpu().numpy()

    return t_grid, q_np, v_np


class POVPolicy(nn.Module):
    """
    Dynamic POV policy.

    Input:
        (t/T, q/Q, market_volume/avg_volume)

    Output:
        alpha in (0, 1), interpreted as the fraction of the allowed POV cap used.

    Trading rate construction in simulation:
        v_t = alpha_t * (phi * V_t)
    where phi is the participation cap and V_t is market volume.
    """

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_size, dtype=TORCH_DTYPE),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, dtype=TORCH_DTYPE),
            nn.ReLU(),
            nn.Linear(hidden_size, 1, dtype=TORCH_DTYPE),
        )

    def forward(
        self,
        t_norm: torch.Tensor,
        q_norm: torch.Tensor,
        vol_norm: torch.Tensor,
    ) -> torch.Tensor:
        # Build per-path feature vector and map it to participation ratio alpha.
        x = torch.stack([t_norm, q_norm, vol_norm], dim=1)
        alpha = torch.sigmoid(self.net(x)).squeeze(1)
        return alpha


def simulate_pov_hard_cap(
    policy: POVPolicy,
    Q: float,
    T: float,
    sigma: float,
    eta: float,
    lam: float,
    phi: float,
    avg_volume: float,
    N: int,
    n_paths: int,
    vol_of_vol: float = 0.2,
    terminal_penalty: float = 1e4,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Simulate a POV strategy with hard participation cap:
        v_t <= phi * V_t

    For each Monte Carlo path and time step, we:
    1) sample stochastic market volume,
    2) query the policy for participation alpha_t,
    3) enforce the hard cap and remaining-inventory constraint,
    4) accumulate impact + risk costs,
    5) add a terminal penalty for leftover inventory.

    Returns
    -------
    loss : torch.Tensor (scalar)
        Mean objective across all simulated paths.
    """
    if phi * avg_volume * T < Q:
        raise ValueError(
            "Infeasible POV setup: phi * avg_volume * T < Q, "
            "so full liquidation is impossible on average."
        )

    dt = T / N
    sqrt_dt = np.sqrt(dt)

    q = torch.full((n_paths,), Q, dtype=TORCH_DTYPE, device=device)
    impact_cost = torch.zeros(n_paths, dtype=TORCH_DTYPE, device=device)
    risk_cost = torch.zeros(n_paths, dtype=TORCH_DTYPE, device=device)

    for k in range(N):
        t_norm = torch.full((n_paths,), k / N, dtype=TORCH_DTYPE, device=device)

        # Lognormal-like positive volume dynamics around avg_volume.
        eps = torch.randn(n_paths, dtype=TORCH_DTYPE, device=device)
        vol = avg_volume * torch.exp(
            (-0.5 * vol_of_vol**2) * dt + vol_of_vol * sqrt_dt * eps
        )
        vol_norm = vol / avg_volume

        q_norm = (q / Q).clamp(0.0, 1.0)
        alpha = policy(t_norm, q_norm, vol_norm)      # in (0,1)
        v_cap = phi * vol                             # hard POV cap
        v = alpha * v_cap

        # Hard physical constraint: cannot trade more than what remains.
        x = torch.minimum(v * dt, q)
        v_eff = x / dt

        q_mid = q - 0.5 * x
        impact_cost = impact_cost + eta * v_eff**2 * dt
        risk_cost = risk_cost + lam * sigma**2 * q_mid**2 * dt

        q = q - x

    # Terminal penalty steers learning toward near-complete liquidation.
    loss = (impact_cost + risk_cost + terminal_penalty * q**2).mean()
    return loss


def train_pov_policy(
    Q: float,
    T: float,
    sigma: float,
    eta: float,
    lam: float,
    phi: float,
    avg_volume: float,
    N: int = 100,
    n_paths: int = 512,
    n_epochs: int = 2000,
    lr: float = 3e-3,
    hidden_size: int = 64,
    seed: int = 42,
    vol_of_vol: float = 0.2,
    terminal_penalty: float = 1e4,
    device: str = "cpu",
) -> tuple[POVPolicy, list[float]]:
    """
    Train a dynamic POV policy with the hard-cap Monte Carlo objective.

    Returns
    -------
    policy : POVPolicy
        Best policy observed during training.
    losses : list[float]
        Loss history by epoch.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = POVPolicy(hidden_size=hidden_size).to(device)
    optimizer = optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-6)

    losses = []
    best_state = None
    best_loss = float("inf")

    for epoch in range(n_epochs):
        policy.train()
        optimizer.zero_grad()

        loss = simulate_pov_hard_cap(
            policy=policy,
            Q=Q,
            T=T,
            sigma=sigma,
            eta=eta,
            lam=lam,
            phi=phi,
            avg_volume=avg_volume,
            N=N,
            n_paths=n_paths,
            vol_of_vol=vol_of_vol,
            terminal_penalty=terminal_penalty,
            device=device,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {
                k: v.detach().clone() for k, v in policy.state_dict().items()
            }

        if (epoch + 1) % 250 == 0:
            print(f"[POV] Epoch {epoch + 1}/{n_epochs} - loss = {loss_value:.8f}")

    if best_state is not None:
        policy.load_state_dict(best_state)

    return policy, losses


@torch.no_grad()
def simulate_pov_trajectories(
    policy: POVPolicy,
    Q: float,
    T: float,
    phi: float,
    avg_volume: float,
    N: int,
    n_paths: int,
    vol_of_vol: float = 0.2,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """
    Simulate trajectories from a trained POV policy for diagnostics and plots.

    Returns
    -------
    dict with numpy arrays:
    - t: shape (N+1,)
    - q_paths: shape (n_paths, N+1)
    - v_paths: shape (n_paths, N)
    - vol_paths: shape (n_paths, N)
    - alpha_paths: shape (n_paths, N)
    """
    policy.eval()

    dt = T / N
    sqrt_dt = np.sqrt(dt)

    q = torch.full((n_paths,), Q, dtype=TORCH_DTYPE, device=device)
    q_paths = torch.zeros((n_paths, N + 1), dtype=TORCH_DTYPE, device=device)
    v_paths = torch.zeros((n_paths, N), dtype=TORCH_DTYPE, device=device)
    vol_paths = torch.zeros((n_paths, N), dtype=TORCH_DTYPE, device=device)
    alpha_paths = torch.zeros((n_paths, N), dtype=TORCH_DTYPE, device=device)

    q_paths[:, 0] = q

    for k in range(N):
        t_norm = torch.full((n_paths,), k / N, dtype=TORCH_DTYPE, device=device)

        eps = torch.randn(n_paths, dtype=TORCH_DTYPE, device=device)
        vol = avg_volume * torch.exp(
            (-0.5 * vol_of_vol**2) * dt + vol_of_vol * sqrt_dt * eps
        )

        q_norm = (q / Q).clamp(0.0, 1.0)
        vol_norm = vol / avg_volume
        alpha = policy(t_norm, q_norm, vol_norm)

        v_cap = phi * vol
        v = alpha * v_cap

        x = torch.minimum(v * dt, q)
        v_eff = x / dt

        q = q - x

        q_paths[:, k + 1] = q
        v_paths[:, k] = v_eff
        vol_paths[:, k] = vol
        alpha_paths[:, k] = alpha

    return {
        "t": np.linspace(0.0, T, N + 1),
        "q_paths": q_paths.cpu().numpy(),
        "v_paths": v_paths.cpu().numpy(),
        "vol_paths": vol_paths.cpu().numpy(),
        "alpha_paths": alpha_paths.cpu().numpy(),
    }


def simulate_pov_hard_cap_stoch_sigma(
    policy: POVPolicy,
    Q: float,
    T: float,
    sigma0: float,
    eta: float,
    lam: float,
    phi: float,
    avg_volume: float,
    N: int,
    n_paths: int,
    vol_of_vol: float = 0.25,
    sigma_of_sigma: float = 0.35,
    terminal_penalty: float = 1e4,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Simulate the extended POV objective with stochastic volume and volatility.

    This is the realistic extension used in Question 3. It keeps the same hard
    POV execution rule as `simulate_pov_hard_cap`,

        v_t <= phi * V_t,

    but replaces the constant risk coefficient sigma^2 by a stochastic path
    sigma_t^2. The policy still observes the same state variables
    (t/T, q/Q, V_t/avg_volume), so the learned rule is a robust POV policy under
    stochastic volatility rather than a volatility-observing feedback policy.

    For each Monte Carlo path and time step, we:
    1) sample stochastic market volume,
    2) update the stochastic volatility path,
    3) query the policy for participation alpha_t,
    4) enforce the hard cap and remaining-inventory constraint,
    5) accumulate impact + stochastic-volatility risk costs,
    6) add a terminal penalty for leftover inventory.

    Returns
    -------
    loss : torch.Tensor (scalar)
        Mean objective across all simulated paths.
    """
    if phi * avg_volume * T < Q:
        raise ValueError(
            "Infeasible POV setup: phi * avg_volume * T < Q, "
            "so full liquidation is impossible on average."
        )

    dt = T / N
    sqrt_dt = np.sqrt(dt)

    q = torch.full((n_paths,), Q, dtype=TORCH_DTYPE, device=device)
    sigma_t = torch.full((n_paths,), sigma0, dtype=TORCH_DTYPE, device=device)

    impact_cost = torch.zeros(n_paths, dtype=TORCH_DTYPE, device=device)
    risk_cost = torch.zeros(n_paths, dtype=TORCH_DTYPE, device=device)

    for k in range(N):
        t_norm = torch.full((n_paths,), k / N, dtype=TORCH_DTYPE, device=device)

        # Lognormal-like positive volume dynamics around avg_volume.
        eps_v = torch.randn(n_paths, dtype=TORCH_DTYPE, device=device)
        vol = avg_volume * torch.exp(
            (-0.5 * vol_of_vol**2) * dt + vol_of_vol * sqrt_dt * eps_v
        )
        vol_norm = vol / avg_volume

        # Positive stochastic volatility path around sigma0.
        eps_sig = torch.randn(n_paths, dtype=TORCH_DTYPE, device=device)
        sigma_t = sigma_t * torch.exp(
            (-0.5 * sigma_of_sigma**2) * dt + sigma_of_sigma * sqrt_dt * eps_sig
        )

        q_norm = (q / Q).clamp(0.0, 1.0)
        alpha = policy(t_norm, q_norm, vol_norm)      # in (0,1)
        v_cap = phi * vol                             # hard POV cap
        v = alpha * v_cap

        # Hard physical constraint: cannot trade more than what remains.
        x = torch.minimum(v * dt, q)
        v_eff = x / dt

        q_mid = q - 0.5 * x
        impact_cost = impact_cost + eta * v_eff**2 * dt
        risk_cost = risk_cost + lam * sigma_t**2 * q_mid**2 * dt

        q = q - x

    # Terminal penalty steers learning toward near-complete liquidation.
    loss = (impact_cost + risk_cost + terminal_penalty * q**2).mean()
    return loss


def train_pov_policy_stoch_sigma(
    Q: float,
    T: float,
    sigma0: float,
    eta: float,
    lam: float,
    phi: float,
    avg_volume: float,
    N: int = 120,
    n_paths: int = 512,
    n_epochs: int = 1200,
    lr: float = 3e-3,
    hidden_size: int = 64,
    seed: int = 123,
    vol_of_vol: float = 0.25,
    sigma_of_sigma: float = 0.35,
    terminal_penalty: float = 1e4,
    device: str = "cpu",
) -> tuple[POVPolicy, list[float]]:
    """
    Train a dynamic POV policy with stochastic volume and volatility.

    Training details:
    - Optimizer: AdamW
    - Gradient clipping: max norm = 1.0
    - Best checkpoint: keeps parameters with lowest observed loss

    Returns
    -------
    policy : POVPolicy
        Best policy observed during training.
    losses : list[float]
        Loss history by epoch.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = POVPolicy(hidden_size=hidden_size).to(device)
    optimizer = optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-6)

    losses = []
    best_state = None
    best_loss = float("inf")

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = simulate_pov_hard_cap_stoch_sigma(
            policy=policy,
            Q=Q,
            T=T,
            sigma0=sigma0,
            eta=eta,
            lam=lam,
            phi=phi,
            avg_volume=avg_volume,
            N=N,
            n_paths=n_paths,
            vol_of_vol=vol_of_vol,
            sigma_of_sigma=sigma_of_sigma,
            terminal_penalty=terminal_penalty,
            device=device,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        value = loss.item()
        losses.append(value)

        if value < best_loss:
            best_loss = value
            best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}

        if (epoch + 1) % 250 == 0:
            print(f"[POV-stoch] Epoch {epoch + 1}/{n_epochs} - loss = {value:.8f}")

    if best_state is not None:
        policy.load_state_dict(best_state)

    return policy, losses
