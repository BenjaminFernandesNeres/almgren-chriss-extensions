"""
Almgren-Chriss continuous-time model — core module.

Covers the three order types studied in the project:
  - IS  (Implementation Shortfall): benchmark = S_0
  - TC  (Target Close):             benchmark = S_T
  - TWAP:                           benchmark = (1/T) * integral of S_t dt

State variables (continuous time):
  q_t  : remaining inventory,  q_0 = Q,  q_T = 0
  v_t  : selling rate,         v_t = -dq_t/dt
  S_t  : asset price,          dS_t = sigma * dW_t  (no permanent impact)
  X_t  : cash,                 dX_t = v_t * (S_t - h(v_t)) dt

Temporary market impact (linear): h(v) = eta * v

Mean-variance objective:
  maximise  E[payoff] - lambda * Var[payoff]
  <=>
  minimise  int_0^T [ eta * v_t^2 + lambda * sigma^2 * penalty(q_t) ] dt

Authors: Benjamin Fernandes Neres, Dorian Deilhes, Yesman
Date: April 2026
"""

import numpy as np


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _kappa(lam: float, sigma: float, eta: float) -> float:
    """
    Compute kappa = sqrt(lambda * sigma^2 / eta).

    This characteristic rate controls the shape of the optimal trading curve:
    large kappa -> aggressive front-loading (IS) or back-loading (TC).
    """
    return np.sqrt(lam * sigma**2 / eta)


# ---------------------------------------------------------------------------
# Optimal static strategies (closed-form, mean-variance)
# ---------------------------------------------------------------------------

def optimal_strategy_is(
    Q: float,
    T: float,
    sigma: float,
    eta: float,
    lam: float,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimal IS strategy in the mean-variance framework (closed form).

    The mean-variance problem for IS reduces to:
        min_q  int_0^T [ eta * q_dot^2 + lambda * sigma^2 * q^2 ] dt
        s.t.   q(0) = Q,  q(T) = 0

    Euler-Lagrange equation: eta * q_ddot = lambda * sigma^2 * q
    => q_ddot = kappa^2 * q,  kappa = sqrt(lambda * sigma^2 / eta)

    Closed-form solution:
        q(t) = Q * sinh(kappa * (T - t)) / sinh(kappa * T)
        v(t) = Q * kappa * cosh(kappa * (T - t)) / sinh(kappa * T)

    When kappa -> 0 (risk-neutral), the solution degenerates to TWAP.

    Parameters
    ----------
    Q : float
        Initial inventory (number of shares to liquidate).
    T : float
        Liquidation horizon.
    sigma : float
        Asset volatility.
    eta : float
        Temporary market impact coefficient.
    lam : float
        Risk-aversion coefficient (lambda >= 0).
    n_points : int
        Number of time grid points for output arrays.

    Returns
    -------
    t : np.ndarray, shape (n_points,)
    q : np.ndarray, shape (n_points,)  — optimal inventory path
    v : np.ndarray, shape (n_points,)  — optimal selling rate
    """
    t = np.linspace(0.0, T, n_points)
    kap = _kappa(lam, sigma, eta)

    if kap < 1e-10:
        # Risk-neutral limit: linear (TWAP) schedule
        q = Q * (T - t) / T
        v = np.full(n_points, Q / T)
    else:
        q = Q * np.sinh(kap * (T - t)) / np.sinh(kap * T)
        v = Q * kap * np.cosh(kap * (T - t)) / np.sinh(kap * T)

    return t, q, v


def optimal_strategy_tc(
    Q: float,
    T: float,
    sigma: float,
    eta: float,
    lam: float,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimal TC (Target Close) strategy in the mean-variance framework (closed form).

    Benchmark = S_T.  The variance of the TC payoff involves (q_t - Q),
    because the tracking error relative to S_T is driven by cumulative sales.

    Setting p(t) = q(t) - Q gives BC p(0) = 0, p(T) = -Q, same ODE as IS.

    Closed-form solution:
        q(t) = Q * (1 - sinh(kappa * t) / sinh(kappa * T))
        v(t) = Q * kappa * cosh(kappa * t) / sinh(kappa * T)

    Key intuition: TC trades slowly at the start and accelerates towards close,
    to track the final price as closely as possible.

    Parameters
    ----------
    Q, T, sigma, eta, lam, n_points : see optimal_strategy_is.

    Returns
    -------
    t, q, v : np.ndarray
    """
    t = np.linspace(0.0, T, n_points)
    kap = _kappa(lam, sigma, eta)

    if kap < 1e-10:
        q = Q * (T - t) / T
        v = np.full(n_points, Q / T)
    else:
        q = Q * (1.0 - np.sinh(kap * t) / np.sinh(kap * T))
        v = Q * kap * np.cosh(kap * t) / np.sinh(kap * T)

    return t, q, v


def optimal_strategy_twap(
    Q: float,
    T: float,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimal TWAP strategy in the mean-variance framework (closed form).

    Benchmark = (1/T) * int_0^T S_t dt.

    The variance term penalises deviations from Q*(T-t)/T, which is exactly the
    linear (TWAP) schedule. The optimal strategy is therefore always TWAP,
    regardless of lambda — the efficient frontier degenerates to a single point.

    Parameters
    ----------
    Q : float
        Initial inventory.
    T : float
        Liquidation horizon.
    n_points : int
        Number of time grid points.

    Returns
    -------
    t, q, v : np.ndarray
    """
    t = np.linspace(0.0, T, n_points)
    q = Q * (T - t) / T
    v = np.full(n_points, Q / T)
    return t, q, v


# ---------------------------------------------------------------------------
# Expected cost and variance
# ---------------------------------------------------------------------------

def cost_and_variance_is(
    q: np.ndarray,
    v: np.ndarray,
    t: np.ndarray,
    sigma: float,
    eta: float,
) -> tuple[float, float]:
    """
    Compute expected cost and variance for an IS strategy.

    IS payoff = X_T - Q * S_0

    E[payoff] = -eta * int_0^T v_t^2 dt   (liquidity cost, always negative)
    Var[payoff] = sigma^2 * int_0^T q_t^2 dt

    Integrals computed via the trapezoidal rule.

    Parameters
    ----------
    q, v, t : np.ndarray
        Inventory, selling rate, time grid.
    sigma : float
        Asset volatility.
    eta : float
        Temporary impact coefficient.

    Returns
    -------
    expected_cost : float  (positive = loss to the trader)
    variance : float
    """
    expected_cost = eta * np.trapz(v**2, t)
    variance = sigma**2 * np.trapz(q**2, t)
    return expected_cost, variance


def cost_and_variance_tc(
    q: np.ndarray,
    v: np.ndarray,
    t: np.ndarray,
    sigma: float,
    eta: float,
    Q: float,
) -> tuple[float, float]:
    """
    Compute expected cost and variance for a TC strategy.

    TC payoff = X_T - Q * S_T

    E[payoff] = -eta * int_0^T v_t^2 dt
    Var[payoff] = sigma^2 * int_0^T (q_t - Q)^2 dt

    Parameters
    ----------
    q, v, t : np.ndarray
    sigma, eta : float
    Q : float  — initial inventory (appears in TC variance formula)

    Returns
    -------
    expected_cost : float
    variance : float
    """
    expected_cost = eta * np.trapz(v**2, t)
    variance = sigma**2 * np.trapz((q - Q)**2, t)
    return expected_cost, variance


# ---------------------------------------------------------------------------
# Efficient frontiers
# ---------------------------------------------------------------------------

def efficient_frontier_is(
    Q: float,
    T: float,
    sigma: float,
    eta: float,
    lambdas: np.ndarray,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the IS efficient frontier by sweeping over risk-aversion values.

    Parameters
    ----------
    Q, T, sigma, eta : float — model parameters.
    lambdas : np.ndarray    — risk-aversion values (increasing = more risk-averse).
    n_points : int          — grid resolution per strategy.

    Returns
    -------
    costs : np.ndarray     — expected liquidation costs along the frontier.
    variances : np.ndarray — variances along the frontier.
    """
    costs = np.zeros(len(lambdas))
    variances = np.zeros(len(lambdas))

    for i, lam in enumerate(lambdas):
        t, q, v = optimal_strategy_is(Q, T, sigma, eta, lam, n_points)
        costs[i], variances[i] = cost_and_variance_is(q, v, t, sigma, eta)

    return costs, variances


def efficient_frontier_tc(
    Q: float,
    T: float,
    sigma: float,
    eta: float,
    lambdas: np.ndarray,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the TC efficient frontier by sweeping over risk-aversion values.

    Parameters
    ----------
    Q, T, sigma, eta : float
    lambdas : np.ndarray
    n_points : int

    Returns
    -------
    costs : np.ndarray
    variances : np.ndarray
    """
    costs = np.zeros(len(lambdas))
    variances = np.zeros(len(lambdas))

    for i, lam in enumerate(lambdas):
        t, q, v = optimal_strategy_tc(Q, T, sigma, eta, lam, n_points)
        costs[i], variances[i] = cost_and_variance_tc(q, v, t, sigma, eta, Q)

    return costs, variances
