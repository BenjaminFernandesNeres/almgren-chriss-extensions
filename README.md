# Almgren-Chriss Model and Extensions
**Electronic Markets — ESILV 2026**
Group project — J. Pu

---

## Overview

This project explores extensions of the **Almgren-Chriss optimal execution model** (2000).
A trader liquidating a large portfolio faces a fundamental trade-off: selling too fast incurs liquidity costs, selling too slowly incurs market risk.

Three sections are covered:
1. Alternative order types (IS, TC, TWAP) — closed-form strategies, efficient frontiers, Bellman grid
2. Euler scheme and shooting method for the Euler-Lagrange boundary value problem
3. Optimal liquidation of two correlated assets via Bellman on a 2D grid

---

## Structure

```
.
├── main.ipynb              # Single notebook — all sections, questions numbered in Markdown cells
├── src/
│   ├── almgren_chriss.py   # Core model: dynamics, mean/variance, optimal strategies (IS, TC, TWAP)
│   ├── bellman.py          # Bellman backward induction on discrete grid (IS, TC, two assets)
│   ├── euler_shooting.py   # Euler-Lagrange ODE system + shooting method
│   └── two_assets.py       # Two-asset extension (correlated Brownian motions)
├── AlmgrenChris2000.pdf
├── Project 1 - Almgren Chris.pdf
└── README.md
```

All logic lives in `src/`. The notebook only calls functions and displays results — no computation inline.

---

## Model Summary

- **Inventory**: $dq_t = -v_t \, dt$, $q_0 = Q$, $q_T = 0$
- **Price**: $dS_t = \sigma \, dW_t - g(v_t) \, dt$ (no permanent impact)
- **Linear temporary impact**: $h(v) = \eta \cdot v$

**Mean-variance objective (IS):**

$$\min_q \int_0^T \left[ -\dot{q}(t)\, h(-\dot{q}(t)) + \lambda \sigma^2 q^2(t) \right] dt$$

| Order | Benchmark |
|-------|-----------|
| IS (Implementation Shortfall) | $S_0$ |
| TC (Target Close) | $S_T$ |
| TWAP | $\frac{1}{T}\int_0^T S_t \, dt$ |

---

## Dependencies

```
numpy
scipy
matplotlib
jupyter
```

```bash
pip install numpy scipy matplotlib jupyter
```

---

## Usage

```bash
jupyter notebook main.ipynb
```

Run all cells top to bottom. Each section is introduced by a Markdown cell indicating the question number.

---

## References

- Almgren, R. & Chriss, N. (2000). *Optimal Execution of Portfolio Transactions*. Journal of Risk.
- J. Pu, *Electronic Markets*, ESILV 2026.
