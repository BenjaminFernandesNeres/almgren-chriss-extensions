# Almgren-Chriss Optimal Execution and Extensions

Course project for Electronic Markets, April 2026.

Authors: Benjamin Fernandes Neres, Dorian Deilhes, Ben Komara

## Objective

This project studies the Almgren-Chriss optimal execution framework and several extensions. The central question is how a trader should liquidate a position over a fixed horizon while balancing temporary market impact against price risk.

The program is organized as one runnable notebook, `main.ipynb`, supported by commented Python modules in `src/`. The notebook can be run from top to bottom and reproduces the numerical experiments, plots, and printed diagnostics used in the project.

## What Is Implemented

1. Single-asset Almgren-Chriss execution
   - Closed-form optimal strategies for Implementation Shortfall (IS), Target Close (TC), and TWAP benchmarks.
   - Expected cost and variance computations.
   - Efficient frontiers for IS and TC.
   - Bellman dynamic programming on a discrete inventory grid.

2. Euler scheme and shooting method
   - Explicit Euler discretization of the Euler-Lagrange ODE.
   - Shooting method to recover the initial selling rate that satisfies the terminal inventory constraint.
   - Comparison with the analytical solution.

3. Two-asset liquidation extension
   - Bellman dynamic programming for two correlated assets.
   - Use of a second, more liquid correlated asset as a hedge during liquidation.
   - Comparison between single-asset liquidation and two-asset hedged liquidation.

4. Neural-network bonus section
   - Static neural schedules for IS and TC.
   - Dynamic POV policy under a hard participation cap.
   - Two-asset neural approximation and stochastic-volatility extension.

## Repository Structure

```text
.
|-- main.ipynb
|-- requirements.txt
|-- README.md
`-- src/
    |-- __init__.py
    |-- almgren_chriss.py
    |-- bellman.py
    |-- euler_shooting.py
    |-- neural_network.py
    `-- two_assets.py
```

Report PDFs, LaTeX files, and generated figures are intentionally ignored in this repository. The report is submitted separately by email.

## Installation

Clone the repository and enter the project directory:

```bash
git clone <github-repository-url>
cd almgren-chriss-extensions
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project uses NumPy, Matplotlib, Jupyter Notebook, and PyTorch. A GPU is not required.

## How To Run

Start Jupyter:

```bash
jupyter notebook main.ipynb
```

Then use:

```text
Kernel -> Restart & Run All
```

This runs the whole project in one pass. The notebook is self-contained: parameters are defined inside the notebook, and all functions are imported from `src/`.

You can also execute the notebook from the command line:

```bash
jupyter nbconvert --to notebook --execute main.ipynb --output executed_main.ipynb --ExecutePreprocessor.timeout=-1
```

The full notebook may take several minutes on CPU because Section 4 trains neural-network policies.

## Where To Find The Results

The important results are printed directly in the notebook outputs and displayed as inline plots.

- Section 1 shows the IS, TC, and TWAP liquidation curves, efficient frontiers, and Bellman comparisons.
- Section 2 prints the value of `kappa`, the theoretical and numerical initial selling rates, and the final shooting-method terminal inventory.
- Section 3 prints the two-asset Bellman progress, the minimum and final position in the hedging asset, and a cost/variance comparison table.
- Section 4 prints neural-network training losses, terminal inventory checks, approximation errors, objective-value gaps, POV terminal inventory diagnostics, and two-asset neural comparison errors.

The most useful checks for correctness are:

- Terminal inventories should be close to zero.
- The Euler shooting solution should match the analytical strategy.
- The Bellman solutions should have the same qualitative shape as the closed-form strategies.
- The two-asset strategy should reduce variance by using the correlated asset as a hedge.

## Code Organization

- `src/almgren_chriss.py`: closed-form strategies, costs, variances, and efficient frontiers.
- `src/bellman.py`: one-dimensional Bellman dynamic programming for IS and TC.
- `src/euler_shooting.py`: Euler discretization and shooting method.
- `src/two_assets.py`: two-asset Bellman dynamic programming.
- `src/neural_network.py`: neural-network policies and training utilities.

Each module starts with a description of its objective and contains comments explaining the numerical method and financial interpretation.

## Notes

- No external market data is needed.
- The notebook uses fixed seeds in the neural-network experiments where reproducibility matters.
- The model parameters can be changed directly in the relevant notebook sections.
