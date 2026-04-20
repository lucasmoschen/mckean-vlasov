import os
import tempfile
from pathlib import Path

CACHE_ROOT = Path(tempfile.gettempdir()) / "mckean-vlasov-mpl"
MPLCONFIGDIR = CACHE_ROOT / ".mplconfig"
XDG_CACHE_HOME = CACHE_ROOT / ".cache"
CACHE_ROOT.mkdir(exist_ok=True)
MPLCONFIGDIR.mkdir(exist_ok=True)
XDG_CACHE_HOME.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main import McKeanVlasovSolver


L = 20
D = 2 * np.pi
SIGMA = 0.5
DELTA = 1.0
STATE_WEIGHT = 1000.0
MIN_FOURIER_SAMPLES = 2000

K_VALUES = [1.1, 1.3, 1.5, 2.0, 3.0]
K_REP = 3.0
T_REP = 5.0
T_COST = 5.0
NUM_POINTS = 700
T_VIEW_MAX = 0.6

PALETTE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
]

# Match the unstable-uniform setup used in the earlier K=5 control panel.
INITIAL_DISTRIBUTION = lambda x: 0.5 * (1.0 + 0.1 * np.cos(x)) / (2.0 * np.pi)
V = lambda x: np.zeros_like(x)

ALPHA = [
    lambda x: np.sin(x),
    lambda x: np.cos(x),
    lambda x: np.sin(2 * x),
    lambda x: np.cos(2 * x),
]

GRAD_ALPHA = [
    lambda x: np.cos(x),
    lambda x: -np.sin(x),
    lambda x: 2 * np.cos(2 * x),
    lambda x: -2 * np.sin(2 * x),
]


def build_solver(k_value):
    return McKeanVlasovSolver(
        L=L,
        d=D,
        V=V,
        W=lambda x, k=k_value: -k * np.cos(x),
        mu_0=INITIAL_DISTRIBUTION,
        alpha=ALPHA,
        grad_alpha=GRAD_ALPHA,
        sigma=SIGMA,
        delta=DELTA,
        state_weight=STATE_WEIGHT,
        min_fourier_samples=MIN_FOURIER_SAMPLES,
    )


def extract_controls(solver, solution):
    return -np.real(solver.B.conj().T @ (solver.Pi @ solution.y))


def state_cost_density(solver, state_trajectory):
    scalar_identity = np.eye(solver.N, dtype=solver.M.dtype) * solver.M[0, 0]
    if np.allclose(solver.M, scalar_identity):
        weighted_norm = solver.weighted_L2_norm(state_trajectory)
        return float(np.real(solver.M[0, 0])) * weighted_norm**2
    return np.real(
        np.einsum("nt,nm,mt->t", np.conjugate(state_trajectory), solver.M, state_trajectory)
    )


def cumulative_trapezoid(t_grid, values):
    result = np.zeros_like(values)
    for idx in range(1, len(t_grid)):
        dt = t_grid[idx] - t_grid[idx - 1]
        result[idx] = result[idx - 1] + 0.5 * dt * (values[idx - 1] + values[idx])
    return result


def cumulative_cost(solver, solution):
    controls = extract_controls(solver, solution)
    state_cost = state_cost_density(solver, solution.y)
    control_cost = np.sum(controls**2, axis=0)
    # J(t) = 0.5 * \int_0^t exp(2 delta s) * (nu ||y(s)||^2 + \sum_j |u_j(s)|^2) ds.
    running_cost = 0.5 * np.exp(2.0 * solver.delta * solution.t) * (state_cost + control_cost)
    return cumulative_trapezoid(solution.t, running_cost)


def main():
    plt.rcParams.update(
        {
            "axes.titlesize": 20,
            "axes.labelsize": 17,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.7))
    control_colors = PALETTE[:4]

    rep_solver = build_solver(K_REP)
    rep_t_eval = np.linspace(0.0, T_REP, NUM_POINTS)
    rep_solution = rep_solver.solve_control_problem(t_span=(0.0, T_REP), t_eval=rep_t_eval)
    rep_controls = extract_controls(rep_solver, rep_solution)

    for idx, color in enumerate(control_colors):
        axes[0].plot(
            rep_solution.t,
            rep_controls[idx],
            color=color,
            linewidth=2.5,
            label=fr"$u_{idx + 1}(t)$",
        )

    axes[0].axhline(0.0, color="0.3", linewidth=1.0, linestyle="--")
    axes[0].set_xlim(0.0, T_VIEW_MAX)
    axes[0].set_title("Controls for the $K=3$ case")
    axes[0].set_xlabel("Time $t$")
    axes[0].set_ylabel("Control amplitude")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(ncol=2, frameon=True)

    cost_t_eval = np.linspace(0.0, T_COST, NUM_POINTS)
    for idx, k_value in enumerate(K_VALUES):
        solver = build_solver(k_value)
        solution = solver.solve_control_problem(t_span=(0.0, T_COST), t_eval=cost_t_eval)
        cost = cumulative_cost(solver, solution)
        axes[1].plot(
            solution.t,
            # The log y-axis cannot display J(0)=0, so we clip only for plotting.
            np.maximum(cost, 1e-16),
            color=PALETTE[idx],
            linewidth=2.8,
            label=fr"$K={k_value:g}$",
        )

    axes[1].set_yscale("log")
    axes[1].set_title("Cumulative cost for varying $K$")
    axes[1].set_xlabel("Time $t$")
    axes[1].set_ylabel(r"$J(t)$")
    axes[1].set_xlim(0.0, T_VIEW_MAX)
    axes[1].grid(True, which="both", linestyle="--", alpha=0.4)
    axes[1].legend(frameon=True)

    fig.tight_layout()

    output_path = (
        Path(__file__).resolve().parents[1]
        / "images"
        / "kuramoto_example_k3_controls_cumulative_cost_logscale.pdf"
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
