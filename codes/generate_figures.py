#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CODES_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODES_DIR.parent
NOTEBOOK_PATH = CODES_DIR / "paper_figures.ipynb"


@dataclass(frozen=True)
class CellStep:
    index: int
    suppress_savefig: bool = False


@dataclass(frozen=True)
class FigurePlan:
    name: str
    description: str
    outputs: tuple[str, ...]
    steps: tuple[CellStep, ...]


FIGURE_PLANS = (
    FigurePlan(
        name="steady_state_and_spectral_gap",
        description="Steady-state distributions and spectral-gap comparison.",
        outputs=(
            "images/steady_state_distributions.pdf",
            "images/kuramoto_example_spectral_gap_comparison.pdf",
        ),
        steps=tuple(CellStep(index) for index in (0, 2, 4, 5, 6, 7, 8, 9)),
    ),
    FigurePlan(
        name="kuramoto_control_panels",
        description="Three-panel controlled vs uncontrolled Kuramoto comparison.",
        outputs=("images/kuramoto_control_panels.pdf",),
        steps=tuple(CellStep(index) for index in (0, 10, 11)),
    ),
    FigurePlan(
        name="kuramoto_example_stabilization",
        description="Kuramoto stabilization snapshots.",
        outputs=("images/kuramoto_example_stabilization.pdf",),
        steps=tuple(CellStep(index) for index in (0, 14)),
    ),
    FigurePlan(
        name="kuramoto_example_changing_k",
        description="Multi-K Kuramoto comparison.",
        outputs=("images/kuramoto_example_changing_K.pdf",),
        steps=tuple(CellStep(index) for index in (0, 6, 14, 15)),
    ),
    FigurePlan(
        name="kuramoto_spectra",
        description="Controlled and uncontrolled spectra for selected K values.",
        outputs=("images/kuramoto_spectra.pdf",),
        steps=(
            CellStep(0),
            CellStep(6),
            CellStep(14),
            # Cell 16 depends on the `solver` variable left behind by cell 15
            # in the notebook, so we replay cell 15 but suppress its savefig.
            CellStep(15, suppress_savefig=True),
            CellStep(16),
            CellStep(17),
        ),
    ),
    FigurePlan(
        name="kuramoto_potential_control_heatmap",
        description="Heatmaps for the nonzero-potential Kuramoto example.",
        outputs=("images/kuramoto_potential_control_heatmap.pdf",),
        steps=tuple(CellStep(index) for index in (0, 2, 19, 20, 21)),
    ),
    FigurePlan(
        name="o2_control_different_sigmas",
        description="O(2) controlled and uncontrolled runs across sigma values.",
        outputs=("images/o2_example_control_different_sigmas.pdf",),
        steps=tuple(CellStep(index) for index in (0, 2, 23, 24)),
    ),
    FigurePlan(
        name="von_mises_control_different_sigmas",
        description="Von Mises control example across sigma values.",
        outputs=("images/von_mises_control_different_sigmas.pdf",),
        steps=tuple(CellStep(index) for index in (0, 2, 25)),
    ),
    FigurePlan(
        name="kuramoto_2d_example",
        description="Two-dimensional Kuramoto example.",
        outputs=("images/kuramoto_2d_example.pdf",),
        steps=tuple(CellStep(index) for index in (0, 26, 27)),
    ),
    FigurePlan(
        name="kuramoto_example_k3_controls_cumulative_cost_logscale",
        description="K=3 controls and cumulative-cost figure.",
        outputs=("images/kuramoto_example_k3_controls_cumulative_cost_logscale.pdf",),
        steps=(),
    ),
)

PLAN_BY_NAME = {plan.name: plan for plan in FIGURE_PLANS}
PLAN_ALIASES = {
    output_path.rsplit("/", 1)[-1].removesuffix(".pdf"): plan.name
    for plan in FIGURE_PLANS
    for output_path in plan.outputs
}
PLAN_ALIASES.update({plan.name: plan.name for plan in FIGURE_PLANS})


def load_notebook_cells(path: Path) -> list[dict]:
    notebook = json.loads(path.read_text())
    return notebook["cells"]


@contextmanager
def pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def temporarily_disable_savefig(namespace: dict):
    plt = namespace.get("plt")
    if plt is None:
        yield
        return

    original = plt.savefig
    plt.savefig = lambda *args, **kwargs: None
    try:
        yield
    finally:
        plt.savefig = original


def execute_cell(cell: dict, namespace: dict, cell_index: int, suppress_savefig: bool) -> None:
    if cell.get("cell_type") != "code":
        raise ValueError(f"Notebook cell {cell_index} is not a code cell.")

    source = "".join(cell.get("source", []))
    context = temporarily_disable_savefig(namespace) if suppress_savefig else nullcontext()
    with context:
        exec(compile(source, f"{NOTEBOOK_PATH.name}:cell_{cell_index}", "exec"), namespace)


def run_figure(plan: FigurePlan, notebook_cells: list[dict]) -> None:
    namespace: dict = {"__name__": "__main__"}

    with pushd(CODES_DIR):
        sys.path.insert(0, str(CODES_DIR))
        try:
            for step in plan.steps:
                execute_cell(
                    notebook_cells[step.index],
                    namespace=namespace,
                    cell_index=step.index,
                    suppress_savefig=step.suppress_savefig,
                )
        finally:
            if sys.path and sys.path[0] == str(CODES_DIR):
                sys.path.pop(0)

    plt = namespace.get("plt")
    if plt is not None:
        plt.close("all")


def run_kuramoto_example_k3_controls_cumulative_cost_logscale() -> None:
    import scienceplots
    from main import McKeanVlasovSolver

    plt.style.use("science")

    l_value = 20
    domain_length = 2 * np.pi
    sigma = 0.5
    delta = 1.0
    state_weight = 1000.0
    min_fourier_samples = 2000

    k_values = [1.1, 1.3, 1.5, 2.0, 3.0]
    k_rep = 3.0
    t_rep = 5.0
    t_cost = 5.0
    num_points = 700
    t_view_max = 0.6

    palette = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
    ]

    initial_distribution = lambda x: 0.5 * (1.0 + 0.1 * np.cos(x)) / (2.0 * np.pi)
    potential = lambda x: np.zeros_like(x)

    alpha = [
        lambda x: np.sin(x),
        lambda x: np.cos(x),
        lambda x: np.sin(2 * x),
        lambda x: np.cos(2 * x),
    ]

    grad_alpha = [
        lambda x: np.cos(x),
        lambda x: -np.sin(x),
        lambda x: 2 * np.cos(2 * x),
        lambda x: -2 * np.sin(2 * x),
    ]

    def build_solver(k_value: float) -> McKeanVlasovSolver:
        return McKeanVlasovSolver(
            L=l_value,
            d=domain_length,
            V=potential,
            W=lambda x, k=k_value: -k * np.cos(x),
            mu_0=initial_distribution,
            alpha=alpha,
            grad_alpha=grad_alpha,
            sigma=sigma,
            delta=delta,
            state_weight=state_weight,
            min_fourier_samples=min_fourier_samples,
        )

    def extract_controls(solver: McKeanVlasovSolver, solution) -> np.ndarray:
        return -np.real(solver.B.conj().T @ (solver.Pi @ solution.y))

    def state_cost_density(solver: McKeanVlasovSolver, state_trajectory: np.ndarray) -> np.ndarray:
        scalar_identity = np.eye(solver.N, dtype=solver.M.dtype) * solver.M[0, 0]
        if np.allclose(solver.M, scalar_identity):
            weighted_norm = solver.weighted_L2_norm(state_trajectory)
            return float(np.real(solver.M[0, 0])) * weighted_norm**2
        return np.real(
            np.einsum("nt,nm,mt->t", np.conjugate(state_trajectory), solver.M, state_trajectory)
        )

    def cumulative_trapezoid(t_grid: np.ndarray, values: np.ndarray) -> np.ndarray:
        result = np.zeros_like(values)
        for idx in range(1, len(t_grid)):
            dt = t_grid[idx] - t_grid[idx - 1]
            result[idx] = result[idx - 1] + 0.5 * dt * (values[idx - 1] + values[idx])
        return result

    def cumulative_cost(solver: McKeanVlasovSolver, solution) -> np.ndarray:
        controls = extract_controls(solver, solution)
        state_cost = state_cost_density(solver, solution.y)
        control_cost = np.sum(controls**2, axis=0)
        # J(t) = 0.5 * \int_0^t exp(2 delta s) * (nu ||y(s)||^2 + \sum_j |u_j(s)|^2) ds.
        running_cost = 0.5 * np.exp(2.0 * solver.delta * solution.t) * (state_cost + control_cost)
        return cumulative_trapezoid(solution.t, running_cost)

    plt.rcParams.update(
        {
            "axes.titlesize": 30,
            "axes.labelsize": 17,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.7))
    control_colors = palette[:4]

    rep_solver = build_solver(k_rep)
    rep_t_eval = np.linspace(0.0, t_rep, num_points)
    rep_solution = rep_solver.solve_control_problem(t_span=(0.0, t_rep), t_eval=rep_t_eval)
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
    axes[0].set_xlim(0.0, t_view_max)
    axes[0].set_title("Controls for the $K=3$ case")
    axes[0].set_xlabel("Time $t$")
    axes[0].set_ylabel("Control amplitude")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(ncol=2, frameon=True)

    cost_t_eval = np.linspace(0.0, t_cost, num_points)
    for idx, k_value in enumerate(k_values):
        solver = build_solver(k_value)
        solution = solver.solve_control_problem(t_span=(0.0, t_cost), t_eval=cost_t_eval)
        cost = cumulative_cost(solver, solution)
        axes[1].plot(
            solution.t,
            # The log y-axis cannot display J(0)=0, so we clip only for plotting.
            np.maximum(cost, 1e-16),
            color=palette[idx],
            linewidth=2.8,
            label=fr"$K={k_value:g}$",
        )

    axes[1].set_yscale("log")
    axes[1].set_title("Cumulative cost for varying $K$")
    axes[1].set_xlabel("Time $t$")
    axes[1].set_ylabel(r"$J(t)$")
    axes[1].set_xlim(0.0, t_view_max)
    axes[1].grid(True, which="both", linestyle="--", alpha=0.4)
    axes[1].legend(frameon=True)

    fig.tight_layout()

    output_path = REPO_ROOT / "images" / "kuramoto_example_k3_controls_cumulative_cost_logscale.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate figures from codes/paper_figures.ipynb without editing the notebook."
    )
    parser.add_argument(
        "--figure",
        action="append",
        dest="figures",
        default=[],
        help="Figure identifier to generate. Use --list to see available values. Repeatable.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the available figure identifiers and exit.",
    )
    return parser.parse_args()


def print_available_figures() -> None:
    print("Available figure identifiers:")
    for plan in FIGURE_PLANS:
        outputs = ", ".join(plan.outputs)
        aliases = sorted(
            alias
            for alias, plan_name in PLAN_ALIASES.items()
            if plan_name == plan.name and alias != plan.name
        )
        print(f"  - {plan.name}: {plan.description}")
        print(f"    outputs: {outputs}")
        if aliases:
            print(f"    aliases: {', '.join(aliases)}")


def main() -> int:
    args = parse_args()

    if args.list:
        print_available_figures()
        return 0

    if not args.figures:
        print("No figure requested. Use --figure <name> or --list.", file=sys.stderr)
        return 2

    requested = []
    for name in args.figures:
        if name == "all":
            requested.extend(plan.name for plan in FIGURE_PLANS)
            continue
        resolved_name = PLAN_ALIASES.get(name)
        if resolved_name is None:
            print(f"Unknown figure identifier: {name}", file=sys.stderr)
            print_available_figures()
            return 2
        requested.append(resolved_name)

    seen = set()
    ordered_names = []
    for name in requested:
        if name not in seen:
            ordered_names.append(name)
            seen.add(name)

    notebook_cells = load_notebook_cells(NOTEBOOK_PATH)
    for name in ordered_names:
        plan = PLAN_BY_NAME[name]
        print(f"Generating {plan.name}...")
        try:
            if plan.name == "kuramoto_example_k3_controls_cumulative_cost_logscale":
                run_kuramoto_example_k3_controls_cumulative_cost_logscale()
            else:
                run_figure(plan, notebook_cells)
        except ModuleNotFoundError as exc:
            print(
                "Missing Python dependency while executing the figure "
                f"wrapper: {exc.name}. Install the same packages required by "
                f"{NOTEBOOK_PATH.relative_to(REPO_ROOT)}.",
                file=sys.stderr,
            )
            return 1
        for output in plan.outputs:
            print(f"  wrote {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
