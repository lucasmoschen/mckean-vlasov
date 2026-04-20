#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import runpy
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


def run_standalone_script(script_name: str) -> None:
    with pushd(CODES_DIR):
        sys.path.insert(0, str(CODES_DIR))
        try:
            runpy.run_path(str(CODES_DIR / script_name), run_name="__main__")
        finally:
            if sys.path and sys.path[0] == str(CODES_DIR):
                sys.path.pop(0)


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
        print(f"  - {plan.name}: {plan.description}")
        print(f"    outputs: {outputs}")


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
        if name not in PLAN_BY_NAME:
            print(f"Unknown figure identifier: {name}", file=sys.stderr)
            print_available_figures()
            return 2
        requested.append(name)

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
                run_standalone_script("kuramoto_example_changing_K.py")
            else:
                run_figure(plan, notebook_cells)
        except ModuleNotFoundError as exc:
            print(
                "Missing Python dependency while executing the notebook-backed figure "
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
