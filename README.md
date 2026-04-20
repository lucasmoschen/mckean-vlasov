# Control of McKean-Vlasov PDEs

Code accompanying *Linearization-Based Feedback Stabilization of McKean-Vlasov PDEs* by Dante Kalise, Lucas M. Moschen, and Grigorios A. Pavliotis.

Paper: [arXiv:2507.12411](https://arxiv.org/abs/2507.12411)

## Abstract

Adapted from the arXiv abstract: we develop a feedback control framework for McKean-Vlasov PDEs on the torus. The goal is to stabilize a prescribed stationary distribution, or to accelerate convergence toward one, by means of a time-dependent control potential. The analysis rewrites the controlled equation in a weighted zero-mean space, applies a ground-state transform, and derives a Riccati feedback law from the linearized dynamics. The numerical section covers the noisy Kuramoto model, the O(2) spin model in a magnetic field, and a von Mises interaction potential.

## Repository Layout

- `codes/`: solver code, plotting scripts, and the original plotting notebooks.
- `images/`: generated figure outputs.
- `notes/`: paper notes and bibliography sources.
- `abstracts/`: abstract sources.

## Core Files

- `codes/main.py`: main solver implementation.
- `codes/fourier_utils.py`: Fourier helper routines.
- `codes/paper_figures.ipynb`: main notebook used for the paper figures.
- `codes/generate_figures.py`: command-line entry point for the figure set.
- `codes/kuramoto_example_changing_K.py`: standalone script for the `K=3` controls/cost figure, now exposed through `codes/generate_figures.py`.

## Figures

The main plotting notebook is `codes/paper_figures.ipynb`. For command-line runs, use `codes/generate_figures.py`.

List the available figure identifiers:

```bash
python codes/generate_figures.py --list
```

Generate one figure group:

```bash
python codes/generate_figures.py --figure kuramoto_control_panels
```

Generate several figure groups:

```bash
python codes/generate_figures.py \
  --figure kuramoto_example_changing_k \
  --figure kuramoto_spectra
```

Generate the `K=3` controls and cumulative-cost figure:

```bash
python codes/generate_figures.py --figure kuramoto_example_k3_controls_cumulative_cost_logscale
```

Generate everything exposed by the wrapper:

```bash
python codes/generate_figures.py --figure all
```

The script writes outputs into `images/` using the same filenames as the notebook.