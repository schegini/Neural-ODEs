# Suggested middle-ground revision

This is a separate suggested revision of Sebastian Chegini's Neural ODE project. The supplied original projects were not changed.

The draft combines the original paper's step-by-step mathematical explanation with the publication project's explicit methods and reproducible numerical results. It uses the publication project's seeded experiments, not the older unseeded screenshot, and avoids claims of untested generalization or certified numerical error bounds.

## Contents

- `main.tex`: complete revised manuscript.
- `references.bib`: three retained primary-source references.
- `figures/neural-ode-fit.pdf`: figure supplied with the publication project.
- `code/neural_ode_logistic.py`: unchanged publication-project implementation.
- `results/seed0.json` and `results/seed_study.json`: unchanged supplied results.
- `verification.json`: independent rerun summary from this review.

The full comparison is supplied separately as `paper-comparison.md`.

## Compile

Upload this folder's contents to Overleaf and select `main.tex`, or run:

```sh
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Tectonic can also compile it with `tectonic main.tex`.

## Reproduce the experiment

The independent review used Python 3.13.1 and PyTorch 2.10.0. The figure-generation script also requires Matplotlib. From this folder, run:

```sh
python code/neural_ode_logistic.py
python code/neural_ode_logistic.py --study 10
```

These commands overwrite the corresponding results and figure files. The second uses multiple processes, with one CPU thread per experiment.

All ten independent reruns matched the numerical fields and recorded loss histories in the supplied result files in this environment. This verifies reproducibility here; it is not a guarantee of bitwise agreement across other library versions or machines.

## Editorial choices

The draft removes the contributions list, section roadmap, theorem wrapper, classification codes, and printed source-code appendix. It retains intermediate derivation steps, the vector-field comparison, repeated runs, and limits of the experiment. It does not assert that seed 0 was selected before inspecting results, and it treats the finite-grid equilibrium search and RK4 step comparison as numerical checks.

The three background references were checked against their primary-source arXiv records. The separate comparison documents specific changes and limitations in more detail.
