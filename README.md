# Learning Logistic Growth with Neural ODEs

**Comparing a Learned Model with an Exact Solution**  
Sebastian Chegini

This project explores how a Neural Ordinary Differential Equation (Neural ODE) learns population growth from data. The network predicts the population's rate of change. A numerical ODE solver uses that prediction to construct a growth curve.

The logistic equation provides an exact solution for comparison. This lets us examine how closely the model fits the data. It also lets us check whether the model learns the correct long-term behavior.

**[Read the paper](main.pdf)** · **[View the LaTeX source](main.tex)** · **[View the implementation](node.py)**

## The experiment

The logistic growth model is

$$
\frac{dy}{dt}=ry\left(1-\frac{y}{K}\right),\qquad y(0)=y_0.
$$

We use a growth rate of $r=0.5$ per hour, a carrying capacity of $K=10$, and an initial population of $y_0=0.1$. Population is measured in arbitrary units.

The paper derives the exact solution using the Bernoulli substitution:

$$
y(t)=\frac{10}{1+99e^{-0.5t}}.
$$

We sample this solution at 100 equally spaced times between 0 and 20 hours. These synthetic data contain no added noise.

The Neural ODE uses the following setup:

- **Network:** one population input, two hidden layers of 50 units, and one derivative output.
- **Activations:** hyperbolic tangent functions in the hidden layers.
- **ODE solver:** classical fourth-order Runge–Kutta (RK4), with a training step size of $20/99$ hours.
- **Training:** 1,000 Adam iterations with a learning rate of 0.01.
- **Loss:** mean squared error over the full trajectory.
- **Repeated runs:** random seeds 0 through 9.

The network receives the current population as its input. It is not given the growth rate, carrying capacity, or logistic formula. The initial population is supplied to the solver.

## Main findings

The model follows the overall shape of the logistic growth curve. However, it does not recover the true carrying capacity in these runs.

For the reference run, using seed 0:

- The root-mean-square error is approximately **0.196 population units**.
- At 10 hours, the model predicts **5.765**, compared with the exact value of **5.999**.
- The predicted population crosses the carrying capacity at approximately **19.13 hours**.
- The learned stable equilibrium is near **10.35**, rather than 10.

All ten runs exceed the true carrying capacity before 20 hours. Their RMSE values range from approximately 0.118 to 0.348. Nine runs have a detected stable equilibrium above 10. No sign-changing zero is detected for seed 5 over the searched interval, $0\le y\le30$.

The main lesson is that fitting an observed trajectory does not necessarily mean recovering the underlying differential equation.

## Run the experiments

The saved numerical results were independently reproduced with **Python 3.13.1** and **PyTorch 2.10.0**. Matplotlib is also needed to regenerate the comparison figure.

From the repository root, create a virtual environment and install the dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 matplotlib
```

Train the reference model and generate its results and figure:

```sh
python code/neural_ode_logistic.py
```

Repeat the experiment for all ten seeds:

```sh
python code/neural_ode_logistic.py --study 10
```

Run a different individual seed:

```sh
python code/neural_ode_logistic.py --seed 3
```

Each experiment uses one CPU thread. The ten-seed study runs several experiments in parallel.

These commands replace the corresponding files in `results/`. An individual run also replaces `figures/neural-ode-fit.pdf` with its own comparison plot. The supplied paper and figure use seed 0.

## Compile the paper

Upload the project to Overleaf and select `main.tex` as the main document. To compile locally with a LaTeX installation, run:

```sh
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Alternatively, use Tectonic:

```sh
tectonic main.tex
```

Regenerating an experiment figure does not update the compiled paper automatically. Compile the LaTeX source again to include the new figure. The numerical values in the manuscript are written explicitly and must be updated separately if the experiment changes.

## Repository contents

```text
.
├── README.md
├── main.tex                       # Paper source
├── main.pdf                       # Compiled paper
├── references.bib                 # Bibliography
├── code/
│   └── neural_ode_logistic.py      # Training, evaluation, and plotting
├── figures/
│   └── neural-ode-fit.pdf          # Reference trajectory and vector fields
├── results/
│   ├── seed0.json                 # Reference metrics and loss history
│   └── seed_study.json            # Results for seeds 0–9
└── verification.json              # Independent reproduction summary
```

## Scope and limitations

The data come from an exact mathematical solution. The experiment does not use measured bacterial populations.

Training and evaluation use the same sample points. The reported errors therefore measure fit to the training trajectory. They do not establish accuracy for new initial conditions.

The experiment uses a fixed training budget. Different training settings could change the results.

The equilibrium search covers a finite interval. It can miss zeros where the learned growth rate does not change sign. The RK4 step-size comparison is a numerical consistency check, rather than a certified error bound.

All ten reruns matched the supplied numerical summaries and recorded loss histories in the verification environment. Exact agreement across other machines or library versions is not guaranteed.

## References

- Chen et al., [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366), 2018.
- He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), CVPR 2016.
- Kingma and Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), ICLR 2015.
