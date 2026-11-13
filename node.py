"""Neural ODE trained on logistic growth, benchmarked against the closed-form
Bernoulli solution y(t) = K / (1 + A exp(-r t)).

Usage (run from any directory):
    python neural_ode_logistic.py            # main run (seed 0): metrics + figure
    python neural_ode_logistic.py --study 10 # repeat with seeds 0..9

Results are written to ../results/ and the figure to ../figures/.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parent.parent

# Logistic model and data
R, K, Y0 = 0.5, 10.0, 0.1     # growth rate (1/h), carrying capacity, initial population
T_MAX, N_POINTS = 20.0, 100   # observation window (h) and number of samples

# Training
N_ITERS, LR = 1000, 0.01


def logistic_rhs(y):
    """True vector field f(y) = r y (1 - y/K)."""
    return R * y * (1.0 - y / K)


def analytical_solution(t):
    """Closed-form solution from the Bernoulli substitution."""
    A = (K - Y0) / Y0
    return K / (1 + A * torch.exp(-R * t))


class ODEFunc(nn.Module):
    """Learned vector field f_theta(y). It sees only y, never r, K, or t."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1),
        )

    def forward(self, t, y):
        return self.net(y)


def rk4_solve(func, y0, t):
    """Classical fourth-order Runge-Kutta with fixed step t[1] - t[0]."""
    dt = t[1] - t[0]
    y_trajectory = [y0]
    curr_y = y0
    for i in range(len(t) - 1):
        t_curr = t[i]
        k1 = func(t_curr, curr_y)
        k2 = func(t_curr + dt / 2, curr_y + dt * k1 / 2)
        k3 = func(t_curr + dt / 2, curr_y + dt * k2 / 2)
        k4 = func(t_curr + dt, curr_y + dt * k3)
        curr_y = curr_y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        y_trajectory.append(curr_y)
    return torch.stack(y_trajectory)


def train(seed):
    """Train one Neural ODE; returns the model, the data, and the loss history."""
    torch.manual_seed(seed)
    t = torch.linspace(0, T_MAX, N_POINTS).view(-1, 1)
    y0 = torch.tensor([Y0])
    true_y = analytical_solution(t)

    func = ODEFunc()
    optimizer = optim.Adam(func.parameters(), lr=LR)
    history = []
    for i in range(N_ITERS):
        optimizer.zero_grad()
        pred_y = rk4_solve(func, y0, t)
        loss = torch.mean((pred_y - true_y) ** 2)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            history.append((i, loss.item()))

    with torch.no_grad():  # loss of the final, fully trained model
        final_loss = torch.mean((rk4_solve(func, y0, t) - true_y) ** 2).item()
    history.append((N_ITERS, final_loss))
    return func, t, y0, true_y, history


def learned_field(func, y):
    with torch.no_grad():
        return func.net(y.view(-1, 1)).view(-1)


def evaluate(func, t, y0, true_y):
    """Error metrics, check of the RK4 discretization, and the learned equilibria."""
    with torch.no_grad():
        pred_y = rk4_solve(func, y0, t)
        # Re-integrate the learned field with 20 substeps per sampling interval.
        sub = 20
        t_fine = torch.linspace(0, T_MAX, sub * (N_POINTS - 1) + 1).view(-1, 1)
        y_fine = rk4_solve(func, y0, t_fine).view(-1)
        # Long-time behavior of the learned model.
        t_long = torch.linspace(0, 100.0, 10001).view(-1, 1)
        y_long = rk4_solve(func, y0, t_long).view(-1)

    err = (pred_y - true_y).view(-1)
    i_max = int(err.abs().argmax())
    i10 = sub * (N_POINTS - 1) // 2          # t_fine[i10] == 10 exactly

    above = torch.nonzero(y_fine >= K).view(-1)
    if len(above) > 0 and int(above[0]) > 0:
        j = int(above[0])
        ta, tb = t_fine[j - 1].item(), t_fine[j].item()
        ya, yb = y_fine[j - 1].item(), y_fine[j].item()
        t_cross = ta + (K - ya) * (tb - ta) / (yb - ya)
    else:
        t_cross = None

    # Zeros of the learned vector field on 0 <= y <= 30.
    y_grid = torch.linspace(0, 30, 30001)
    f = learned_field(func, y_grid)
    flips = torch.nonzero(f[:-1] * f[1:] < 0).view(-1)
    zeros = []
    for i in flips.tolist():
        y_zero = y_grid[i] - f[i] * (y_grid[i + 1] - y_grid[i]) / (f[i + 1] - f[i])
        zeros.append({"y": y_zero.item(), "stable": bool(f[i] > 0)})

    visited = (y_grid >= Y0) & (y_grid <= true_y.max())
    field_err = (f - logistic_rhs(y_grid))[visited].abs()

    return {
        "final_mse": torch.mean(err ** 2).item(),
        "rmse": torch.sqrt(torch.mean(err ** 2)).item(),
        "max_abs_err": err.abs().max().item(),
        "t_at_max_err": t[i_max].item(),
        "y_hat_10": y_fine[i10].item(),
        "y_true_10": analytical_solution(torch.tensor(10.0)).item(),
        "y_hat_20": pred_y[-1].item(),
        "y_true_20": true_y[-1].item(),
        "t_cross_K": t_cross,
        "rk4_coarse_vs_fine": (y_fine[::sub] - pred_y.view(-1)).abs().max().item(),
        "f_theta_0": learned_field(func, torch.tensor([0.0])).item(),
        "f_theta_K": learned_field(func, torch.tensor([K])).item(),
        "max_field_err_on_data_range": field_err.max().item(),
        "learned_zeros": zeros,
        "y_hat_100": y_long[-1].item(),
        "trajectory_monotone": bool((y_fine[1:] >= y_fine[:-1]).all()),
    }


def run(seed):
    torch.set_num_threads(1)  # single-threaded CPU run, for reproducibility
    func, t, y0, true_y, history = train(seed)
    return seed, func, t, y0, true_y, history, evaluate(func, t, y0, true_y)


def run_summary(seed):
    seed, _, _, _, _, history, metrics = run(seed)
    return {"seed": seed, **metrics, "loss_history": history}


def plot(func, t, y0, true_y, metrics, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    with torch.no_grad():
        pred_y = rk4_solve(func, y0, t).view(-1)
    t_dense = torch.linspace(0, T_MAX, 500)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.7), constrained_layout=True)

    ax1.axhline(K, color="0.6", ls=":", lw=1, label=f"$K = {K:g}$")
    ax1.plot(t_dense, analytical_solution(t_dense), color="#2b6cb0", lw=2.5,
             alpha=0.6, label="Exact solution")
    ax1.plot(t.view(-1), pred_y, color="black", ls="--", lw=1.4, label="Neural ODE")
    ax1.set(xlabel="$t$ (h)", ylabel="$y(t)$", xlim=(0, T_MAX))
    ax1.set_title("(a) Trajectories", loc="left", fontsize=9)
    ax1.legend(frameon=False, loc="lower right")

    stable = [z["y"] for z in metrics["learned_zeros"] if z["stable"]]
    y_max = max([14.0] + [s + 2.0 for s in stable if s < 28])
    y_plot = torch.linspace(0, y_max, 600)
    ax2.axvspan(Y0, true_y.max().item(), color="0.92", lw=0)  # range of the data
    ax2.axhline(0, color="0.6", lw=0.8)
    ax2.plot(y_plot, logistic_rhs(y_plot), color="#2b6cb0", lw=2.5, alpha=0.6,
             label="True $f(y)$")
    ax2.plot(y_plot, learned_field(func, y_plot), color="black", ls="--", lw=1.4,
             label=r"Learned $f_\theta(y)$")
    ax2.plot([K], [0], "o", color="#2b6cb0", ms=5)
    for s in stable:
        if s <= y_max:
            ax2.plot([s], [0], "o", mfc="white", mec="black", ms=5)
    ax2.set(xlabel="$y$", ylabel="$dy/dt$", xlim=(0, y_max), ylim=(-1.5, 1.6))
    ax2.set_title("(b) Vector fields", loc="left", fontsize=9)
    ax2.legend(frameon=False, loc="lower left")

    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--study", type=int, default=0,
                        help="if > 0, train with seeds 0..N-1 and summarize")
    args = parser.parse_args()
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)

    if args.study > 0:
        with Pool(min(args.study, 6)) as pool:
            rows = pool.map(run_summary, range(args.study))
        (ROOT / "results" / "seed_study.json").write_text(json.dumps(rows, indent=2))
        for r in rows:
            zs = ", ".join(f"{z['y']:.3f}{'s' if z['stable'] else 'u'}" for z in r["learned_zeros"])
            print(f"seed {r['seed']}: mse {r['final_mse']:.4f}  rmse {r['rmse']:.4f}  "
                  f"maxerr {r['max_abs_err']:.3f}  y20 {r['y_hat_20']:.3f}  "
                  f"fK {r['f_theta_K']:+.4f}  zeros [{zs}]")
        return

    seed, func, t, y0, true_y, history, metrics = run(args.seed)
    for i, loss in history:
        print(f"Iter {i:04d} | Loss {loss:.6f}")
    print(json.dumps(metrics, indent=2))
    (ROOT / "results" / f"seed{seed}.json").write_text(
        json.dumps({"seed": seed, "torch": torch.__version__, "loss_history": history,
                    **metrics}, indent=2))
    plot(func, t, y0, true_y, metrics, ROOT / "figures" / "neural-ode-fit.pdf")


if __name__ == "__main__":
    main()
