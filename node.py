import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data using Logistic Growth Equation
# dy/dt = ry * (1 - y/K)
class LogisticGrowth:
  def __init__(self, r=0.5, K=10.0):
    self.r = r
    self.K = K

  def __call__(self, t, y):
    return self.r * y * (1.0 - y / self.K)

  def analytical_solution(self, y0, t_tensor):
    A = (self.K - y0) / y0
    return self.K / (1 + A * torch.exp(-self.r * t_tensor))

# This network takes state y (population) and outputs derivative dy/dt.
# It has no idea what "r" or "K" are; it has to learn the curve shape.
class ODEFunc(nn.Module):
  def __init__(self):
    super(ODEFunc, self).__init__()
    self.net = nn.Sequential(
      nn.Linear(1, 50),
      nn.Tanh(),
      nn.Linear(50, 50),
      nn.Tanh(),
      nn.Linear(50, 1)
    )

  def forward(self, t, y):
    return self.net(y)

# Use RK4 instead of Euler for better stability and accuracy
def rk4_solve(func, y0, t):
  dt = t[1] - t[0]
  y_trajectory = [y0]
  curr_y = y0

  for i in range(len(t) - 1):
    t_curr = t[i]

    # k1 = f(t, y)
    k1 = func(t_curr, curr_y)

    # k2 = f(t + dt/2, y + dt*k1/2)
    k2 = func(t_curr + dt/2, curr_y + dt * k1/2)

    # k3 = f(t + dt/2, y + dt*k2/2)
    k3 = func(t_curr + dt/2, curr_y + dt * k2/2)

    # k4 = f(t + dt, y + dt*k3)
    k4 = func(t_curr + dt, curr_y + dt * k3)

    # Update rule
    curr_y = curr_y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
  
    y_trajectory.append(curr_y)

  return torch.stack(y_trajectory)

# Generate Data
t_size = 100
t_max = 20
t = torch.linspace(0, t_max, t_size).view(-1, 1)

# Initial Population: 0.1
y0 = torch.tensor([0.1])

# Generate 'true' training data
true_model = LogisticGrowth(r=0.5, K=10.0)
true_y = true_model.analytical_solution(y0, t)

# Now train
func = ODEFunc()
optimizer = optim.Adam(func.parameters(), lr=0.01)
n_iters = 1000

print("Training Neural ODE on Logistic Growth")

for i in range(n_iters):
  optimizer.zero_grad()

  # Forward pass: Integrate NN using RK4
  pred_y = rk4_solve(func, y0, t)

  # Compute loss (MSE)
  loss = torch.mean((pred_y - true_y)**2)
  loss.backward()
  optimizer.step()

  if i % 100 == 0:
    print(f"Iter {i:04d} | Loss {loss.item():.6f}")

print("Training Complete")

# Visualize the data
t_np = t.detach().numpy().flatten()
true_y_np = true_y.detach().numpy().flatten()
pred_y_np = pred_y.detach().numpy().flatten()

plt.figure(figsize=(10,6))
plt.title("Neural ODE Learning Logistic Growth (Bernoulli Example)")

# Plot True Data (Analytical Solution)
plt.plot(t_np, true_y_np, 'g-', label='Analytical Solution (Bernoulli)', linewidth=4, alpha=0.4)

# Plot Neural ODE Prediction
plt.plot(t_np, pred_y_np, 'k--', label='Neural ODE (Learned)', linewidth=2)

plt.axhline(y=10.0, color='r', linestyle=':', label='Carrying Capacity K=10')
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\nVerifying 'Carrying Capacity' behavior")
last_val = pred_y_np[-1]
print(f"Final Population at t={t_max}: {last_val:.4f} (Should be close to K=10.0)")

