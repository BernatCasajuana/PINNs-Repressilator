# Prediction of Repressilator dynamics using PINNs and DeepXDE

# %% Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import deepxde as dde

# %% Define parameters, initial conditions, and time domain (normalized)
beta = 10
n = 3
x0 = np.array([1, 1, 1.2])
n_points = 1000
t = np.linspace(0, 1, n_points)[:, None]
t_max = 20
t_eval = t * t_max

# %% Simulation with ODEINT
def protein_repressilator_rhs(x, t, beta, n):
    x1, x2, x3 = x
    return [
        beta / (1 + x3 ** n) - x1,
        beta / (1 + x1 ** n) - x2,
        beta / (1 + x2 ** n) - x3,
    ]

x_ode = scipy.integrate.odeint(protein_repressilator_rhs, x0, t_eval.flatten(), args=(beta, n))

# Save as .npz file for inverse problem
np.savez("Repressilator.npz", t=t, y=x_ode)

# %% PINN simulation setup
scale = np.max(x_ode) # Normalization scale based on odeint solution

# Geometry of the problem (time domain)
geom = dde.geometry.TimeDomain(0, 1)

# Define ODE system with normalization
def ode_system(x, y):
    y1, y2, y3 = y[:, 0:1] * scale, y[:, 1:2] * scale, y[:, 2:3] * scale
    dy1 = dde.grad.jacobian(y, x, i=0, j=0) * scale / t_max
    dy2 = dde.grad.jacobian(y, x, i=1, j=0) * scale / t_max
    dy3 = dde.grad.jacobian(y, x, i=2, j=0) * scale / t_max

    eq1 = dy1 - (beta / (1 + y3**n) - y1)
    eq2 = dy2 - (beta / (1 + y1**n) - y2)
    eq3 = dy3 - (beta / (1 + y2**n) - y3)

    # Loss calculation with scaling for stability
    return [eq1 / scale, eq2 / scale, eq3 / scale]

# Initial conditions
def boundary(_, on_initial):
    return on_initial
ic1 = dde.icbc.IC(geom, lambda x: 1, boundary, component=0)
ic2 = dde.icbc.IC(geom, lambda x: 1, boundary, component=1)
ic3 = dde.icbc.IC(geom, lambda x: 1.2, boundary, component=2)

# Implement observed data from odeint solution
indices = np.linspace(0, n_points - 1, 200).astype(int)
observed_t = t[indices]
observed_y = (x_ode / scale)[indices]

# Define observed boundary conditions
observed_bcs = [
    dde.icbc.PointSetBC(observed_t, observed_y[:, i:i+1], component=i)
    for i in range(3)]

# Problem setup
data = dde.data.PDE(geom, ode_system, [ic1, ic2, ic3, *observed_bcs], num_domain=2000, num_boundary=2, num_test=1000)

# Neural network architecture
layer_size = [1] + [50] * 3 + [3]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

# %% Compile and train
# Define data, optimizer, learning rate, metrics and training iterations
model = dde.Model(data, net)
model.compile("adam", lr=0.001) # implement: loss_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1])
model.train(epochs=30000)

# Fine tuning with L-BFGS optimizer
model.compile("L-BFGS")
model.train()

# %% Obtain the PINN prediction
y_pred = model.predict(t) * scale  # Denormalize the output

# %% Plot the results
plt.figure(figsize=(12, 6))
labels = ["x1", "x2", "x3"]
colors = ["tab:blue", "tab:orange", "tab:green"]

for i in range(3):
    plt.plot(t_eval, x_ode[:, i], "-", color=colors[i], label=f"{labels[i]} (ODE)")
    plt.plot(t_eval, y_pred[:, i], "--", color=colors[i], label=f"{labels[i]} (PINN)")

plt.xlabel("Time")
plt.ylabel("Protein Concentration")
plt.title("Repressilator Comparison: ODE vs PINN")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()