# %% Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import deepxde as dde

# %% Define parameters, initial conditions, and time domain
beta = 10
n = 3
x0 = np.array([1, 1, 1.2])
n_points = 1000
t_max = 40
t = np.linspace(0, t_max, n_points)[:, None]

# %% Simulation with ODEINT
def protein_repressilator_rhs(x, t, beta, n):
    x1, x2, x3 = x
    return [
        beta / (1 + x3 ** n) - x1,
        beta / (1 + x1 ** n) - x2,
        beta / (1 + x2 ** n) - x3,
    ]

x_ode = scipy.integrate.odeint(protein_repressilator_rhs, x0, t.flatten(), args=(beta, n))

# Save as .npz file for inverse problem
np.savez("Repressilator.npz", t=t, y=x_ode)

# %% PINN simulation setup
# Geometry of the problem (time domain)
geom = dde.geometry.TimeDomain(0, t_max)

# Define ODE system
def ode_system(x, y):
    y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    dy1 = dde.grad.jacobian(y, x, i=0, j=0)
    dy2 = dde.grad.jacobian(y, x, i=1, j=0)
    dy3 = dde.grad.jacobian(y, x, i=2, j=0)

    eq1 = dy1 - (beta / (1 + y3**n) - y1)
    eq2 = dy2 - (beta / (1 + y1**n) - y2)
    eq3 = dy3 - (beta / (1 + y2**n) - y3)

    return [eq1, eq2, eq3]

# Initial conditions
def boundary(_, on_initial):
    return on_initial
ic1 = dde.icbc.IC(geom, lambda x: 1, boundary, component=0)
ic2 = dde.icbc.IC(geom, lambda x: 1, boundary, component=1)
ic3 = dde.icbc.IC(geom, lambda x: 1.2, boundary, component=2)

# Problem setup
data = dde.data.PDE(geom, ode_system, [ic1, ic2, ic3], num_domain=2000, num_boundary=2, num_test=1000)

# Neural network architecture
layer_size = [1] + [50] * 3 + [3]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

# %% Compile and train
# Define data, optimizer, learning rate, metrics and training iterations
model = dde.Model(data, net)
model.compile("adam", lr=0.001) # implement loss_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1])
model.train(epochs=5000)

# Fine tuning with L-BFGS optimizer
model.compile("L-BFGS")
model.train()

# %% Obtain the PINN prediction
y_pred = model.predict(t)

# %% Plot the results
plt.figure(figsize=(12, 6))
labels = ["x1", "x2", "x3"]
colors = ["tab:blue", "tab:orange", "tab:green"]

for i in range(3):
    plt.plot(t, x_ode[:, i], "-", color=colors[i], label=f"{labels[i]} (ODE)")
    plt.plot(t, y_pred[:, i], "--", color=colors[i], label=f"{labels[i]} (PINN)")

plt.xlabel("Time")
plt.ylabel("Protein Concentration")
plt.title("Repressilator Comparison: ODE vs PINN")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()