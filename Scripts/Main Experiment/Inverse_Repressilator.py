# Parameter estimation in the repressilator model using PINNs and DeepXDE

# %% Import necessary libraries
import os
os.environ["DDE_BACKEND"] = "tensorflow"  # Force TensorFlow backend before importing deepxde

import tensorflow as tf
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import csv

# %% Define parameters (initial suspected values and range), initial conditions, and time domain
C1 = dde.Variable(0.5, min=0.01, max=20) # C1 = beta
C2 = dde.Variable(2.0, min=0.5, max=6) # C2 = n
x0 = np.array([1, 1, 1.2])
t_max = 20

# %% PINN simulation setup
# Geometry of the problem
geom = dde.geometry.TimeDomain(0, t_max)

# Define the ODE system
def ode_system(x, y):
    y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    dy1 = dde.grad.jacobian(y, x, i=0, j=0)
    dy2 = dde.grad.jacobian(y, x, i=1, j=0)
    dy3 = dde.grad.jacobian(y, x, i=2, j=0)

    eq1 = dy1 - (C1 / (1 + y3**C2) - y1)
    eq2 = dy2 - (C1 / (1 + y1**C2) - y2)
    eq3 = dy3 - (C1 / (1 + y2**C2) - y3)

    return [eq1, eq2, eq3]

# Initial conditions
def boundary(_, on_initial):
    return on_initial
ic1 = dde.icbc.IC(geom, lambda x: x0[0], boundary, component=0)
ic2 = dde.icbc.IC(geom, lambda x: x0[1], boundary, component=1)
ic3 = dde.icbc.IC(geom, lambda x: x0[2], boundary, component=2)

# Obtain observed data from odeint solution (experimental data in practice)
# Load the data from the .npz file
data = np.load("/Users/bernatcasajuana/github/PINNs_Repressilator/Datasets/Repressilator.npz")

# Extract time and concentration data
t_full = data["t"]
x_full = data["y"]
t_obs = t_full[::10] # Every 10 time points
x_obs = x_full[::10] # Corresponding concentrations

# Implement observed data as boundary conditions
observe_bc = []
for i in range(3):
    bc = dde.icbc.PointSetBC(t_obs, x_obs[:, i:i+1], component=i)
    observe_bc.append(bc)

# Problem setup, with anchors as extra points where the model will be evaluated
data = dde.data.PDE(geom, ode_system, [ic1, ic2, ic3] + observe_bc, num_domain=5000, num_boundary=2, anchors=t_obs)

# Neural network architecture
layer_size = [1] + [100] * 5 + [3]
activation = "sin"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

# %% Compile and train
# Define data, optimizer, learning rate, training iterations and external trainable variables (C1 and C2)
model = dde.Model(data, net)
model.compile("adam", lr=0.001, external_trainable_variables=[C1, C2]) # implement weight for each loss term if needed: loss_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1])
variable = dde.callbacks.VariableValue([C1, C2], period=100, filename="variables.dat") # Save the values of C1 and C2 during training
model.train(iterations=5000, callbacks=[variable])

# Fine tuning with L-BFGS optimizer
model.compile("L-BFGS", external_trainable_variables=[C1, C2])
model.train()

# %% Obtain the PINN prediction
y_pred = model.predict(t_full)

# %% Obtain the estimated parameters
print(f"Real C1 value = 10.000000")
print(f"Real C2 value = 3.000000")
print(f"Estimated value of C1 = {C1.value():.6f}")
print(f"Estimated value of C2 = {C2.value():.6f}")

# Save in CSV
with open("estimated_parameters.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Parameter", "Estimated Value"])
    writer.writerow(["C1", f"{C1.value():.6f}"])
    writer.writerow(["C2", f"{C2.value():.6f}"])

# %% Plot the training loss
loss_history = model.losshistory
loss_train = np.array(loss_history.loss_train)
epochs = np.arange(len(loss_train))

# Transpose the loss array to separate components
loss_components = loss_train.T

# Name the components for clarity
component_names = [
    "Eq1 (dx1/dt)", "Eq2 (dx2/dt)", "Eq3 (dx3/dt)",
    "IC x1", "IC x2", "IC x3",
    "Obs x1", "Obs x2", "Obs x3"
]

plt.figure(figsize=(10, 6))
for i in range(len(component_names)):
    plt.semilogy(epochs, loss_components[i], label=component_names[i])

plt.xlabel("Iteration")
plt.ylabel("Loss (log scale)")
plt.title("Training Loss over Iterations")
plt.legend()
plt.tight_layout()
plt.show()

# %% Plot the prediction results
plt.figure(figsize=(12, 6))
labels = ["Repressor 1", "Repressor 2", "Repressor 3"]
colors = ["tab:blue", "tab:orange", "tab:green"]

for i in range(3):
    plt.plot(t_full, x_full[:, i], "-", color=colors[i], label=f"{labels[i]} (ODE simulation)")
    plt.plot(t_full, y_pred[:, i], "--", color=colors[i], label=f"{labels[i]} (PINN prediction)")

plt.xlabel("Time")
plt.ylabel("Protein Concentration")
plt.title("Repressilator Dynamics: ODE vs PINN")
plt.legend()
plt.tight_layout()
plt.show()

# %% Plot the evolution of the estimated parameters
variables = np.loadtxt("variables.dat")
plt.plot(variables[:, 0], variables[:, 1], label="C1")
plt.plot(variables[:, 0], variables[:, 2], label="C2")
plt.xlabel("Iteration")
plt.ylabel("Estimated Parameter Value")
plt.title("Evolution of Estimated Parameters")
plt.legend()
plt.show()