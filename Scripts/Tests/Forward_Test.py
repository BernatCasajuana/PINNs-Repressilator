# Predictive model for a simple ODE system using PINNs and DeepXDE
# From: https://deepxde.readthedocs.io

# %% Import necessary libraries
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os

# %% Define the model
# Geometry of the problem (time domain)
geom = dde.geometry.TimeDomain(0, 30)

# Define ODE system
def ode_system(x, y):
    y1, y2 = y[:, 0:1], y[:, 1:]
    dy1_x = dde.grad.jacobian(y, x, i=0)
    dy2_x = dde.grad.jacobian(y, x, i=1)
    return [dy1_x - y2, dy2_x + y1]

# Initial conditions
def boundary(x, _):
    return np.isclose(x[0], 0)

ic1 = dde.icbc.IC(geom, lambda x: 0, boundary, component=0)
ic2 = dde.icbc.IC(geom, lambda x: 1, boundary, component=1)

# Solution function (if known, for testing purposes)
# def func(x):
    # return np.hstack((np.sin(x), np.cos(x)))

# Set up the problem
data = dde.data.PDE(geom, ode_system, [ic1, ic2], num_domain=2000, num_boundary=2, num_test=1000)

# Neural network architecture
layer_size = [1] + [100] * 4 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

# %% Compile and train the model
# Define data, optimizer, learning rate, metrics and training iterations
model = dde.Model(data, net)
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(iterations=20000)

# Fine tuning with L-BFGS optimizer
model.compile("L-BFGS")
losshistory, train_state = model.train()

# %% Plot the loss results
# Save and plot the best trained result and loss history
def saveplot_manual(losshistory, train_state, model, data, issave=True, isplot=True):
    folder = "Your path to the folder where you want to save the results"
    os.makedirs(folder, exist_ok=True)
    
    loss_steps = np.array(losshistory.steps).flatten()
    loss_train = np.array(losshistory.loss_train).flatten()
    min_len = min(len(loss_steps), len(loss_train))
    loss_steps = loss_steps[:min_len]
    loss_train = loss_train[:min_len]
    np.savetxt(f"{folder}/loss.dat", np.vstack((loss_steps, loss_train)).T)
    print(f"Saved loss history to {folder}/loss.dat")

    # Save training data
    if issave:
        if train_state.y_train is not None:
            np.savetxt(f"{folder}/train.dat", np.hstack((train_state.X_train, train_state.y_train[:, None])))
            print(f"Saved training data to {folder}/train.dat")
        else:
            print("train_state.y_train is None, skipping saving training data.")

    # Save test data and predictions
    if issave:
        X_test = train_state.X_test
        np.savetxt(f"{folder}/X_test.dat", X_test)
        print(f"Saved test inputs to {folder}/X_test.dat")
        y_pred = model.predict(X_test)
        np.savetxt(f"{folder}/y_pred.dat", y_pred)
        print(f"Saved test predictions to {folder}/y_pred.dat")

    # Plot loss
    if isplot:
        plt.figure(figsize=(8, 6))
        plt.semilogy(losshistory.steps, losshistory.loss_train, label="Training Loss")
        if losshistory.loss_test is not None:
            plt.semilogy(losshistory.steps, losshistory.loss_test, label="Testing Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.title("Loss History")
        plt.show()

saveplot_manual(losshistory, train_state, model, data, issave=True, isplot=False)

# %% Plot the prediction results
# Obtain the prediction in [0, 30] domain
X_full = np.linspace(0, 30, 1000)[:, None]
Y_full = model.predict(X_full)

# Set up the plot
plt.figure(figsize=(10, 6))

# [0, 10] as "real"
mask_train = X_full[:, 0] <= 10
plt.plot(X_full[mask_train], Y_full[mask_train, 0], color="C0", label="y1 (t ≤ 10)")
plt.plot(X_full[mask_train], Y_full[mask_train, 1], color="C1", label="y2 (t ≤ 10)")

# (10, 30] as "extrapolation"
mask_extrap = X_full[:, 0] > 10
plt.plot(X_full[mask_extrap], Y_full[mask_extrap, 0], "--", color="darkblue", label="y1 (t > 10)")
plt.plot(X_full[mask_extrap], Y_full[mask_extrap, 1], "--", color="orangered", label="y2 (t > 10)")

plt.xlabel("Time")
plt.ylabel("Output")
plt.legend()
plt.grid()
plt.title("PINN prediction over full domain with extrapolation after t=10")
plt.show()