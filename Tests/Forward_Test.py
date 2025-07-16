# Import necessary libraries
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the geometry of the problem (time domain), where the NN will learn the ODE system (not necessarily with data points)
geom = dde.geometry.TimeDomain(0, 30)

# Define ODE system and parameters
def ode_system(x, y):
    y1, y2 = y[:, 0:1], y[:, 1:]
    dy1_x = dde.grad.jacobian(y, x, i=0)
    dy2_x = dde.grad.jacobian(y, x, i=1)
    return [dy1_x - y2, dy2_x + y1]

# Define initial conditions within the geometry, function and boundary (apply initial conditions only at t=0)
def boundary(x, _):
    return np.isclose(x[0], 0)

ic1 = dde.icbc.IC(geom, lambda x: 0, boundary, component=0)
ic2 = dde.icbc.IC(geom, lambda x: 1, boundary, component=1)

# Define the solution function
# def func(x):
    # return np.hstack((np.sin(x), np.cos(x)))

# Define the ODE problem (geometry, ODE system, initial conditions, number of training points, and number of test points)
data = dde.data.PDE(geom, ode_system, [ic1, ic2], num_domain=2000, num_boundary=2, num_test=1000)

# Define the neural network architecture
layer_size = [1] + [100] * 4 + [2] # Number of layers and neurons (1 input, 3 hidden layers with 50 neurons each, and 2 outputs, y1 and y2)
activation = "tanh" # Activation function for the hidden layers (ideally for oscillatory functions)
initializer = "Glorot uniform" # Initializer for the weights (Glorot uniform is a common choice)
net = dde.nn.FNN(layer_size, activation, initializer) # Define NN model

# Compile the model with the data, optimizer (adam), learning rate, and metrics. Train with iterations and store the loss history and training state. Fine tune with L-BFGS.
model = dde.Model(data, net)
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(iterations=20000)

model.compile("L-BFGS")
losshistory, train_state = model.train()

# Save and plot the best trained result and loss history
def saveplot_manual(losshistory, train_state, model, data, issave=True, isplot=True):
    folder = "/Users/bernatcasajuana/Documents/Pràctiques CBBL/PCII PINN/Data"
    os.makedirs(folder, exist_ok=True)
    
    loss_steps = np.array(losshistory.steps).flatten()
    loss_train = np.array(losshistory.loss_train).flatten()
    min_len = min(len(loss_steps), len(loss_train))
    loss_steps = loss_steps[:min_len]
    loss_train = loss_train[:min_len]
    np.savetxt(f"{folder}/loss.dat", np.vstack((loss_steps, loss_train)).T)
    print(f"Saved loss history to {folder}/loss.dat")

    # Guarda training data
    if issave:
        if train_state.y_train is not None:
            np.savetxt(f"{folder}/train.dat", np.hstack((train_state.X_train, train_state.y_train[:, None])))
            print(f"Saved training data to {folder}/train.dat")
        else:
            print("train_state.y_train is None, skipping saving training data.")

    # Guarda test data i prediccions (cal calcular prediccions manualment)
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

# I crida la funció passant el model i les dades:
saveplot_manual(losshistory, train_state, model, data, issave=True, isplot=False)

# Predicció en tot el rang [0, 30]
X_full = np.linspace(0, 30, 1000)[:, None]
Y_full = model.predict(X_full)

# Plot the predictions over the full domain with extrapolation
plt.figure(figsize=(10, 6))

# Zona [0, 10] com "real"
mask_train = X_full[:, 0] <= 10
plt.plot(X_full[mask_train], Y_full[mask_train, 0], color="C0", label="y1 (t ≤ 10)")
plt.plot(X_full[mask_train], Y_full[mask_train, 1], color="C1", label="y2 (t ≤ 10)")

# Zona (10, 30] com extrapolació
mask_extrap = X_full[:, 0] > 10
plt.plot(X_full[mask_extrap], Y_full[mask_extrap, 0], "--", color="darkblue", label="y1 (t > 10)")
plt.plot(X_full[mask_extrap], Y_full[mask_extrap, 1], "--", color="orangered", label="y2 (t > 10)")

plt.xlabel("Time")
plt.ylabel("Output")
plt.legend()
plt.grid()
plt.title("PINN prediction over full domain with extrapolation after t=10")
plt.show()