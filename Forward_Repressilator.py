# Import necessary libraries
import deepxde as dde
import numpy as np

# Define the geometry of the problem (time domain)
geom = dde.geometry.TimeDomain(0, 40)

# Define ODE system
def ode_system(x, y):
    y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:3] # y1, y2, y3 are the three protein concentrations
    dy1_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy2_x = dde.grad.jacobian(y, x, i=1, j=0)
    dy3_x = dde.grad.jacobian(y, x, i=2, j=0)
    
    # Parameters of the model
    beta = 10
    n = 5

    # Repressilator equations with pairs (1,3), (2,1), (3,2)
    eq1 = dy1_x - (beta / (1 + y3**n) - y1)
    eq2 = dy2_x - (beta / (1 + y1**n) - y2)
    eq3 = dy3_x - (beta / (1 + y2**n) - y3)
    
    return [eq1, eq2, eq3]

# Define initial conditions within the geometry, function and boundary (apply initial conditions only at t=0)
def boundary(_, on_initial):
    return on_initial
ic1 = dde.icbc.IC(geom, lambda x: 1, boundary, component=0)
ic2 = dde.icbc.IC(geom, lambda x: 1, boundary, component=1)
ic3 = dde.icbc.IC(geom, lambda x: 1.2, boundary, component=2)

# Define the solution function (if we have a known solution, we can use it to compare)
# def func(x):
    # return np.hstack((np.sin(x), np.cos(x)))

# Define the ODE problem (geometry, ODE system, initial conditions, number of training points, and number of test points)
data = dde.data.PDE(geom, ode_system, [ic1, ic2, ic3], num_domain=2000, num_boundary=2, num_test=1000)

# Define the neural network architecture
layer_size = [1] + [50] * 3 + [3] # Number of layers and neurons (1 input, 3 hidden layers with 50 neurons each, and 3 outputs, y1, y2 and y3)
activation = "tanh" # Activation function for the hidden layers (ideally for oscillatory functions)
initializer = "Glorot uniform" # Initializer for the weights (Glorot uniform is a common choice)
net = dde.nn.FNN(layer_size, activation, initializer) # Define NN model

# Compile the model with the data, optimizer (adam), learning rate, and metric (if solution function is known). Define iterations for training.

model = dde.Model(data, net)
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(iterations=50000)

# Fine-tune with L-BFGS
model.compile("L-BFGS")
losshistory, train_state = model.train()

# Save and plot the best trained result and loss history
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

import matplotlib.pyplot as plt

X = np.linspace(0, 40, 1000)[:, None]
Y_pred = model.predict(X)

# Obtaint training data for plotting
train_x = data.train_x
train_y_pred = model.predict(train_x)

plt.figure(figsize=(10, 5))
plt.plot(X, Y_pred[:, 0], label="x1")
plt.plot(X, Y_pred[:, 1], label="x2")
plt.plot(X, Y_pred[:, 2], label="x3")

plt.scatter(train_x, train_y_pred[:, 0], s=10, color="darkblue", marker='o', label="x1 train pts")
plt.scatter(train_x, train_y_pred[:, 1], s=10, color="darkorange", marker='s', label="x2 train pts")
plt.scatter(train_x, train_y_pred[:, 2], s=10, color="darkgreen", marker='^', label="x3 train pts")

plt.xlabel("Time")
plt.ylabel("Protein Concentration")
plt.title("Repressilator dynamics learned by PINN")
plt.legend()
plt.grid()
plt.show()