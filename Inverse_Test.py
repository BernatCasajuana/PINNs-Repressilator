# Lorentz Equations with PINN using DeepXDE. True parameters: C1=10, C2=15, C3=8/3.

# Import necessary libraries
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the Lorenz system parameters (initial suspected values = 1)
C1 = dde.Variable(1.0)
C2 = dde.Variable(1.0)
C3 = dde.Variable(1.0)

# Define the geometry of the problem (time domain)
geom = dde.geometry.TimeDomain(0, 3)

# Define the Lorenz system of ODEs. Solution (y) is a vector of three components (y1, y2, y3) representing the Lorenz attractor (coordinates x, y, z).
def Lorenz_system(x, y):
    y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:]
    dy1_x = dde.grad.jacobian(y, x, i=0)
    dy2_x = dde.grad.jacobian(y, x, i=1)
    dy3_x = dde.grad.jacobian(y, x, i=2)
    return [
        dy1_x - C1 * (y2 - y1),
        dy2_x - y1 * (C2 - y3) + y2,
        dy3_x - y1 * y2 + C3 * y3,
    ]

# Define initial conditions within the geometry, function and boundary
def boundary(_, on_initial):
    return on_initial

ic1 = dde.icbc.IC(geom, lambda X: -8, boundary, component=0)
ic2 = dde.icbc.IC(geom, lambda X: 7, boundary, component=1)
ic3 = dde.icbc.IC(geom, lambda X: 27, boundary, component=2)

# Assign the data from Lorenz.npz to the corresponding t (time), x, y and z (coordinates) values for training.
def gen_traindata():
    data = np.load("Lorenz.npz")
    return data["t"], data["y"]

# Organize and assign the training data to the corresponding variables.
observe_t, ob_y = gen_traindata()
observe_y0 = dde.icbc.PointSetBC(observe_t, ob_y[:, 0:1], component=0)
observe_y1 = dde.icbc.PointSetBC(observe_t, ob_y[:, 1:2], component=1)
observe_y2 = dde.icbc.PointSetBC(observe_t, ob_y[:, 2:3], component=2)

# Define the PDE data for the Lorenz system, including the initial conditions and observation points. Anchors are extra points where the model will be evaluated, in this case the time points of the training data.
data = dde.data.PDE(
    geom,
    Lorenz_system,
    [ic1, ic2, ic3, observe_y0, observe_y1, observe_y2],
    num_domain=400,
    num_boundary=2,
    anchors=observe_t,
)

# Define the neural network architecture
net = dde.nn.FNN([1] + [40] * 3 + [3], "tanh", "Glorot uniform")

# Compile the model with the data, optimizer (adam), learning rate, and external trainable variables (C1, C2, C3). The external trainable variables are the Lorenz system parameters.
model = dde.Model(data, net)
model.compile("adam", lr=0.001, external_trainable_variables=[C1, C2, C3])
variable = dde.callbacks.VariableValue([C1, C2, C3], period=600, filename="variables.dat")

# Train the model with the specified number of iterations and callbacks. The variable callback saves the values of C1, C2, and C3 every 600 iterations.
losshistory, train_state = model.train(iterations=60000, callbacks=[variable])

# Predict the numeric values of the parameters after training
import tensorflow as tf

print("Predicted parameters:")
print(f"C1 = {tf.keras.backend.get_value(C1):.6f}")
print(f"C2 = {tf.keras.backend.get_value(C2):.6f}")
print(f"C3 = {tf.keras.backend.get_value(C3):.6f}")