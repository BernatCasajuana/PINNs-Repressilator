# Parameter estimation of the Lorenz ODE system using PINNs and DeepXDE
# From: https://deepxde.readthedocs.io

# %% Import necessary libraries
import os
os.environ["DDE_BACKEND"] = "tensorflow"  # Force TensorFlow backend before importing deepxde
import tensorflow as tf
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# Define the Lorenz system parameters (initial suspected values = 1)
C1 = dde.Variable(1.0)
C2 = dde.Variable(1.0)
C3 = dde.Variable(1.0)

# Define the geometry of the problem (time domain)
geom = dde.geometry.TimeDomain(0, 3)

# Define the Lorenz system ODEs as a function for DeepXDE (with TensorFlow tensors)
def Lorenz_system(x, y):
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:3]
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
    data = np.load("/Users/bernatcasajuana/github/PINNs_Repressilator/Datasets/Lorenz.npz")
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

# Get the predicted values of the parameters
print(f"C1 = {C1.value():.6f}")
print(f"C2 = {C2.value():.6f}")
print(f"C3 = {C3.value():.6f}")

# Obtained:
# C1 = 10.005228
# C2 = 14.999722
# C3 = 2.667669