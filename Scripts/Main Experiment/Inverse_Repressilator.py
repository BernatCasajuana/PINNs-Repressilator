# Import necessary libraries
import os
os.environ["DDE_BACKEND"] = "tensorflow"  # Force TensorFlow backend before importing deepxde
import tensorflow as tf
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# Define the system parameters (initial suspected values)
C1 = dde.Variable(1.0)
C2 = dde.Variable(1.0)

# Define the geometry of the problem (time domain)
geom = dde.geometry.TimeDomain(0, 40)

# Define the ODE system as a function for DeepXDE (with TensorFlow tensors)
def ode_system(x, y):
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    dy1 = dde.grad.jacobian(y, x, i=0, j=0)
    dy2 = dde.grad.jacobian(y, x, i=1, j=0)
    dy3 = dde.grad.jacobian(y, x, i=2, j=0)

    eq1 = dy1 - (C1 / (1 + y3**C2) - y1)
    eq2 = dy2 - (C1 / (1 + y1**C2) - y2)
    eq3 = dy3 - (C1 / (1 + y2**C2) - y3)

    return [eq1, eq2, eq3]

# Define initial conditions within the geometry, function and boundary
def boundary(_, on_initial):
    return on_initial
ic1 = dde.icbc.IC(geom, lambda x: 1, boundary, component=0)
ic2 = dde.icbc.IC(geom, lambda x: 1, boundary, component=1)
ic3 = dde.icbc.IC(geom, lambda x: 1.2, boundary, component=2)

# Load observed data from Repressilator.npz
def gen_traindata():
    data = np.load("/Users/bernatcasajuana/github/PINNs_Repressilator/Datasets/Repressilator.npz")
    return data["t"], data["y"]

observe_t, observe_y = gen_traindata()

# Define PointSetBCs using the observed data
observe_y0 = dde.icbc.PointSetBC(observe_t, observe_y[:, 0:1], component=0)
observe_y1 = dde.icbc.PointSetBC(observe_t, observe_y[:, 1:2], component=1)
observe_y2 = dde.icbc.PointSetBC(observe_t, observe_y[:, 2:3], component=2)

# Define the PDE data for the system, including the initial conditions and observation points. Anchors are extra points where the model will be evaluated, in this case the time points of the training data.
data = dde.data.PDE(
    geom,
    ode_system,
    [ic1, ic2, ic3, observe_y0, observe_y1, observe_y2],
    num_domain=400,
    num_boundary=2,
    anchors=observe_t,
)

# Define the neural network architecture
net = dde.nn.FNN([1] + [40] * 3 + [3], "tanh", "Glorot uniform")

# Compile the model with the data, optimizer (adam), learning rate, and external trainable variables (C1 and C2). The external trainable variables are the parameters.
model = dde.Model(data, net)
model.compile("adam", lr=0.0001, external_trainable_variables=[C1, C2])
variable = dde.callbacks.VariableValue([C1, C2], period=600, filename="variables.dat")
model.compile("L-BFGS", external_trainable_variables=[C1, C2])
model.train()

# Train the model with the specified number of iterations and callbacks. The variable callback saves the values of C1, C2, and C3 every 600 iterations.
losshistory, train_state = model.train(iterations=60000, callbacks=[variable])

# Get the predicted values of the parameters
print(f"C1 = {C1.value():.6f}")
print(f"C2 = {C2.value():.6f}")