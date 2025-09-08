# run_forward.py
# Prediction of Repressilator dynamics using PINNs and DeepXDE
# Modular version: load dataset, train PINN, save predictions and loss plots

# %% Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
import tensorflow as tf
import os
import argparse

# %% Define ODE system for PINN
def ode_system(x, y, beta, n):
    y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    dy1 = dde.grad.jacobian(y, x, i=0, j=0)
    dy2 = dde.grad.jacobian(y, x, i=1, j=0)
    dy3 = dde.grad.jacobian(y, x, i=2, j=0)

    eq1 = dy1 - (beta / (1 + y3**n) - y1)
    eq2 = dy2 - (beta / (1 + y1**n) - y2)
    eq3 = dy3 - (beta / (1 + y2**n) - y3)

    return [eq1, eq2, eq3]

# %% Main function to run PINN forward
def run_forward(dataset_path, loss_weights=None, outdir_base="results"):
    # Load dataset
    data_npz = np.load(dataset_path)
    t = data_npz["t"]
    x_obs = data_npz["y"]          # data with noise
    x0 = x_obs[0]                   # initial condition
    beta, n = float(data_npz["beta"]), float(data_npz["n"])
    noise_sigma = data_npz["noise"]
    
    # Create output directory
    outdir = os.path.join(outdir_base, f"beta{beta}_n{n}_noise{noise_sigma}")
    os.makedirs(outdir, exist_ok=True)
    
    # Geometry
    geom = dde.geometry.TimeDomain(0, float(t.max()))

    # Initial conditions
    def boundary(_, on_initial):
        return on_initial
    ic1 = dde.icbc.IC(geom, lambda x: x0[0], boundary, component=0)
    ic2 = dde.icbc.IC(geom, lambda x: x0[1], boundary, component=1)
    ic3 = dde.icbc.IC(geom, lambda x: x0[2], boundary, component=2)

    # Observations (subsampling)
    t_obs = t[::10]
    x_obs_sub = x_obs[::10]

    observe_bc = []
    for i in range(3):
        bc = dde.icbc.PointSetBC(t_obs, x_obs_sub[:, i:i+1], component=i)
        observe_bc.append(bc)

    # Define PDE function with parameters
    def ode_func(x, y):
        return ode_system(x, y, beta, n)

    # Data object
    data_pde = dde.data.PDE(
        geom,
        ode_func,
        [ic1, ic2, ic3] + observe_bc, 
        num_domain=5000, 
        num_boundary=2, 
        num_test=1000,
    )
    
    # Neural network
    layer_size = [1] + [100] * 5 + [3]
    net = dde.nn.FNN(layer_size, "sin", "Glorot uniform")
    net.apply_output_transform(lambda x, y: tf.nn.softplus(y))  # positive outputs

    # Model
    model = dde.Model(data_pde, net)
    model.compile("adam", lr=0.001, loss_weights=loss_weights)
    model.train(epochs=5000)

    # Fine-tuning with L-BFGS
    model.compile("L-BFGS")
    model.train()

    # Predictions
    y_pred = model.predict(t)