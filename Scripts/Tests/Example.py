# Simple example of solving an ODE using PINNs
# From: https://i-systems.github.io/tutorial/KSNVE/220525/01_PINN.html#3.-Methond-for-Solving-ODE-with-Neural-Networks

# %% Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

# %% Define the neural network
# 1 input layer, 3 hidden layers with 32 nodes each, tanh activation function, and 1 output layer
NN = tf.keras.models.Sequential([
    tf.keras.layers.Input((1,)),
    tf.keras.layers.Dense(units = 32, activation = 'tanh'),
    tf.keras.layers.Dense(units = 32, activation = 'tanh'),
    tf.keras.layers.Dense(units = 32, activation = 'tanh'),
    tf.keras.layers.Dense(units = 1)
])

NN.summary()

# %% Select optimizer
# Adam optimizer with a learning rate of 0.001 (minimizing error between the predicted function and actual ODE solution)
optm = tf.keras.optimizers.Adam(learning_rate = 0.001)

# %% Define ODE System and Loss Function
def ode_system(t, net):
    t = t.reshape(-1,1)
    t = tf.constant(t, dtype = tf.float32)
    t_0 = tf.zeros((1,1))
    one = tf.ones((1,1))

    with tf.GradientTape() as tape:
        tape.watch(t)

        u = net(t)
        u_t = tape.gradient(u, t)

    ode_loss = u_t - tf.math.cos(2*np.pi*t)
    IC_loss = net(t_0) - one

    square_loss = tf.square(ode_loss) + tf.square(IC_loss)
    total_loss = tf.reduce_mean(square_loss)

    return total_loss

# %% Training
# Training data, time points
train_t = (np.array([0., 0.25, 0.475, 0.5, 0.525, 0.6, 0.9, 0.95, 1., 1.05, 1.1, 1.25, 1.4, 1.45, 1.5, 1.55, 1.6, 1.75, 1.95, 2.])).reshape(-1, 1)
train_loss_record = []

# Training loop: 6000 iterations, loss calculation to update the weights of the NN
for itr in range(6000):
    with tf.GradientTape() as tape:
        train_loss = ode_system(train_t, NN)
        train_loss_record.append(train_loss)

        grad_w = tape.gradient(train_loss, NN.trainable_variables)
        optm.apply_gradients(zip(grad_w, NN.trainable_variables))

    if itr % 1000 == 0:
        print(train_loss.numpy())

# %% Plot Loss
# Plotting the resulting loss over iterations
plt.figure(figsize = (10,8))
plt.plot(train_loss_record)
plt.xlabel("Iteration")        
plt.ylabel("Loss")      
plt.title("Loss evolution during training")
plt.grid(True)
plt.show()

# %% Prediction and actual solution
test_t = np.linspace(0, 2, 100).reshape(-1,1)

train_u = np.sin(2*np.pi*train_t)/(2*np.pi) + 1
true_u = np.sin(2*np.pi*test_t)/(2*np.pi) + 1
pred_u = NN.predict(test_t).ravel()

# %% Plot Results
plt.figure(figsize = (10,8))
plt.plot(train_t, train_u, 'ok', label = 'Train')
plt.plot(test_t, true_u, '-k',label = 'True')
plt.plot(test_t, pred_u, '--r', label = 'Prediction')
plt.legend(fontsize = 15)
plt.xlabel('t', fontsize = 15)
plt.ylabel('u', fontsize = 15)
plt.show()