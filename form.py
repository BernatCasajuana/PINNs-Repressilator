import deepxde as dde
dde.config.set_default_backend("tensorflow")  # <-- ACTIVA EAGER amb TF2

import numpy as np
import tensorflow as tf

beta = 10
n = 3
x0 = np.array([1.0, 1.0, 1.2])
t_test = 5.0

def protein_repressilator_rhs(x, t, beta, n):
    x1, x2, x3 = x
    dx1 = beta / (1 + x3 ** n) - x1
    dx2 = beta / (1 + x1 ** n) - x2
    dx3 = beta / (1 + x2 ** n) - x3
    return [dx1, dx2, dx3]

odeint_dx = protein_repressilator_rhs(x0, t_test, beta, n)
print(f"[ODEINT] t={t_test}, x={x0}, dx={odeint_dx}")

def ode_system_manual(t_tensor, y_tensor):
    y1, y2, y3 = y_tensor[:, 0:1], y_tensor[:, 1:2], y_tensor[:, 2:3]
    dy1 = beta / (1 + y3**n) - y1
    dy2 = beta / (1 + y1**n) - y2
    dy3 = beta / (1 + y2**n) - y3
    return tf.concat([dy1, dy2, dy3], axis=1)

x_tensor = tf.constant([[t_test]], dtype=tf.float32)
y_tensor = tf.constant(x0.reshape(1, 3), dtype=tf.float32)

dx_pinn = ode_system_manual(x_tensor, y_tensor).numpy().flatten()
print(f"[PINN]   t={t_test}, x={x0}, dx={dx_pinn}")