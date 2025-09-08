# generate_data.py

# %% Import necessary libraries
import numpy as np
import scipy.integrate
import os

# %% Define ODE system
def protein_repressilator_rhs(x, t, beta, n):
    x1, x2, x3 = x
    return [
        beta / (1 + x3 ** n) - x1,
        beta / (1 + x1 ** n) - x2,
        beta / (1 + x2 ** n) - x3,
    ]

# %% Add noise to data
def add_noise(y, sigma):
    """Add Gaussian noise with sigma standard deviation."""
    return y + np.random.normal(0, sigma, y.shape)

# %% Generate dataset
def generate_dataset(beta, n, x0, t_max, n_points, noise_sigma=0.0, outdir="datasets"):
    os.makedirs(outdir, exist_ok=True)
# Time vector
    t = np.linspace(0, t_max, n_points)[:, None]
# Solve ODE
    y_clean = scipy.integrate.odeint(protein_repressilator_rhs, x0, t.flatten(), args=(beta, n))
# Add noise if specified
    if noise_sigma > 0:
        y_noisy = add_noise(y_clean, noise_sigma)
    else:
        y_noisy = y_clean
# Generate filename
    fname = f"Repressilator_beta{beta}_n{n}_noise{noise_sigma}.npz"
    fpath = os.path.join(outdir, fname)
# Save dataset as .npz file
    np.savez(fpath, t=t, y=y_noisy, y_clean=y_clean, beta=beta, n=n, noise=noise_sigma)
# Path to saved file
    print(f"Saved dataset: {fpath}")

# %% Main execution block
if __name__ == "__main__":
    # Stable system without noise
    generate_dataset(beta=10, n=1.5, x0=[1, 1, 1.2], t_max=20, n_points=1000, noise_sigma=0.0)

    # Stable system with noise
    generate_dataset(beta=10, n=1.5, x0=[1, 1, 1.2], t_max=20, n_points=1000, noise_sigma=0.05)
    
    # Stable system without noise
    generate_dataset(beta=5.0, n=1.5, x0=[1, 1, 1.2], t_max=20, n_points=1000, noise_sigma=0.0)

    # Stable system with noise
    generate_dataset(beta=5.0, n=1.5, x0=[1, 1, 1.2], t_max=20, n_points=1000, noise_sigma=0.05)

    # Unstable system without noise
    generate_dataset(beta=10, n=3.0, x0=[1, 1, 1.2], t_max=20, n_points=1000, noise_sigma=0.0)

    # Unstable system with noise
    generate_dataset(beta=10, n=3.0, x0=[1, 1, 1.2], t_max=20, n_points=1000, noise_sigma=0.05)
    
    # Unstable system without noise
    generate_dataset(beta=5.0, n=3.0, x0=[1, 1, 1.2], t_max=20, n_points=1000, noise_sigma=0.0)

    # Unstable system with noise
    generate_dataset(beta=5.0, n=3.0, x0=[1, 1, 1.2], t_max=20, n_points=1000, noise_sigma=0.05)