# run_all_forward.py

# Import necessary libraries
import os

# Import the modular "run_forward" function from run_forward.py
from run_forward import run_forward

# Define dataset folder and results folder
dataset_folder = "datasets"
outdir_base = "results"

# Run PINN for each file in the datasets folder
for file in os.listdir(dataset_folder):
    if file.endswith(".npz"):
        dataset_path = os.path.join(dataset_folder, file)
        print(f"\n=== Running PINN for {dataset_path} ===")
        run_forward(dataset_path, loss_weights=None, outdir_base=outdir_base)