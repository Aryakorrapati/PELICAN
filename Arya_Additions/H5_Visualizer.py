import h5py
import numpy as np
import matplotlib.pyplot as plt

file_path = "C:/Users/aryak/OneDrive/Desktop/PELICAN/data/sample_data/run12/train.h5"

with h5py.File(file_path, "r") as f:
    Nobj = f["Nobj"][:]
    Pmu = f["Pmu"][:] 
    is_signal = f["is_signal"][:] 
    truth_Pmu = f["truth_Pmu"][:] 

plt.figure(figsize=(8, 5))
plt.hist(Nobj, bins=20, alpha=0.7, color='b', edgecolor='black')
plt.xlabel("Number of Particles per Event")
plt.ylabel("Count")
plt.title("Distribution of Particles per Event")
plt.grid()
plt.show()

signal_mask = is_signal == 1
background_mask = is_signal == 0

plt.figure(figsize=(8, 5))
plt.hist(Pmu[signal_mask, :, 0].flatten(), bins=50, alpha=0.6, color='r', label="Signal", edgecolor='black')
plt.hist(Pmu[background_mask, :, 0].flatten(), bins=50, alpha=0.6, color='b', label="Background", edgecolor='black')
plt.xlabel("Energy (E) of Particles")
plt.ylabel("Count")
plt.title("Energy Distribution: Signal vs Background")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(truth_Pmu[:, 0], Pmu[:, 0, 0], alpha=0.5, color='g', edgecolor='black')
plt.xlabel("True Energy (E)")
plt.ylabel("Predicted Energy (E)")
plt.title("True vs Predicted Energy")
plt.grid()
plt.show()
