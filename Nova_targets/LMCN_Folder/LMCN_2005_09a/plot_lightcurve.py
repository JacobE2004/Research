import numpy as np
import matplotlib.pyplot as plt

file_path = "LMCN2005_09a.dat"

data = np.loadtxt(file_path)

time = data[:, 0]
mag = data[:, 1]

plt.figure(figsize=(8, 5))
plt.scatter(time, mag, s=12)
plt.gca().invert_yaxis()
plt.xlabel("Time")
plt.ylabel("Magnitude")
plt.title("LMCN 2005-09a Light Curve")
plt.tight_layout()
plt.show()
