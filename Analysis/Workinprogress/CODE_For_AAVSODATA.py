# --- Standard Libraries ---
import scipy                         # Scientific computing library (not used directly in this script)
import numpy as np                   # Core numerical computing library
import pylab as p                    # Legacy MATLAB-like plotting interface (not used directly)
from numpy import linspace, array    # Import specific NumPy functions (also not needed as np covers them)

# --- Plotting Libraries ---
from PIL import Image                # Python Imaging Library (not used here)
import matplotlib as mpl             # Main matplotlib module (used for config)
from operator import itemgetter      # Not used
import pylab                         # Redundant import
from pylab import *                  # Imports all pylab functions — discouraged practice (already imported)
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting (not used)
import matplotlib.pyplot as plt      # Recommended way to use matplotlib
import random                        # Random module (not used)

import os                            # OS-level functions (not used)
from matplotlib import rc            # Used for matplotlib configuration

# --- Load photometric data ---
file_path = r"C:\Users\Jmell\Dropbox\Research File\Nova_targets\SMCN_FOLDER\SMCN_2016_10a\SMCN_2016_10a_ASASSN_light-curves.csv"

# Load time column (x), magnitude (z), and filter label (y)
x = np.genfromtxt(file_path, delimiter=',', dtype='<f8', usecols=0, skip_header=1)
z = np.genfromtxt(file_path, delimiter=',', dtype='<f8', usecols=1, skip_header=1)
y = np.genfromtxt(file_path, delimiter=',', dtype='S15', usecols=4, skip_header=1)

# Remove NaN entries for time and mag
mask = ~np.isnan(x) & ~np.isnan(z)
x = x[mask]
z = z[mask]
y = y[mask]

# Estimate photometric errors as 1% of flux (arbitrary assumption)
err = 0.01 * z

# Set time zero point
x = x - 2458094.3198

# --- Allocate space for each photometric band ---
nV = 7255
nCV = 32432
nVis = 5797
nB = 2215
nR = 1405
nI = 4565

# Allocate arrays for each band and time/err
V = [0 for _ in range(nV)];     Vx = [0 for _ in range(nV)];     Verr = [0 for _ in range(nV)]
CV = [0 for _ in range(nCV)];   CVx = [0 for _ in range(nCV)];   CVerr = [0 for _ in range(nCV)]
Vis = [0 for _ in range(nVis)]; Visx = [0 for _ in range(nVis)]; Viserr = [0 for _ in range(nVis)]
B = [0 for _ in range(nB)];     Bx = [0 for _ in range(nB)];     Berr = [0 for _ in range(nB)]
I = [0 for _ in range(nI)];     Ix = [0 for _ in range(nI)];     Ierr = [0 for _ in range(nI)]
R = [0 for _ in range(nR)];     Rx = [0 for _ in range(nR)];     Rerr = [0 for _ in range(nR)]

# Counters for each band
i2 = b = v = r = j = h = k = 0
a = len(y)

# --- Sort observations by photometric filter ---
for i in range(a):
    if y[i] == b'V':
        V[v] = z[i]; Vx[v] = x[i]; Verr[v] = err[i]; v += 1
    elif y[i] == b'CV':
        CV[j] = z[i]; CVx[j] = x[i]; CVerr[j] = err[i]; j += 1
    elif y[i] == b'Vis.':
        Vis[h] = z[i]; Visx[h] = x[i]; Viserr[h] = err[i]; h += 1
    elif y[i] == b'I':
        I[i2] = z[i]; Ix[i2] = x[i]; Ierr[i2] = err[i]; i2 += 1
    elif y[i] == b'R':
        R[r] = z[i]; Rx[r] = x[i]; Rerr[r] = err[i]; r += 1
    elif y[i] == b'B':
        B[b] = z[i]; Bx[b] = x[i]; Berr[b] = err[i]; b += 1

# --- Begin Plotting ---
fig, ax = plt.subplots(figsize=(15, 9))        # Create figure and axis
ax = plt.gca()                                 # Get current axis
ax.get_xaxis().get_major_formatter().set_useOffset(False)  # Avoid scientific notation for x-axis
plt.gca().invert_yaxis()                       # Magnitude scale: brighter = lower number

# --- Plot light curves with error bars ---
ax.errorbar(Ix, I, yerr=Ierr, fmt='ko', markersize=7, label=r"$\it{\;I}$", rasterized=True)
ax.errorbar(Rx, R, yerr=Rerr, fmt='mp', markersize=7, label=r"$\it{\;R}$", rasterized=True)
ax.errorbar(Vx, V, yerr=Verr, fmt='g*', markersize=8, label=r"$\it{\;V}$", rasterized=True)
ax.errorbar(Bx, B, yerr=Berr, fmt='bo', markersize=7, label=r"$\it{\;B}$", rasterized=True)

# (Optional) Add CV and Vis filters if desired:
# ax.errorbar(CVx, CV, ...)
# ax.errorbar(Visx, Vis, ...)

# --- Font size config ---
matplotlib.rcParams.update({'font.size': 20})

# --- Mark important dates (e.g., tgamma, tcharas) ---
tgamma = 2459355.14071 - 2459291.93237     # Gamma-ray detection date offset from t0
barx = [tgamma, tgamma]
bary = [15, 0]

# Define additional event lines (like onset of features or observations)
tchara1 = 2459345.14071 - 2459291.93237
tchara2 = 2459347.14071 - 2459291.93237
tchara3 = 2459354.14071 - 2459291.93237
tchara4 = 2459359.14071 - 2459291.93237

print(tchara1)
print(tchara2)
print(tchara4)

# Plot vertical dashed lines at each event
chara1 = [tchara1, tchara1]
chara2 = [tchara2, tchara2]
chara3 = [tchara3, tchara3]
chara4 = [tchara4, tchara4]

plt.plot(chara1, bary, 'r--', linewidth=2.5)
plt.plot(chara2, bary, 'r--', linewidth=2.5)
# plt.plot(chara3, bary, 'r--', linewidth=2.0)  # commented out
plt.plot(chara4, bary, 'r--', linewidth=2.5)

# --- Axis configuration ---
minor_xticks = np.arange(-200, 600, 5)         # Define minor ticks every 5 days
ax.set_ylim(10.0, 4)                           # Invert y-axis range (mag scale)
ax.set_xticks(minor_xticks, minor=True)       # Enable minor ticks
ax.set_xlim(-5, 100)                           # Set x-axis range

# --- Labels and Legend ---
ax.legend(bbox_to_anchor=(0.99, 0.98), borderaxespad=0., loc='upper right')
plt.grid(False)                                # Disable grid
plt.tick_params(width=3, length=8)             # Major tick size
plt.tick_params(which='minor', width=2, length=4)  # Minor tick size
plt.xlabel('Day since $t_0$', fontsize=20)
plt.ylabel('Magnitude', fontsize=20)

# --- Final layout ---
plt.tight_layout()
plt.show()                                     # Display the plot
