# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 14:16:26 2024

@author: Jmell
"""


import numpy as np
import pylab as p
from numpy import linspace
from numpy import array

from PIL import Image
import matplotlib as mpl
from operator import itemgetter
import pylab
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

import os
from matplotlib import rc

# Use the full file path
file_path = r"C:\Users\Jmell\Dropbox\Research File - Copy\Nova_targets\SMCN_2005_08a\SMCN_2005_08a.dat"  # Raw string to handle backslashes
readfile = np.loadtxt(file_path)

# Extract columns
time = readfile[:, 0]
Mag = readfile[:, 1]
Magerr = readfile[:, 2]

# Exclude points where magnitude is 99.999
valid_indices = Mag != 99.999
time = time[valid_indices]
Mag = Mag[valid_indices]
Magerr = Magerr[valid_indices]

# Adjust time relative to peak
time = time - 2453300.61720   # Change this depending on tpeak

# Plot the data
plt.figure(figsize=(16, 9))
ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_useOffset(False)
plt.gca().invert_yaxis()

ax.errorbar(time, Mag, yerr=Magerr, fmt='go-', markersize=8, label="$\it{\;I}$-band")

############################################
matplotlib.rcParams.update({'font.size': 20})
Ipeakx = [-20, 3000]  # Changes where the dotted line starts and ends for x-axis
Ipeaky = [12.783, 12.783]  # Peak of novae
Ipeakyplus2 = [12.783 + 2, 12.783 + 2]  # When the nova drops below 2 mag
plt.plot(Ipeakx, Ipeaky, 'k--')
plt.plot(Ipeakx, Ipeakyplus2, 'k--')

tpeakx = [354.034, 354.034]  # Peak of nova days
tpeaky = [40, 5]  # Changes the where the dotted line starts and ends for y-axis
t2x = [579.04, 579.04]  # Days since peak where it drops below 2 mag last time
plt.plot(tpeakx, tpeaky, 'k--')
plt.plot(t2x, tpeaky, 'k--')

major_yticks = np.arange(0, 30, 1)
minor_yticks = np.arange(0, 30, 0.5)

ax.set_yticks(major_yticks)
ax.set_yticks(minor_yticks, minor=True)

major_xticks = np.arange(-20, 2800, 100)  # (Changes where it starts, changes where it begins, distance of the ticks)
minor_xticks = np.arange(-20, 2800, 50)

ax.set_xticks(major_xticks)
ax.set_xticks(minor_xticks, minor=True)
plt.ylim(30, 11.7)  # Changes what the graph will show for the y-axis
plt.xlim(-20, 1550)  # Changes what the graph will show based on JD starting point
plt.xlabel('Day since peak brightness')
plt.ylabel('Optical Brightness')
plt.tick_params(width=3, length=8)
plt.title('SMCN_2004_06a', fontsize=20)
plt.tick_params(which='minor', width=2, length=4)
plt.rcParams.update({'font.size': 20})
plt.tight_layout()
plt.legend(bbox_to_anchor=(0.85, 1.0), loc=2, borderaxespad=0.)
plt.legend(numpoints=1)
plt.savefig('SMCN_2004_06a.pdf')
plt.show()
