#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 15:19:54 2023

@author: ashleystone
"""

# Use NumPy for mathematical functions and constants
# Use NumPy for mathematical functions and constants
import numpy as np
import pylab as p
from numpy import linspace, array, pi, sin, cos, tan, sqrt, arctan, arccos
from PIL import Image
import matplotlib as mpl
from operator import itemgetter
import pylab
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import os
from matplotlib import rc

filename = 'V1710_Sco.txt'  # change filename here

x = np.loadtxt(filename, delimiter=',', dtype='<f8', usecols=[0])
z = np.loadtxt(filename, delimiter=',', dtype='<f8', usecols=[1])
err = 0.01 * z  # Alternatively, load error values from the file
y = np.loadtxt(filename, delimiter=',', dtype='S15', usecols=[4])
print(y)

t0 = 2459318.13211  # change the date here to t0 in JD

x = x - t0

# Number of observations for each filter (update these numbers as needed)
nV = 322  
nCV = 25  
nVis = 57  
nUF = 0
nUL = 3
nB = 298
nAN = 862

# Create lists for each filter
V = [0] * nV
Vx = [0] * nV
Verr = [0] * nV

CV = [0] * nCV
CVx = [0] * nCV
CVerr = [0] * nCV

Vis = [0] * nVis
Visx = [0] * nVis
Viserr = [0] * nVis

UF = [0] * nUF
UFx = [0] * nUF
UFerr = [0] * nUF

UL = [0] * nUL
ULx = [0] * nUL
ULerr = [0] * nUL

B = [0] * nB
Bx = [0] * nB
Berr = [0] * nB

AN = [0] * nAN
ANx = [0] * nAN
ANerr = [0] * nAN

# Initialize counters for each type of observation
v = j = h = k = m = f = c = 0
a = len(y)

for i in range(a):
    if y[i] == b'V':
        V[v] = z[i]
        Vx[v] = x[i]
        Verr[v] = err[i]
        v += 1           
    elif y[i] == b'CV':
        CV[j] = z[i]
        CVx[j] = x[i]
        CVerr[j] = err[i]
        j += 1
    elif y[i] == b'Vis.':
        Vis[h] = z[i]
        Visx[h] = x[i]
        Viserr[h] = 0
        h += 1
    elif y[i] == b'UF':
        UF[k] = z[i]
        UFx[k] = x[i]
        UFerr[k] = err[i]
        k += 1        
    elif y[i] == b'UL':
        UL[m] = z[i]
        ULx[m] = x[i]
        ULerr[m] = err[i]
        m += 1 
    elif y[i] == b'B':
        B[f] = z[i]
        Bx[f] = x[i]
        Berr[f] = err[i]
        f += 1  
    elif y[i] == b'AN':
        AN[c] = z[i]
        ANx[c] = x[i]
        ANerr[c] = err[i]
        c += 1
     
# Plotting section
fig, ax = plt.subplots(figsize=(11, 6.4))
ax.get_xaxis().get_major_formatter().set_useOffset(False)
plt.gca().invert_yaxis()

ax.errorbar(Visx, Vis, yerr=0, fmt='rd', markersize=5, label="Vis.")
ax.errorbar(CVx, CV, yerr=0, fmt='bo', markersize=5, label=r"$\it{CV}$")
ax.errorbar(ULx, UL, yerr=0, fmt='kv', markersize=7)
ax.errorbar(ANx, AN, yerr=0, fmt='mo', markersize=6, label="ASAS-SN")
ax.errorbar(Vx, V, yerr=0, fmt='g*', markersize=8, label=r"$\it{\;V}$")

# Additional plotting commands (plotting lines, text, error bars, etc.) go here
# For example, drawing reference lines:
Vmax = [8.526, 8.526]  # Example Vmax; change as needed
maxdate = 2459318.13211 - t0  # Date of maximum brightness (adjust as needed)
tmax = [maxdate, maxdate]
t2 = [4.4944, 4.4944]  # t2 values (adjust as needed)
tmaxt2 = [tmax[0] + t2[0], tmax[1] + t2[1]]
Vmaxt2 = [Vmax[0] + 2.0, Vmax[1] + 2.0]
ymax = [0, 25]
xmax = [-10, 300]
ax.plot(tmax, ymax, 'k--', linewidth=1.0)
ax.plot(tmaxt2, ymax, 'k--', linewidth=1.0)
ax.plot(xmax, Vmax, 'k--', linewidth=1.0)
ax.plot(xmax, Vmaxt2, 'k--', linewidth=1.0)

# Set tick intervals and labels
matplotlib.rcParams.update({'font.size': 20})
minor_xticks = np.arange(-50, 600, 5)
minor_yticks = np.arange(0, 19.0, 0.5)
ax.set_xticks(minor_xticks, minor=True)
ax.set_yticks(minor_yticks, minor=True)
ax.set_xlim(-8, 63.5)
ax.set_ylim(18, 8)
plt.tick_params(width=3, length=8)
plt.tick_params(which='minor', width=2, length=4)
ax.set_xlabel('Days since peak magnitude', fontsize=20)
ax.set_ylabel('Magnitude', fontsize=20)
ax.legend(bbox_to_anchor=(0.2, 0.1), loc=1, borderaxespad=0.)
ax.legend(numpoints=1)
plt.text(50, 9.2, 'V1710 Sco')

# Save and close the plot
plt.tight_layout()
plt.savefig('V1710_Sco_LC.pdf')
plt.close()
