import numpy as np
import matplotlib.pyplot as plt

r_cm = np.array([1,2,3,4], dtype=float)
eps_mV = np.array([14.78,21.37,30.09,32.23], dtype=float)
sig_mV = np.array([5.9786286,3.9553199,8.0086689,7.5352579], dtype=float)

# Best-fit (weighted) model from your data:
k = 1.159477836232451   # mV / cm^2
eps0 = 15.970566521774314  # mV

r_fine = np.linspace(r_cm.min(), r_cm.max(), 200)
eps_fit = eps0 + k * r_fine**2

plt.errorbar(r_cm, eps_mV, yerr=sig_mV, fmt='o', capsize=4, label='data')
plt.plot(r_fine, eps_fit, label=r'fit: $\varepsilon=16.0 + 1.16\,r^2$ (mV, r in cm)')
plt.xlabel('Probe radius r (cm)')
plt.ylabel('EMF (mV)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
