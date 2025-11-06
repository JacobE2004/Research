import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# ======================================================
# User-defined inputs
# ======================================================
file_path = r"C:\Users\Jmell\Dropbox\Research File\Nova_targets\LMCN_Folder\LMCN_2005_11a\LMCN_2005_11a.dat"
t_peak_JD = 2453708.72607  # JD of your observed brightest point
m_peak = 11.386            # Magnitude at t_peak_JD

# ======================================================
# Load Data (JD, Mag, MagErr)
# ======================================================
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Data file not found: {file_path}")

data = np.loadtxt(file_path)
time = data[:, 0]     # JD
mag = data[:, 1]      # Magnitude
mag_err = data[:, 2]  # Magnitude error

# ======================================================
# Find index of user-defined t_peak in the data
# ======================================================
idx_peak = np.argmin(np.abs(time - t_peak_JD))

# Confirm m_peak matches data
m_peak_data = mag[idx_peak]
if abs(m_peak_data - m_peak) > 0.1:
    print(f"Warning: provided m_peak ({m_peak}) differs from data ({m_peak_data:.3f}) at closest JD.")

# ======================================================
# Calculate t_peak uncertainty (half of the smaller adjacent gap)
# ======================================================
if 0 < idx_peak < len(time) - 1:
    gap_before = time[idx_peak] - time[idx_peak - 1]
    gap_after = time[idx_peak + 1] - time[idx_peak]
    sigma_tpeak = min(gap_before, gap_after) / 2
elif idx_peak == 0:
    sigma_tpeak = (time[1] - time[0]) / 2
else:
    sigma_tpeak = (time[-1] - time[-2]) / 2

# Shift time axis so t_peak = 0
time_shifted = time - t_peak_JD

# ======================================================
# Compute t2 and its uncertainty
# ======================================================
m_t2 = m_peak + 2.0
# Only consider times after t_peak for t2
mask_after_peak = time > t_peak_JD
idx_after = np.where(mask_after_peak & (mag > m_t2))[0]

if len(idx_after) == 0:
    raise ValueError("t2 (2-mag decline) not reached in the dataset after t_peak.")
i2 = idx_after[0]
t1, m1 = time[i2 - 1], mag[i2 - 1]
t2, m2 = time[i2], mag[i2]

# Linear interpolation for exact JD of t2
frac = (m_t2 - m1) / (m2 - m1)
t2_exact = t1 + frac * (t2 - t1)
t2_central = t2_exact - t_peak_JD

# Uncertainty from bracketing interval
sigma_interval_t2 = (t2 - t1) / 2
# Total t2 uncertainty via quadrature
sigma_t2 = np.sqrt(sigma_interval_t2**2 + sigma_tpeak**2)

# ======================================================
# Plot the light curve with uncertainty labels
# ======================================================
plt.figure(figsize=(12, 7))
ax = plt.gca()
ax.invert_yaxis()

# Data with error bars
ax.errorbar(time_shifted, mag, yerr=mag_err, fmt='o', color='green', markersize=6, label="Data")

# Horizontal lines for m_peak and m_t2
plt.axhline(m_peak, linestyle='--', label=f"m_peak = {m_peak:.3f}")
plt.axhline(m_t2, linestyle='--', label=f"m_peak+2 = {m_t2:.3f}")

# Vertical lines for t_peak and t2
plt.axvline(0, linestyle='--', color='red', 
            label=f"t_peak = 0 ± {sigma_tpeak:.2f} d")
plt.axvline(t2_central, linestyle='--', color='blue',
            label=f"t2 = {t2_central:.2f} ± {sigma_t2:.2f} d")

# Shaded uncertainty regions (use same info as legend)
plt.axvspan(-sigma_tpeak, sigma_tpeak, alpha=0.3, color='red')
plt.axvspan(t2_central - sigma_t2, t2_central + sigma_t2, alpha=0.3, color='cyan')

# Labels and formatting
plt.xlabel("Days since t_peak")
plt.ylabel("Magnitude")
plt.title("Nova Light Curve with t_peak and t2 Uncertainties")
mpl.rcParams.update({'font.size': 14})
plt.legend(loc='best', fontsize=12)
plt.tight_layout()
plt.show()


# ======================================================
# Print results
# ======================================================
print(f"t_peak (JD): {t_peak_JD:.5f} ± {sigma_tpeak:.5f} days")
print(f"t2 (days after t_peak): {t2_central:.5f} ± {sigma_t2:.5f} days")
