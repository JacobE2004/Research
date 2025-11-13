# By Jacob Ellerbook

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# ======================================================
# Load Light Curve
# ======================================================
file_path = r"C:\Users\Jmell\Dropbox\Research File\Nova_targets\SMCN_FOLDER\SMCN_2001_10a\SMC_2001_10a.dat"
readfile = np.loadtxt(file_path)

# Remove bad data points
readfile = readfile[readfile[:, 1] != 99.999]

# Extract columns
time = readfile[:, 0]
Mag = readfile[:, 1]
Magerr = readfile[:, 2]

# User-set values
tpeak_JD = 2452206.61720
m_peak = 11.912
t2_JD = 2452248.84640
manual_tpeak_before = None

# Shift for plotting
time_shifted = time - tpeak_JD

# ============================
# Manual overrides for t_peak bracket
# ============================
manual_tpeak_prev = 2452192.67324
manual_tpeak_next = 2452211.65131

# Manual t2 bracket
manual_t2_prev = 2452248.61823
manual_t2_next = 2452250.58688

# ======================================================
# t_peak UNCERTAINTY (using delta difference method)
# ======================================================

i_after = np.searchsorted(time, tpeak_JD)
i_before = i_after - 1

if manual_tpeak_prev is not None and manual_tpeak_next is not None:
    t_prev = manual_tpeak_prev
    t_next = manual_tpeak_next
else:
    if i_before < 0:
        t_prev = tpeak_JD
        t_next = time[0]
    elif i_after >= len(time):
        t_prev = time[-1]
        t_next = tpeak_JD
    else:
        t_prev = time[i_before]
        t_next = time[i_after]

# NEW method: difference of deltas, not full bracket
delta_t_before = tpeak_JD - t_prev
delta_t_after = t_next - tpeak_JD
tpeak_err = 0.5 * (delta_t_after - delta_t_before)

# Store these for plotting range (always absolute)
tpeak_plot_before = abs(delta_t_before)
tpeak_plot_after = abs(delta_t_after)

# ======================================================
# t2 UNCERTAINTY (manual bracket + delta difference method)
# ======================================================

if manual_t2_prev is not None and manual_t2_next is not None:
    t2_prev = manual_t2_prev
    t2_next = manual_t2_next
else:
    idx_t2 = np.argmin(np.abs(time - t2_JD))
    if idx_t2 == 0:
        t2_prev = t2_JD
        t2_next = time[1]
    elif idx_t2 == len(time)-1:
        t2_prev = time[-2]
        t2_next = t2_JD
    else:
        t2_prev = time[idx_t2 - 1]
        t2_next = time[idx_t2 + 1]

# Your new delta-based method for internal uncertainty
delta_t2_before = t2_JD - t2_prev
delta_t2_after = t2_next - t2_JD
t2_internal_err = 0.5 * (delta_t2_after - delta_t2_before)

# Final total quadrature
t2_total_err = np.sqrt(t2_internal_err**2 + tpeak_err**2)
t2_central = t2_JD - tpeak_JD

# ======================================================
# PLOT — Finalized Style Like Reference Image
# ======================================================

x_label_fontsize = 18
y_label_fontsize = 18

plt.figure(figsize=(16, 10))
ax = plt.gca()

ax.errorbar(time_shifted, Mag, yerr=Magerr, fmt='o',
            color='green', markersize=6, label="I-band data")

# Connect points with lines
#ax.plot(time_shifted, Mag, color='green', linewidth=1.5, alpha=0.6)

plt.gca().invert_yaxis()

plt.axhline(m_peak, color='blue', linestyle='--', linewidth=1.5,
            label=f"m_peak = {m_peak:.3f}")
plt.axhline(m_peak + 2.0, color='red', linestyle='--', linewidth=1.5,
            label=f"m_peak+2 = {m_peak + 2.0:.3f}")

plt.axvline(0, color='black', linestyle='--', linewidth=2, label="t_peak = 0")
plt.axvspan(-tpeak_plot_before, tpeak_plot_after,
            color='red', alpha=0.3, label="t_peak bracket")

plt.axvline(t2_central, color='purple', linestyle='--', linewidth=2,
            label=f"t2 = {t2_central:.2f} d")
plt.axvspan(t2_central - t2_total_err, t2_central + t2_total_err,
            color='blue', alpha=0.25, label="t2 uncertainty")

plt.ylim(23, 10)
plt.xlim(-60, 120)

ax.set_xlabel("Days since peak brightness", fontsize=x_label_fontsize)
ax.set_ylabel("Optical Brightness (mag)", fontsize=y_label_fontsize)
ax.set_title("SMCN-2001-10a", fontsize=20)

plt.legend(fontsize=13, loc='upper right')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# ======================================================
# TXT OUTPUT OPTION
# ======================================================
save_txt = input("Would you like to export the uncertainties to TXT? (y/n): ").strip().lower()

if save_txt == 'y':
    txt_filename = "uncertainty_output.txt"
    with open(txt_filename, "w") as f:
        f.write("Uncertainty Output Report\n")
        f.write("=========================\n")
        f.write(f"t_peak JD: {tpeak_JD:.5f}\n")
        f.write(f"t_peak bracket: {t_prev:.5f} to {t_next:.5f}\n")
        f.write(f"t_peak uncertainty: ±{tpeak_err:.5f} days\n\n")
        f.write(f"t2 JD: {t2_JD:.5f}\n")
        f.write(f"t2 bracket (manual): {t2_prev:.5f} to {t2_next:.5f}\n")
        f.write(f"t2 internal uncertainty: ±{t2_internal_err:.5f} days\n")
        f.write(f"t2 total uncertainty (quadrature): ±{t2_total_err:.5f} days\n")
    print(f"Saved: {txt_filename}")
