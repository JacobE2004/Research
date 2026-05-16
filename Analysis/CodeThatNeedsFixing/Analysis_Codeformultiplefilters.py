import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import matplotlib as mpl

# ======================================================
# User Input
# ======================================================
file_path = r"C:\Users\Jmell\Dropbox\Research File - Copy\Nova_targets\SMCN_FOLDER\SMCN_1999_08a\SMCN1999_08a_query_2025-07-03_01_08_35.txt"  # ← CHANGE THIS TO YOUR FILE

t_peak_JD = 2453603.77450  # Change as needed
tmax_before_JD = 2453594.86300
tmax_after_JD  = 2453607.86717

t2_before_JD = 2453617.82348
t2_central_JD = 2453618.14102
t2_after_JD   = 2453619.79935

# ======================================================
# Load File into DataFrame (assumed: JD, Filter, Mag, MagErr)
# ======================================================
df = pd.read_csv(file_path, delim_whitespace=True, comment='#', header=None,
                 names=["JD", "Filter", "Mag", "MagErr"])

# Time shift to days since peak
df["DaysAfterPeak"] = df["JD"] - t_peak_JD

# ======================================================
# Compute uncertainties
# ======================================================
tmax_err_before = t_peak_JD - tmax_before_JD
tmax_err_after  = tmax_after_JD - t_peak_JD

t2_central = t2_central_JD - t_peak_JD
t2_err_before = t2_central_JD - t2_before_JD
t2_err_after  = t2_after_JD - t2_central_JD

# ======================================================
# Plotting Setup
# ======================================================
fig, ax = plt.subplots(figsize=(16, 9))
plt.subplots_adjust(left=0.25)

ax.invert_yaxis()
ax.set_xlabel("Days Since Peak")
ax.set_ylabel("Optical Brightness (Mag)")
ax.set_title("Nova Light Curve with Multi-Filter + Uncertainties")

# Plot horizontal lines: Peak and Peak+2
peak_mag = df["Mag"].min()
ax.plot([-100, 500], [peak_mag, peak_mag], 'k--', label="Peak Mag")
ax.plot([-100, 500], [peak_mag + 2, peak_mag + 2], 'k--', label="Peak +2 Mag")

# Plot vertical lines: t_max and t2
ax.axvline(0, linestyle='--', color='k', label='tₘₐₓ')
ax.axvline(t2_central, linestyle='--', color='k', label='t₂')

# Plot shaded uncertainties
ax.axvspan(-tmax_err_before, tmax_err_after, color='red', alpha=0.2,
           label=f"tₘₐₓ uncertainty: -{tmax_err_before:.2f}/+{tmax_err_after:.2f}")
ax.axvspan(t2_central - t2_err_before, t2_central + t2_err_after, color='cyan', alpha=0.2,
           label=f"t₂ uncertainty: -{t2_err_before:.2f}/+{t2_err_after:.2f}")

# Plot all filters but keep track for interactivity
plots = {}
for filt in df['Filter'].unique():
    filt_data = df[df['Filter'] == filt]
    line = ax.errorbar(filt_data["DaysAfterPeak"], filt_data["Mag"], yerr=filt_data["MagErr"],
                       fmt='o', label=f"{filt}-band", alpha=0.8)
    plots[filt] = line[0]  # Keep only the Line2D artist

# Add tmax/t2 uncertainty text
ax.text(-tmax_err_before + 0.1, peak_mag + 2.5,
        f"tₘₐₓ uncertainty: -{tmax_err_before:.2f} / +{tmax_err_after:.2f} days",
        fontsize=12, color='red')
ax.text(t2_central - t2_err_before + 0.1, peak_mag + 2.9,
        f"t₂ uncertainty: -{t2_err_before:.2f} / +{t2_err_after:.2f} days",
        fontsize=12, color='blue')

# Axis limits and ticks
ax.set_xlim(-30, 180)
ax.set_ylim(peak_mag + 3, peak_mag - 0.5)
ax.set_xticks(np.arange(-30, 200, 20))
ax.set_xticks(np.arange(-30, 200, 10), minor=True)
ax.set_yticks(np.arange(0, 30, 1))
ax.set_yticks(np.arange(0, 30, 0.5), minor=True)
ax.tick_params(which='both', width=2)
ax.tick_params(which='major', length=8)
ax.tick_params(which='minor', length=4)

# ======================================================
# Interactive Checkboxes
# ======================================================
rax = plt.axes([0.03, 0.4, 0.15, 0.25])
labels = list(plots.keys())
visibility = [True] * len(labels)
check = CheckButtons(rax, labels, visibility)

def func(label):
    plots[label].set_visible(not plots[label].get_visible())
    plt.draw()

check.on_clicked(func)

# ======================================================
# Show Plot
# ======================================================
plt.legend(loc='upper right', fontsize=12)
plt.tight_layout()
plt.show()
