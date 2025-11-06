#By Jacob Ellerbook



import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ======================================================
# Load Light Curve
# ======================================================
file_path = r"C:\Users\Jmell\Dropbox\Research File\Nova_targets\SMCN_FOLDER\SMCN_2002_10a\SMCN_2002_10a.dat"
readfile = np.loadtxt(file_path)

readfile = readfile[readfile[:,1] != 99.999]  # Remove bad data points

#lines_to_skip = range(345, 348)
#mask = np.ones(readfile.shape[0], dtype=bool)
#mask[lines_to_skip] = False
#readfile = readfile[mask]

# Extract columns
time = readfile[:, 0]     # JD
Mag = readfile[:, 1]      # Magnitude
Magerr = readfile[:, 2]   # Mag error

# User-defined peak info
tpeak_JD = 2452563.69365   # Observed peak JD
m_peak = 12.212         # Observed peak mag

# Manual override for t_peak "before" uncertainty (days).
# Set to None to use the bracket-based value.
manual_tpeak_before = None  # or None


# This will manually set the brackets for tpeak before



# User-defined t2 JD (set this manually)
t2_JD = 2452570.38265    # Set your t2 JD here manually

time_shifted = time - tpeak_JD  # Shift so peak = 0

# ======================================================
# Calculate t_peak Uncertainties (bracket by previous/next observation)
# ======================================================
idx_peak = np.argmin(np.abs(time - tpeak_JD))

if idx_peak == 0:
    gap_before = 0.0
    # distance from tpeak_JD to the next observation
    gap_after = float(time[1] - tpeak_JD)
elif idx_peak == len(time)-1:
    # distance from tpeak_JD to the previous observation
    gap_before = float(tpeak_JD - time[-2])
    gap_after = 0.0
else:
    # bracket tpeak_JD by the neighbouring observation times (asymmetric)
    gap_before = float(tpeak_JD - time[idx_peak-1])
    gap_after  = float(time[idx_peak+1] - tpeak_JD)

# Optional manual override
if manual_tpeak_before is not None:
    gap_before = float(manual_tpeak_before)

# --- new: determine plotting-only bracket for t_peak that goes exactly between the two surrounding observations ---
# Find indices that bracket the user tpeak_JD (so plotting spans from previous obs to next obs)
idx_plot_after = np.searchsorted(time, tpeak_JD)
idx_plot_before = idx_plot_after - 1

if idx_plot_before < 0:
    # tpeak before the first observation: use first two observations (or fallback)
    tpeak_plot_before = 0.0
    if len(time) > 1:
        tpeak_plot_after = float(time[0] - tpeak_JD)
    else:
        tpeak_plot_after = 0.0
elif idx_plot_after >= len(time):
    # tpeak after the last observation: use last two observations (or fallback)
    if len(time) > 1:
        tpeak_plot_before = float(tpeak_JD - time[-1])
    else:
        tpeak_plot_before = 0.0
    tpeak_plot_after = 0.0
else:
    # Normal case: bracket between time[idx_plot_before] and time[idx_plot_after]
    tpeak_plot_before = float(tpeak_JD - time[idx_plot_before])
    tpeak_plot_after  = float(time[idx_plot_after] - tpeak_JD)

# ======================================================
# Calculate t2 Uncertainties (bracket by previous/next observation)
# ======================================================
idx_t2 = np.argmin(np.abs(time - t2_JD))

if idx_t2 == 0:
    t2_gap_before = 0.0
    t2_gap_after = float(time[1] - t2_JD)
elif idx_t2 == len(time)-1:
    t2_gap_before = float(t2_JD - time[-2])
    t2_gap_after = 0.0
else:
    # use distances from the user t2_JD to neighbouring observation times
    t2_gap_before = float(t2_JD - time[idx_t2-1])
    t2_gap_after  = float(time[idx_t2+1] - t2_JD)

# ensure non-negative (defensive)
t2_gap_before = abs(t2_gap_before)
t2_gap_after  = abs(t2_gap_after)

t2_central = t2_JD - tpeak_JD

# Debug prints to confirm what is being plotted for t2
print(f"[debug] t2_JD = {t2_JD}")
print(f"[debug] t2_central (days since peak) = {t2_central}")
print(f"[debug] nearest observation for t2: time[idx_t2] = {time[idx_t2]} (idx={idx_t2}), days since peak = {time[idx_t2] - tpeak_JD}")

# Asymmetric t2 uncertainties using quadrature (keep for terminal/calculation)
t2_err_before = np.sqrt(t2_gap_before**2 + gap_before**2)
t2_err_after  = np.sqrt(t2_gap_after**2 + gap_after**2)

# Bracket distances (for plotting of quadrature, kept for printing)
t2_bracket_before = t2_err_before
t2_bracket_after  = t2_err_after

# --- new: determine plotting-only bracket for t2 that goes exactly between the two surrounding observations ---
idx_plot_after_t2 = np.searchsorted(time, t2_JD)
idx_plot_before_t2 = idx_plot_after_t2 - 1

if idx_plot_before_t2 < 0:
    # t2 before first observation
    t2_plot_before = 0.0
    t2_plot_after = float(time[0] - t2_JD) if len(time) > 0 else 0.0
elif idx_plot_after_t2 >= len(time):
    # t2 after last observation
    t2_plot_before = float(t2_JD - time[-1]) if len(time) > 0 else 0.0
    t2_plot_after = 0.0
else:
    # Normal case: bracket between time[idx_plot_before_t2] and time[idx_plot_after_t2]
    t2_plot_before = float(t2_JD - time[idx_plot_before_t2])
    t2_plot_after  = float(time[idx_plot_after_t2] - t2_JD)

# ensure non-negative
t2_plot_before = abs(t2_plot_before)
t2_plot_after  = abs(t2_plot_after)

# ======================================================
# Plot Light Curve
# ======================================================
plt.figure(figsize=(16, 10))
ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_useOffset(False)
# Data points (single call; label only once so legend has one entry)
ax.errorbar(time_shifted, Mag, yerr=Magerr, fmt='o', color='green', markersize=8, label="I-band data")

# Add lines connecting all data points
#plt.plot(time_shifted, Mag, 'g-', alpha=0.7, linewidth=1.5)
plt.gca().invert_yaxis()

# Horizontal lines for peak and +2 mag
plt.axhline(m_peak, color='b', linestyle='--', label=f"m_peak = {m_peak:.3f}")
plt.axhline(m_peak + 2.0, color='r', linestyle='--', label=f"m_peak+2 = {m_peak + 2.0:.3f}")

# Vertical lines for t_peak and t2
plt.axvline(0, color='k', linestyle='--', label="t_peak = 0", linewidth=1.5, zorder=2)
plt.axvline(t2_central, color='purple', linestyle='--', label=f"t2 = {t2_central:.2f} d", linewidth=2.0, zorder=2)

# Shaded asymmetric uncertainty regions
# Use plotting-only t_peak brackets (span exactly between the two surrounding observations)
plt.axvspan(-tpeak_plot_before, tpeak_plot_after, color='red', alpha=0.25, zorder=1)
# Use plotting-only t2 brackets (span exactly between the two surrounding observations for t2)
plt.axvspan(t2_central - t2_plot_before, t2_central + t2_plot_after, color='cyan', alpha=0.25, zorder=1)

# ======================================================
# Axis Formatting
# ======================================================
ax.set_yticks(np.arange(0, 30, 1))
ax.set_yticks(np.arange(0, 30, 0.5), minor=True)
ax.set_xticks(np.arange(-200, 2800, 10))
ax.set_xticks(np.arange(-200, 2800, 5), minor=True)

# User-configurable tick font settings
axis_tick_fontsize = 14            # change this number to set tick label size
axis_tick_fontfamily = "DejaVu Sans"  # change to any installed font (e.g. 'serif', 'Arial')

# Apply tick font size and family to the axis numbering
ax.tick_params(axis='both', which='both', labelsize=axis_tick_fontsize)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontname(axis_tick_fontfamily)

plt.ylim(23, 10)
plt.xlim(-20, 180)
plt.xlabel('Days since peak brightness', fontsize=20)
plt.ylabel('Optical Brightness (mag)', fontsize=20)
plt.title('SMCN-2012-09a', fontsize=20)
mpl.rcParams.update({'font.size': 16})
plt.legend(loc='upper right', fontsize=14)
plt.tight_layout()
plt.show()

# ======================================================
# Print uncertainties
# ======================================================
print("---- Asymmetric uncertainties (quadrature for t2) ----")
print(f"t_peak uncertainty: -{gap_before:.5f} / +{gap_after:.5f} days")
# computed t2 uncertainties (quadrature) -- keep these as the terminal "calculated" values
print(f"t2 uncertainty (computed, quadrature): -{t2_err_before:.5f} / +{t2_err_after:.5f} days")
# plotted t2 brackets (span between the two neighbouring observations) -- what you show on the plot
print(f"t2 plotted bracket (between neighbouring observations): -{t2_plot_before:.5f} / +{t2_plot_after:.5f} days")
# (optional) also print the quadrature values used earlier for reference
print(f"(Reference: quadrature t2 bracket used previously: -{t2_bracket_before:.5f} / +{t2_bracket_after:.5f} days)")