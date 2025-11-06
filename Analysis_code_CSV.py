#THIS CODE NEEDS TO BE FIXED TO RUN UNCERTANTIES CORRECTLY



import numpy as np          # Numerical operations
import pandas as pd         # Data handling
import matplotlib.pyplot as plt  # Plotting
import matplotlib as mpl     # Matplotlib configuration

# --- Data Loading and Cleaning ---
file_path = r"C:\Users\Jmell\Dropbox\Research File\Nova_targets\LMCN_Folder\LMCN_2019_07a\light_curve_b1542ffe-ccfc-42c2-92ad-77caa56716b0.csv"  # Corrected path
_df = pd.read_csv(file_path)  # Read CSV into DataFrame
print("Columns in the CSV file:", _df.columns)  # Display column names
# Convert 'mag' column to numeric first
_df['mag'] = pd.to_numeric(_df['mag'], errors='coerce')
_df['mag_err'] = pd.to_numeric(_df['mag_err'], errors='coerce')
# Filter out magnitudes >= 99.00 (non-detections/invalid data)
df = _df[_df['mag'] < 99.00].copy()
# Coerce necessary columns to numeric, converting invalid entries to NaN
df['HJD']     = pd.to_numeric(df['HJD'],     errors='coerce')  # Convert HJD to float
df['mag']     = pd.to_numeric(df['mag'],     errors='coerce')  # Convert magnitude to float
df['mag_err'] = pd.to_numeric(df['mag_err'], errors='coerce')  # Convert magnitude error to float
df = df.dropna(subset=['HJD', 'mag', 'mag_err'])  # Drop rows with NaNs in key columns

# --- User-defined peak info ---
tpeak_JD = 2458697.66345   # Observed peak JD (USER DEFINED)
m_peak = 11.355           # Observed peak mag (USER DEFINED)
t2_JD = 2458726.02055     # Observed t2 JD (USER DEFINED)
# --- Time Shift ---
time       = df['HJD'].to_numpy()    # Extract HJD as numpy array
Mag        = df['mag'].to_numpy()    # Extract magnitude array
Magerr     = df['mag_err'].to_numpy()# Extract magnitude error array
time_shifted = time - tpeak_JD         # Shift time so peak occurs at day 0

# ======================================================
# Calculate t_peak Uncertainties (bracket by previous/next observation)
# ======================================================
idx_peak = np.argmin(np.abs(time - tpeak_JD))

if idx_peak == 0:
    gap_before = 0
    gap_after = time[1] - time[0]
elif idx_peak == len(time)-1:
    gap_before = time[-1] - time[-2]
    gap_after = 0
else:
    gap_before = time[idx_peak] - time[idx_peak-1]
    gap_after  = time[idx_peak+1] - time[idx_peak]

# If there are very few observations before peak, the true rise could have
# occurred earlier than the first detection. In that case, extend the
# "before" uncertainty continuously back to the earliest observation.
n_before = np.sum(time < tpeak_JD)
continuous_back = False
if n_before < 1:
    # Span back to earliest observation time (so shading goes back continuously)
    earliest = np.min(time)
    gap_before = tpeak_JD - earliest
    continuous_back = True

# ======================================================
# Calculate t2 and Its Uncertainties
# ======================================================
# Use user-defined t2_JD instead of calculating it
t2_exact = t2_JD
t2_central = t2_exact - tpeak_JD

# Find the bracketing observations around the user-defined t2_JD using searchsorted
idx_after_t2 = np.searchsorted(time, t2_exact)
if idx_after_t2 == 0:
    i_before, i_after = 0, 1
elif idx_after_t2 >= len(time):
    i_before, i_after = len(time) - 2, len(time) - 1
else:
    i_before, i_after = idx_after_t2 - 1, idx_after_t2

# times bracketing t2_exact
t1 = time[i_before]
t2 = time[i_after]

# Distances from t2_exact to bracketing observation times (used for plotting and errors)
t2_bracket_before = t2_exact - t1
t2_bracket_after = t2 - t2_exact

# Asymmetric t2 uncertainties using quadrature that include tpeak uncertainty (gap_before/gap_after)
t2_err_before = np.sqrt((t2_bracket_before)**2 + gap_before**2)
t2_err_after  = np.sqrt((t2_bracket_after)**2 + gap_after**2)

# ------------------------------------------------------
# Interpolate magnitude at t2 and estimate mag uncertainty
# using the two bracketing observations and error propagation
# ------------------------------------------------------
# Find indices that bracket the user-defined t2_JD
idx_after_t2 = np.searchsorted(time, t2_exact)
if idx_after_t2 == 0:
    i_before, i_after = 0, 1
elif idx_after_t2 >= len(time):
    i_before, i_after = len(time) - 2, len(time) - 1
else:
    i_before, i_after = idx_after_t2 - 1, idx_after_t2

t_before = time[i_before]
m_before = Mag[i_before]
merr_before = Magerr[i_before]
t_after = time[i_after]
m_after = Mag[i_after]
merr_after = Magerr[i_after]

if t_after == t_before:
    frac_t = 0.0
else:
    frac_t = (t2_exact - t_before) / (t_after - t_before)

# Interpolated magnitude at t2 (for reporting only)
m_at_t2 = m_before + frac_t * (m_after - m_before)

# Propagate mag errors through linear interpolation
# Var(m_interp) = (1-frac)^2 * err1^2 + frac^2 * err2^2
merr_at_t2 = np.sqrt((1.0 - frac_t)**2 * merr_before**2 + (frac_t**2) * merr_after**2)

# Note: we do NOT plot the mag uncertainty at t2; only the time uncertainties

# --- Plot Setup ---
plt.figure(figsize=(16, 9))       # Create figure sized 16x9 inches
ax = plt.gca()                    # Get current axes
ax.get_xaxis().get_major_formatter().set_useOffset(False)  # Disable scientific offset on x-axis
plt.gca().invert_yaxis()          # Invert y-axis: lower magnitude = brighter

# --- Plot Data Points ---
ax.errorbar(time_shifted, Mag, yerr=Magerr,
             fmt='go', markersize=6, linestyle='None',
             label='I-band')      # Plot magnitudes with green circles (no line)

# Add line connecting the data points (comment/uncomment to toggle)
#plt.plot(time_shifted, Mag, 'g', linewidth=1, alpha=0.7, label='Light curve')

# --- Reference Lines ---
Ipeakx = [-200, 3000]             # X-range for horizontal lines
Ipeaky = [m_peak, m_peak]         # Y-value for peak magnitude
plt.plot(Ipeakx, Ipeaky, 'k--', label=f'Peak mag = {m_peak:.3f}')             # Dashed line at peak
plt.plot(Ipeakx, [y+2 for y in Ipeaky], 'k--', label=f'Peak + 2 mag = {m_peak+2:.3f}')    # Dashed line at peak+2
plt.plot([0, 0], [40, 5], 'k--')    # Vertical line at peak day
plt.plot([t2_central]*2, [40, 5], 'k--')   # Vertical line at t2

# --- Uncertainty Regions ---
plt.axvspan(-gap_before, gap_after,
            color='red', alpha=0.25,
            label=f"tₘₐₓ uncertainty")  # Shade tₘₐₓ uncertainty in red
plt.axvspan(t2_central - t2_bracket_before,
            t2_central + t2_bracket_after,
            color='blue', alpha=0.25,
            label=f"t₂ uncertainty")     # Shade t₂ uncertainty in blue

# --- Axis Formatting ---
ax.set_yticks(np.arange(0, 30, 1))            # Major y-ticks every 1 magnitude
ax.set_yticks(np.arange(0, 30, 0.5), minor=True)  # Minor y-ticks every 0.5 mag
ax.set_xticks(np.arange(-80, 181, 10))        # Major x-ticks every 20 days
ax.set_xticks(np.arange(-80, 181, 1), minor=True) # Minor x-ticks every 10 days
plt.ylim(19, 8)                            # Inverted y-axis limits
plt.xlim(-20, 180)                             # X-axis limits
plt.xlabel('Days since peak brightness')      # X-axis label
plt.ylabel('Magnitude')                       # Y-axis label
plt.title('LMCN-2019-07a Light Curve', fontsize=18)  # Plot title

# --- Legend ---
plt.legend(loc='upper right', title='Key')    # Legend with title 'Key'

# --- Final Touches ---
mpl.rcParams.update({'font.size': 16})       # Increase font size
plt.tight_layout()                           # Tight layout
plt.show()                                   # Display plot

# --- Print Uncertainty Values in Console ---
print("---- Asymmetric uncertainties (quadrature for t2) ----")
print(f"t_peak uncertainty: -{gap_before:.5f} / +{gap_after:.5f} days")
print(f"t2 uncertainty: -{t2_err_before:.5f} / +{t2_err_after:.5f} days")
print(f"(t2 observation gaps (brackets): -{t2_bracket_before:.5f} / +{t2_bracket_after:.5f} days)")
print(f"t2 central value: {t2_central:.2f} days after peak")
print(f"User-defined t2 JD: {t2_JD}")
print("")
print("---- Interpolated magnitude at t2 (reported only) ----")
print(f"Interpolated mag at t2 (JD={t2_exact}): {m_at_t2:.4f} mag")
print(f"Interpolated mag uncertainty at t2: {merr_at_t2:.4f} mag")
if continuous_back:
    print("NOTE: fewer than 10 observations before tpeak — 'before' uncertainty extended back to earliest observation")
