import numpy as np          # Numerical operations
import pandas as pd         # Data handling
import matplotlib.pyplot as plt  # Plotting
import matplotlib as mpl     # Matplotlib configuration

# --- Data Loading and Cleaning ---
file_path = r"C:\Users\Jmell\Dropbox\Research File\Nova_targets\SMCN_FOLDER\SMCN_2016_10a\SMCN_2016_10a_ASASSN_light-curves.csv"  # Corrected path
_df = pd.read_csv(file_path)  # Read CSV into DataFrame
print("Columns in the CSV file:", _df.columns)  # Display column names
# Convert 'mag' column to numeric first
_df['mag'] = pd.to_numeric(_df['mag'], errors='coerce')
df = _df[_df['mag'] < 99] #Ignores all plots that will have 99.99
# Coerce necessary columns to numeric, converting invalid entries to NaN
df = _df.copy()
df['HJD']     = pd.to_numeric(df['HJD'],     errors='coerce')  # Convert HJD to float
df['mag']     = pd.to_numeric(df['mag'],     errors='coerce')  # Convert magnitude to float
df['mag_err'] = pd.to_numeric(df['mag_err'], errors='coerce')  # Convert magnitude error to float
df = df.dropna(subset=['HJD', 'mag', 'mag_err'])  # Drop rows with NaNs in key columns

# --- Time Shift ---
time       = df['HJD'].to_numpy()    # Extract HJD as numpy array
Mag        = df['mag'].to_numpy()    # Extract magnitude array
Magerr     = df['mag_err'].to_numpy()# Extract magnitude error array
tpeak_JD   = 2457683.7499564        # Julian Date of peak brightness
time       = time - tpeak_JD         # Shift time so peak occurs at day 0

# --- Define Uncertainties ---
# tₘₐₓ uncertainty: days before and after peak
JD1, JD2, JD3      = 2459155.74379, 2459155.74508, 2459155.74639   # JDs bracketing the peak
tmax_err_before    = JD2 - JD1  # Days before peak
tmax_err_after     = JD3 - JD2  # Days after peak

# t₂ uncertainty: days before and after fading by 2 magnitudes
JD1_t2, JD2_t2, JD3_t2 = 2459174.42143, 2459174.42188, 2459174.42265  # JDs bracketing t2
t2_err_before       = JD2_t2 - JD1_t2  # Days before t2
t2_err_after        = JD3_t2 - JD2_t2  # Days after t2
t2_central          = JD2_t2 - tpeak_JD        # t2 relative to peak

# --- Plot Setup ---
plt.figure(figsize=(16, 9))       # Create figure sized 16x9 inches
ax = plt.gca()                    # Get current axes
ax.get_xaxis().get_major_formatter().set_useOffset(False)  # Disable scientific offset on x-axis
plt.gca().invert_yaxis()          # Invert y-axis: lower magnitude = brighter

# --- Plot Data Points ---
ax.errorbar(time, Mag, yerr=Magerr,
             fmt='go', markersize=6,
             label='I-band')      # Plot magnitudes with green circles

# --- Reference Lines ---
Ipeakx = [-200, 3000]             # X-range for horizontal lines
Ipeaky = [11.474, 11.474]         # Y-value for peak magnitude
plt.plot(Ipeakx, Ipeaky, 'k--', label='Peak mag = 11.474')             # Dashed line at peak
plt.plot(Ipeakx, [y+2 for y in Ipeaky], 'k--', label='Peak + 2 mag')    # Dashed line at peak+2
plt.plot([0, 0], [40, 5], 'k--')    # Vertical line at peak day
plt.plot([t2_central]*2, [40, 5], 'k--')   # Vertical line at t2

# --- Uncertainty Regions ---
plt.axvspan(-tmax_err_before, tmax_err_after,
            color='red', alpha=0.25,
            label=f"tₘₐₓ unc.: −{tmax_err_before:.3f}/+{tmax_err_after:.3f} d")  # Shade tₘₐₓ uncertainty in red
plt.axvspan(t2_central - t2_err_before,
            t2_central + t2_err_after,
            color='blue', alpha=0.25,
            label=f"t₂ unc.: −{t2_err_before:.3f}/+{t2_err_after:.3f} d")     # Shade t₂ uncertainty in blue

# --- Axis Formatting ---
ax.set_yticks(np.arange(0, 30, 1))            # Major y-ticks every 1 magnitude
ax.set_yticks(np.arange(0, 30, 0.5), minor=True)  # Minor y-ticks every 0.5 mag
ax.set_xticks(np.arange(-80, 181, 10))        # Major x-ticks every 20 days
ax.set_xticks(np.arange(-80, 181, 1), minor=True) # Minor x-ticks every 10 days
plt.ylim(19, 8)                            # Inverted y-axis limits
plt.xlim(-20, 180)                             # X-axis limits
plt.xlabel('Days since peak brightness')      # X-axis label
plt.ylabel('Magnitude')                       # Y-axis label
plt.title('SMCN 2020_10a Light Curve', fontsize=18)  # Plot title

# --- Legend ---
plt.legend(loc='upper right', title='Key')    # Legend with title 'Key'

# --- Final Touches ---
mpl.rcParams.update({'font.size': 16})       # Increase font size
plt.tight_layout()                           # Tight layout
plt.show()                                   # Display plot

# --- Print Uncertainty Values in Console ---
print(f"tₘₐₓ uncertainty: −{tmax_err_before:.5f} d / +{tmax_err_after:.5f} d")
print(f"t₂   uncertainty: −{t2_err_before:.5f} d / +{t2_err_after:.5f} d")
