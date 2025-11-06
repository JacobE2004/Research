import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
import os

from astropy.io import fits
import pandas as pd

file_path = r"C:\Users\Jmell\Dropbox\Research File - Copy\Nova_targets\LMCN_FOLDER\LMCN_2015_03a\light_curve_LMCN_2015_03.csv"

def load_data_generic(file_path):
    ext = os.path.splitext(file_path)[-1].lower()

    if ext in ['.txt', '.dat', '.csv']:
        try:
            # Force whitespace-delimited parsing, no header
            df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=['jd', 'mag', 'magerr'])
            df.columns = [col.strip().lower() for col in df.columns]
            print("Read as whitespace-delimited file.")
        except Exception as e:
            raise ValueError(f"Could not read file {file_path} due to: {e}")

        print("Detected columns:", df.columns)
        print(df.head())  # For verification

        return df['jd'].values, df['mag'].values, df['magerr'].values

    elif ext in ['.fits', '.fit']:
        with fits.open(file_path) as hdul:
            data = hdul[1].data
            colnames = [col.lower() for col in data.names]

            jd_col = next((c for c in colnames if c in ['jd', 'hjd', 'time']), None)
            mag_col = next((c for c in colnames if c in ['mag', 'magnitude']), None)
            err_col = next((c for c in colnames if 'err' in c), None)

            if None in [jd_col, mag_col, err_col]:
                raise ValueError("Couldn't find all required columns in FITS file.")

            return data[jd_col], data[mag_col], data[err_col]

    else:
        raise ValueError("Unsupported file format.")


# Load the data correctly now
time, Mag, Magerr = load_data_generic(file_path)

# Define the JD of peak brightness (t_peak) for this nova.
tpeak_JD = 2457160.44559

# Adjust the time so that t_peak is at 0 days.
time = time - tpeak_JD

# === t_max and t2 uncertainties ===
JD1 = 2453594.86300
JD2 = 2453603.77450
JD3 = 2453607.86717
tmax_err_before = JD2 - JD1
tmax_err_after  = JD3 - JD2

JD1_t2 = 2453617.82348
JD2_t2 = 2453618.14102
JD3_t2 = 2453619.79935
t2_err_before = JD2_t2 - JD1_t2
t2_err_after  = JD3_t2 - JD2_t2
t2_central = JD2_t2 - tpeak_JD

# === Plotting ===
plt.figure(figsize=(16, 9))
ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_useOffset(False)
plt.gca().invert_yaxis()

ax.errorbar(time, Mag, yerr=Magerr, fmt='go', markersize=8, label=r"$\it{\;I}$-band")

Ipeakx = [-200, 3000]
Ipeaky = [12.783, 12.783]
Ipeakyplus2 = [12.783 + 2, 12.783 + 2]

plt.plot(Ipeakx, Ipeaky, 'k--')
plt.plot(Ipeakx, Ipeakyplus2, 'k--')

plt.plot([0, 0], [40, 5], 'k--')
plt.plot([t2_central, t2_central], [40, 5], 'k--')

plt.axvspan(-tmax_err_before, tmax_err_after, color='red', alpha=0.25, label=f"tₘₐₓ uncertainty: -{tmax_err_before:.2f}/+{tmax_err_after:.2f} days")
plt.axvspan(t2_central - t2_err_before, t2_central + t2_err_after, color='cyan', alpha=0.25, label=f"t₂ uncertainty: -{t2_err_before:.2f}/+{t2_err_after:.2f} days")

ax.text(-tmax_err_before + 0.1, 24, f"tₘₐₓ uncertainty: -{tmax_err_before:.3f} / +{tmax_err_after:.3f} days", color='red', fontsize=14)
ax.text(t2_central - t2_err_before + 0.1, 23, f"t₂ uncertainty: -{t2_err_before:.3f} / +{t2_err_after:.3f} days", color='red', fontsize=14)

major_yticks = np.arange(0, 30, 1)
minor_yticks = np.arange(0, 30, 0.5)
ax.set_yticks(major_yticks)
ax.set_yticks(minor_yticks, minor=True)

major_xticks = np.arange(-200, 2800, 20)
minor_xticks = np.arange(-200, 2800, 10)
ax.set_xticks(major_xticks)
ax.set_xticks(minor_xticks, minor=True)

plt.ylim(23, 10.5)
plt.xlim(-30, 180)
plt.xlabel('Days since peak brightness')
plt.ylabel('Optical Brightness')
plt.tick_params(width=3, length=8)
plt.title('SMCN_2005_08a', fontsize=20)
plt.tick_params(which='minor', width=2, length=4)

mpl.rcParams.update({'font.size': 20})
plt.tight_layout()

plt.legend(bbox_to_anchor=(0.85, 1.0), loc=2, borderaxespad=0.)
plt.legend(numpoints=1)
plt.show()

# Print uncertainties
print(f"tₘₐₓ uncertainty: -{tmax_err_before:.5f} days / +{tmax_err_after:.5f} days")
print(f"t₂ uncertainty: -{t2_err_before:.5f} days / +{t2_err_after:.5f} days")
