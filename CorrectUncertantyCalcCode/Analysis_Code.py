# By Jacob Ellerbook

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# ======================================================
# Load Light Curve
# ======================================================
file_path = r"C:\Users\Jmell\Dropbox\Research File\Nova_targets\LMCN_Folder\LMCN_2018_05a-Light curve not made\download.csv"

ext = os.path.splitext(file_path)[1].lower()

if ext == ".csv":
    import csv


    # read entire file into memory so we can handle a couple of
    # special lines (object name on first line, a header line prefixed
    # with "#", etc.) before handing the remainder off to csv.DictReader.
    raw_lines = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # ignore empty lines completely
            if not line.strip():
                continue
            raw_lines.append(line.rstrip("\n"))

    # drop an initial name/comment line if it doesn't look like data
    # (no comma and does not start with a digit)
    if raw_lines and "," not in raw_lines[0] and not raw_lines[0][0].isdigit():
        raw_lines.pop(0)

    # if the header line begins with '#', strip that off so csv
    # sees clean field names
    if raw_lines and raw_lines[0].lstrip().startswith("#"):
        raw_lines[0] = raw_lines[0].lstrip().lstrip("#").lstrip()

    # hand the processed lines back to csv using StringIO
    import io
    rows = []
    with io.StringIO("\n".join(raw_lines)) as f:
        reader = csv.DictReader(f)
        # normalise the fieldnames just as before, but now the leading '#'
        # has already been removed if it existed
        fieldnames = [fn.strip() for fn in (reader.fieldnames or [])]
        fn_map = {fn.lower(): fn for fn in fieldnames}

        def _get(row, keys, default=""):
            for k in keys:
                k = k.lower()
                if k in fn_map:
                    return row.get(fn_map[k], default)
            return default

        for r in reader:
            t = _get(r, ["HJD", "JD", "JD(TCB)", "Julian Date", "julian_date", "jd_utc", "JD (UTC)"])
            filt = _get(r, ["Filter", "Band", "Passband", "filter", "band"], default="NA")
            m = _get(r, ["mag", "Magnitude", "magnitude", "Mag", "mag_value", "averagemag"])
            me = _get(r, ["mag_err", "Uncertainty", "uncertainty", "Mag Error", "magerror", "MagErr", "error", "err"], default="0.05")

            # convert null/untrusted/empty magnitudes to NaN rather than
            # outright dropping the row; the downstream numeric filtering
            # will handle the NaNs gracefully but we preserve the time
            # information in case that's useful.
            if m and m.lower() not in ["untrusted", ""]:
                if m.lower() == "null":
                    m_value = "nan"
                else:
                    m_value = m
                rows.append([t, filt, m_value, me])

    raw = np.array(rows, dtype=str)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

else:
    raw = np.genfromtxt(file_path, dtype=str, comments='#', invalid_raise=False)

if raw.size == 0:
    raise ValueError("No data found in the file (file may be empty or comment-only).")

if raw.ndim == 1:
    raw = raw.reshape(1, -1)

def _is_float(x):
    try:
        float(x)
        return True
    except:
        return False

# Extract data based on file format
if ext == ".csv":
    raw_time = raw[:, 0]
    raw_mag = np.char.lstrip(raw[:, 2], "><")
    raw_err = np.char.lstrip(raw[:, 3], "><") if raw.shape[1] > 3 else np.full_like(raw_mag, "0.05")
else:
    readfile = np.loadtxt(file_path)
    readfile = readfile[readfile[:, 1] != 99.999]
    raw_time = readfile[:, 0].astype(str)
    raw_mag = readfile[:, 1].astype(str)
    raw_err = readfile[:, 2].astype(str)

# Keep only rows where time and mag are numeric
mask_numeric = (np.vectorize(_is_float)(raw_time) &
                np.vectorize(_is_float)(raw_mag))

raw_time = raw_time[mask_numeric]
raw_mag = raw_mag[mask_numeric]
raw_err = raw_err[mask_numeric]

time = raw_time.astype(float)
Mag = raw_mag.astype(float)
Magerr = raw_err.astype(float)

# if we managed to read rows but all mags are NaN the user probably
# supplied a file with only null values; warn them and continue so the
# downstream code does not crash but the plot will be empty.
if Mag.size > 0 and np.all(np.isnan(Mag)):
    print("Warning: all magnitude entries are NaN (file may contain only 'null' values).")

# if everything was stripped out by the numeric mask above there is
# nothing to do, so error early.
if time.size == 0:
    raise ValueError("No numeric time/magnitude data found in the file after parsing.")

# Remove bad data points
good = (Mag != 99.999) & (~np.isclose(Magerr, 99.990, atol=1e-6))
time = time[good]
Mag = Mag[good]
Magerr = Magerr[good]

# User-set values
tpeak_JD = 2458253.94785
m_peak = 13.01
t2_JD = 2458308.01985

# Shift for plotting
time_shifted = time - tpeak_JD

# ============================
# Manual overrides for t_peak bracket
# ============================
manual_tpeak_prev = 2458000.85397
manual_tpeak_next = 2458288.85397

# Manual t2 bracket
manual_t2_prev = 2458288.92797
manual_t2_next = 2458319.09355

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
            color='green', markersize=6, label=(f"{chosen_filter}-band data" if 'chosen_filter' in globals() else "Light curve data"))

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
ax.set_title("LMCN-2018-05a", fontsize=20)

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

