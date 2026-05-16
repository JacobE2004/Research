# By Jacob Ellerbook

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import csv
import io

# ======================================================
# Load Light Curve
# ======================================================
file_path = r"C:\Users\Jmell\Dropbox\Research File\Nova_targets\LMCN_Folder\LMCN_2020_11a\aavsodata_69ace0ffa9975.txt"

ext = os.path.splitext(file_path)[1].lower()

def _is_float(x):
    try:
        float(str(x).strip())
        return True
    except:
        return False

def read_header_based_file(file_path):
    """
    Reads header-based files such as:
    - AAVSO .csv
    - AAVSO .txt that is really comma-separated
    - other delimited text files with headers

    Returns array with columns:
    [time, filter, mag, err]
    """
    with open(file_path, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        text = f.read()

    if not text.strip():
        raise ValueError("File is empty.")

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("File contains no non-empty lines.")

    # Find the header line
    header_idx = None
    for i, line in enumerate(lines):
        test = line.lstrip("#").strip().lower()
        if ("jd" in test or "hjd" in test) and ("mag" in test or "magnitude" in test):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find a valid header line containing JD/HJD and Magnitude.")

    lines = lines[header_idx:]
    lines[0] = lines[0].lstrip("#").strip()

    sample = "\n".join(lines[:10])

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        delim = dialect.delimiter
    except:
        header_line = lines[0]
        if "\t" in header_line:
            delim = "\t"
        elif "," in header_line:
            delim = ","
        elif ";" in header_line:
            delim = ";"
        elif "|" in header_line:
            delim = "|"
        else:
            raise ValueError("Could not detect delimiter in header-based file.")

    rows = []
    reader = csv.reader(io.StringIO("\n".join(lines)), delimiter=delim)
    all_rows = list(reader)

    if not all_rows:
        raise ValueError("No rows found in header-based file.")

    headers = [str(h).strip().lstrip("\ufeff") for h in all_rows[0]]
    headers_lower = [h.lower() for h in headers]

    def _find_col(possible_names, required=False, default=None):
        for name in possible_names:
            name_l = name.lower()
            for i, h in enumerate(headers_lower):
                if h == name_l:
                    return i
        if required:
            raise ValueError(f"Required column not found. Tried: {possible_names}")
        return default

    # Prefer JD first, because some AAVSO files have blank HJD columns
    time_idx = _find_col(
        ["JD", "HJD", "JD(TCB)", "Julian Date", "julian_date", "jd_utc", "JD (UTC)"],
        required=True
    )
    filt_idx = _find_col(
        ["Filter", "Band", "Passband", "filter", "band"],
        required=False,
        default=None
    )
    mag_idx = _find_col(
        ["mag", "Magnitude", "magnitude", "Mag", "mag_value", "averagemag"],
        required=True
    )
    err_idx = _find_col(
        ["mag_err", "Uncertainty", "uncertainty", "HQuncertainty",
         "Mag Error", "magerror", "MagErr", "error", "err"],
        required=False,
        default=None
    )

    for r in all_rows[1:]:
        if not r:
            continue

        if len(r) < len(headers):
            r = r + [""] * (len(headers) - len(r))

        t = str(r[time_idx]).strip() if time_idx is not None and time_idx < len(r) else ""
        filt = str(r[filt_idx]).strip() if filt_idx is not None and filt_idx < len(r) else "NA"
        m = str(r[mag_idx]).strip() if mag_idx is not None and mag_idx < len(r) else ""
        me = str(r[err_idx]).strip() if err_idx is not None and err_idx < len(r) else "0.05"

        m = m.lstrip("><")
        me = me.lstrip("><")

        if m.lower() in ["null", "untrusted", ""]:
            m = "nan"

        if me.lower() in ["null", "none", "nan", ""]:
            me = "0.05"

        rows.append([t, filt, m, me])

    raw = np.array(rows, dtype=str)
    if raw.size == 0:
        raise ValueError("Header parser found no data rows.")

    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    return raw

def read_numeric_text_file(file_path):
    """
    Reads plain numeric files with no header.
    Expected columns:
      0 = time
      1 = magnitude
      2 = uncertainty
    """
    readfile = np.loadtxt(file_path)

    if readfile.ndim == 1:
        readfile = readfile.reshape(1, -1)

    if readfile.shape[1] < 3:
        raise ValueError("Numeric text file must have at least 3 columns: time, mag, err.")

    raw_time = readfile[:, 0].astype(str)
    raw_mag = readfile[:, 1].astype(str)
    raw_err = readfile[:, 2].astype(str)

    raw = np.column_stack([raw_time, np.full_like(raw_time, "NA"), raw_mag, raw_err])
    return raw

# ======================================================
# Choose parser automatically
# ======================================================
raw = None

if ext in [".csv", ".txt", ".dat"]:
    try:
        raw = read_header_based_file(file_path)
    except Exception as e:
        print("Header-based parser failed, trying numeric fallback:")
        print(e)
        raw = read_numeric_text_file(file_path)
else:
    raw = read_numeric_text_file(file_path)

if raw is None or raw.size == 0:
    raise ValueError("No data found in the file (file may be empty or comment-only).")

if raw.ndim == 1:
    raw = raw.reshape(1, -1)

# ======================================================
# Extract parsed columns
# ======================================================
raw_time = np.array([str(x).strip() for x in raw[:, 0]])
raw_mag  = np.array([str(x).strip().lstrip("><") for x in raw[:, 2]])
raw_err  = np.array([str(x).strip().lstrip("><") for x in raw[:, 3]]) if raw.shape[1] > 3 else np.full_like(raw_mag, "0.05")

# Keep only rows where time and mag are numeric
mask_numeric = np.array([_is_float(t) and _is_float(m) for t, m in zip(raw_time, raw_mag)])

raw_time = raw_time[mask_numeric]
raw_mag = raw_mag[mask_numeric]
raw_err = raw_err[mask_numeric]

time = raw_time.astype(float)
Mag = raw_mag.astype(float)
Magerr = np.array([float(x) if _is_float(x) else 0.05 for x in raw_err], dtype=float)

if Mag.size > 0 and np.all(np.isnan(Mag)):
    print("Warning: all magnitude entries are NaN (file may contain only 'null' values).")

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

delta_t_before = tpeak_JD - t_prev
delta_t_after = t_next - tpeak_JD
tpeak_err = 0.5 * (delta_t_after - delta_t_before)

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
    elif idx_t2 == len(time) - 1:
        t2_prev = time[-2]
        t2_next = t2_JD
    else:
        t2_prev = time[idx_t2 - 1]
        t2_next = time[idx_t2 + 1]

delta_t2_before = t2_JD - t2_prev
delta_t2_after = t2_next - t2_JD
t2_internal_err = 0.5 * (delta_t2_after - delta_t2_before)

t2_total_err = np.sqrt(t2_internal_err**2 + tpeak_err**2)
t2_central = t2_JD - tpeak_JD

# ======================================================
# PLOT
# ======================================================

x_label_fontsize = 18
y_label_fontsize = 18

plt.figure(figsize=(16, 10))
ax = plt.gca()

ax.errorbar(
    time_shifted,
    Mag,
    yerr=Magerr,
    fmt='o',
    color='green',
    markersize=6,
    label=(f"{chosen_filter}-band data" if 'chosen_filter' in globals() else "Light curve data")
)

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
