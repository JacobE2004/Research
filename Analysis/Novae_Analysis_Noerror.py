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
file_path = r"C:\Users\Jmell\Dropbox\Research File\Nova_targets\LMCN_Folder\LMCN_2016_04a __\LMCN_2016_04a_OGLE.dat"
# If running on WSL, convert Windows path to WSL path automatically
if os.name == "posix" and file_path.startswith("C:"):
    file_path = file_path.replace("C:\\", "/mnt/c/").replace("\\", "/")
# If running on Windows, convert WSL path to Windows path
elif os.name == "nt" and file_path.startswith("/mnt/c/"):
    file_path = file_path.replace("/mnt/c/", "C:\\").replace("/", "\\")
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

        # Detect if this is an upper or lower limit before stripping markers
        is_limit = False
        if m.startswith(">") or m.startswith("<"):
            is_limit = True
        
        m = m.lstrip("><")
        me = me.lstrip("><")

        if m.lower() in ["null", "untrusted", ""]:
            m = "nan"

        if me.lower() in ["null", "none", "nan", ""]:
            me = "0.05"

        rows.append([t, filt, m, me, str(is_limit)])

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
# Format X-axis function
# ======================================================
def format_xaxis_days(ax, major_interval=100, minor_interval=25):
    """
    Format x-axis with major ticks every 'major_interval' days and 
    minor ticks every 'minor_interval' days. Removes grid lines.
    
    Parameters:
    -----------
    ax : matplotlib axis object
        The axis to format
    major_interval : int
        Spacing between major tick marks in days (default: 10)
    minor_interval : int
        Spacing between minor tick marks in days (default: 5)
    """
    from matplotlib.ticker import MultipleLocator
    
    ax.xaxis.set_major_locator(MultipleLocator(major_interval))
    ax.xaxis.set_minor_locator(MultipleLocator(minor_interval))
    ax.grid(False)

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
raw_is_limit = np.array([x == 'True' for x in raw[:, 4]]) if raw.shape[1] > 4 else np.full(len(raw_mag), False)

# Keep only rows where time and mag are numeric
mask_numeric = np.array([_is_float(t) and _is_float(m) for t, m in zip(raw_time, raw_mag)])

raw_time = raw_time[mask_numeric]
raw_mag = raw_mag[mask_numeric]
raw_err = raw_err[mask_numeric]
raw_is_limit = raw_is_limit[mask_numeric]

time = raw_time.astype(float)
Mag = raw_mag.astype(float)
Magerr = np.array([float(x) if _is_float(x) else 0.05 for x in raw_err], dtype=float)
is_limit = raw_is_limit.copy()

if Mag.size > 0 and np.all(np.isnan(Mag)):
    print("Warning: all magnitude entries are NaN (file may contain only 'null' values).")

if time.size == 0:
    raise ValueError("No numeric time/magnitude data found in the file after parsing.")

# Remove bad data points: defined as finite values in time and magnitude
good = np.isfinite(time) & np.isfinite(Mag)

if not np.any(good):
    raise ValueError("No valid numeric data after cleaning NaN/Inf values.")

time = time[good]
Mag = Mag[good]
Magerr = Magerr[good]
is_limit = is_limit[good]

# User-set values
tpeak_JD = 2457637.89406
m_peak = 16.662
t2_JD = 2456015.870

# Shift for plotting
time_shifted = time - tpeak_JD

# ============================
# Manual overrides for t_peak bracket
# ============================
manual_tpeak_prev = 2456011.62340
manual_tpeak_next = 2456014.55166

# Manual t2 bracket
manual_t2_prev = 2456015.58192
manual_t2_next = 2456016.58763

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

# Extract filter column for multi-filter analysis
raw_filter = np.array([str(x).strip() for x in raw[mask_numeric, 1]])
unique_filters = np.unique(raw_filter[raw_filter != 'NA'])

# Default: do not connect data points with a line
connect_line = False

# Default: use all filters for analysis
if len(unique_filters) > 0:
    print(f"Available filters: {unique_filters}")
selected_filter = 'all'

if selected_filter.lower() == 'all':
    # Use all data
    time_to_plot = time_shifted
    mag_to_plot = Mag
    err_to_plot = Magerr
    is_limit_to_plot = is_limit
    plot_title = "LMCN-2016-04a"
else:
    # Filter to specific filter
    filt_mask = raw_filter == selected_filter
    time_to_plot = time_shifted[filt_mask]
    mag_to_plot = Mag[filt_mask]
    err_to_plot = Magerr[filt_mask]
    is_limit_to_plot = is_limit[filt_mask]
    plot_title = f"LMCN-2019-11a - Filter {selected_filter}"

# ======================================================
# PLOT
# ======================================================

x_label_fontsize = 20
y_label_fontsize = 20

plt.figure(figsize=(16, 9))
ax = plt.gca()

# Separate regular detections from upper/lower limits
detection_mask = ~is_limit_to_plot
limit_mask = is_limit_to_plot

# Plot the regular data points
if connect_line:
    if np.any(detection_mask):
        plt.plot(time_to_plot[detection_mask], mag_to_plot[detection_mask], 'o-', color='blue', label='Magnitude Data')
    if np.any(limit_mask):
        plt.plot(time_to_plot[limit_mask], mag_to_plot[limit_mask], 'v', color='black', markersize=8, label='Upper Limits')
else:
    if np.any(detection_mask):
        plt.errorbar(time_to_plot[detection_mask], mag_to_plot[detection_mask], yerr=err_to_plot[detection_mask], fmt='o', color='blue', label='Magnitude Data')
    if np.any(limit_mask):
        plt.plot(time_to_plot[limit_mask], mag_to_plot[limit_mask], 'v', color='black', markersize=8, label='Upper Limits')

# Add peak magnitude to legend (no line on plot)
plt.plot([], [], ' ', label=f'Peak Magnitude: {m_peak}')

plt.gca().invert_yaxis()
plt.ylim(23, 10)
plt.xlim(-60, 120)

ax.set_ylabel("Optical Brightness (mag)", fontsize=y_label_fontsize)
ax.set_xlabel("Time (days since peak)", fontsize=x_label_fontsize)
ax.set_title(plot_title, fontsize=20)

plt.legend(fontsize=13, loc='upper right')
format_xaxis_days(ax, major_interval=20, minor_interval=10)
plt.tight_layout()
plt.show()



