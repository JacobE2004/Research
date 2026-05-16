# By Jacob Ellerbook

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import csv
import io


# Load Light Curve

file_path = r"C:\Users\Jmell\Dropbox\Research File\Nova_targets\LMCN_Folder\LMCN_2020_11a_notanerruption\aavsodata_6a037713b757b.txt"

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
    - Space-delimited files (like LMCN .dat files)
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

    # Try to detect delimiter
    delim = None
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
            # Default to space-delimited if no other delimiter found
            delim = None

    # Parse rows based on detected delimiter
    all_rows = []
    if delim is not None:
        # Use csv reader for known delimiters
        reader = csv.reader(io.StringIO("\n".join(lines)), delimiter=delim)
        all_rows = list(reader)
    else:
        # Space-delimited parsing (handles multiple spaces)
        for line in lines:
            row = line.split()
            all_rows.append(row)

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
        ["mag", "Magnitude", "magnitude", "Mag", "mag.", "mag_value", "averagemag"],
        required=True
    )
    err_idx = _find_col(
        ["mag_err", "Uncertainty", "uncertainty", "HQuncertainty", "+/-",
         "Mag Error", "magerror", "MagErr", "error", "err"],
        required=False,
        default=None
    )

    rows = []
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
raw_filter = np.array([str(x).strip() for x in raw[:, 1]])
raw_mag  = np.array([str(x).strip().lstrip("><") for x in raw[:, 2]])
raw_err  = np.array([str(x).strip().lstrip("><") for x in raw[:, 3]]) if raw.shape[1] > 3 else np.full_like(raw_mag, "0.05")

# Keep only rows where time and mag are numeric
mask_numeric = np.array([_is_float(t) and _is_float(m) for t, m in zip(raw_time, raw_mag)])

raw_time = raw_time[mask_numeric]
raw_filter = raw_filter[mask_numeric]
raw_mag = raw_mag[mask_numeric]
raw_err = raw_err[mask_numeric]

time = raw_time.astype(float)
Mag = raw_mag.astype(float)
Magerr = np.array([float(x) if _is_float(x) else 0.05 for x in raw_err], dtype=float)
filter_array = raw_filter  # Keep track of filters

if Mag.size > 0 and np.all(np.isnan(Mag)):
    print("Warning: all magnitude entries are NaN (file may contain only 'null' values).")

if time.size == 0:
    raise ValueError("No numeric time/magnitude data found in the file after parsing.")

# ======================================================
# FILTER SELECTION - Handle multi-filter files
# ======================================================
unique_filters = np.unique(filter_array)
print(f"\nAvailable filters: {unique_filters}")
if len(unique_filters) > 1:
    selected_filter = input(f"Select filter to plot (options: {', '.join(unique_filters)}): ").strip().upper()
    if selected_filter not in unique_filters:
        print(f"Warning: '{selected_filter}' not found. Using '{unique_filters[0]}'")
        selected_filter = unique_filters[0]
else:
    selected_filter = unique_filters[0]

# Filter data to selected filter only
filter_mask = (filter_array == selected_filter)
time = time[filter_mask]
Mag = Mag[filter_mask]
Magerr = Magerr[filter_mask]
filter_array = filter_array[filter_mask]

print(f"Using filter: {selected_filter}")
print(f"Data points for {selected_filter}: {len(time)}")

# Remove bad data points
good = (Mag != 99.990)
time = time[good]
Mag = Mag[good]
Magerr = Magerr[good]
filter_array = filter_array[good]

# Separate valid and upper limit points
valid = ~np.isclose(Magerr, 99.990, atol=1e-6)
upper_limit = np.isclose(Magerr, 99.990, atol=1e-6)

# User-set values
tpeak_JD = 2459171.670457 
m_peak = 9.936
t2_JD = 8197.10268

# Shift for plotting
time_shifted = time - tpeak_JD

# ============================
# Manual overrides for t_peak bracket
# ============================
manual_tpeak_prev = 2460524.62531
manual_tpeak_next = 2460528.67884

# Manual t2 bracket
manual_t2_prev = 2460524.62531
manual_t2_next = 2460528.67884

# Calculate t2 time shift for plotting
t2_central = t2_JD - tpeak_JD

#Toggle for connecting data points with a line
connect_line = input("Connect data points with a line? (y/n): ").strip().lower() == 'y'


# PLOT


x_label_fontsize = 20
y_label_fontsize = 20

plt.figure(figsize=(16, 9))
ax = plt.gca()

ax.plot(
    time_shifted[valid],
    Mag[valid],
    '-o' if connect_line else 'o',
    color='green',
    markersize=6,
    label="Valid data",
)

if np.any(upper_limit):
    ax.scatter(
        time_shifted[upper_limit],
        Mag[upper_limit],
        marker='v',
        color='black',
        s=36,
        label="Upper limits",
    )


plt.gca().invert_yaxis()



# Add peak magnitude value to legend without plotting line
ax.plot([], [], ' ', label=f"m_peak = {m_peak:.3f}")

plt.ylim(23, 10)
plt.xlim(-60, 120)

ax.set_xlabel("Days since peak brightness", fontsize=x_label_fontsize)
ax.set_ylabel("Optical Brightness (mag)", fontsize=y_label_fontsize)
ax.set_title("LMCN-2020-11a", fontsize=20)

plt.legend(fontsize=13, loc='upper right')
plt.grid(False)
plt.tight_layout()
plt.show()