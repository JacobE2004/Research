import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# ======================================================
# CONFIG (edit these)
# ======================================================
file_path = r"C:\Users\Jmell\Dropbox\Research File\Nova_targets\SMCN_FOLDER\SMCN_2016_10a\SMCN_2016_10a_ASASSN_light-curves.csv"

# Observed peak (JD, mag)
tpeak_JD = 2458094.1447
m_peak   = 11.82

# t2 specified manually by JD (the exact epoch when the light curve is 2 mag below peak)
t2_JD = 2458203.6254   # <-- PUT YOUR MEASURED t2 JD HERE

# Band selection for headered AAVSO files: set to "I", ["I","CV"], or None to keep all
target_bands = "I"

# Plotting
plot_window_days = 400.0
connect_style = "o"   # "o", "o-", "o--"
marker_color = "green"

# ============================================Hey h==========
# LOADERS
# ======================================================
def load_lightcurve(path, target_bands="I"):
    """
    Robust loader:
      1) Try headered AAVSO-style CSV/TXT (expects columns JD/Magnitude/Uncertainty and optional Band)
      2) Fallback to plain numeric 3-col file: JD, Mag, Magerr
    Optionally filter by band(s) if 'Band' exists.
    Returns sorted arrays: time, Mag, Magerr, band_labels (or None)
    """
    # First try: headered
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        # normalize column names
        cols = {c.strip(): c for c in df.columns}
        required = ["JD", "Magnitude", "Uncertainty"]
        if all(c in cols for c in required):
            df_work = df.copy()

            # If Band exists and user requested filter
            band_labels = None
            if "Band" in cols:
                band_labels = df_work["Band"].astype(str).str.upper().str.strip()
                if target_bands is not None:
                    if isinstance(target_bands, str):
                        keep = {target_bands.upper()}
                    else:
                        keep = {b.upper() for b in target_bands}
                    df_work = df_work[band_labels.str.upper().isin(keep)]
                    band_labels = df_work["Band"].astype(str).str.upper().str.strip()

            # Coerce numerics and drop bad rows
            df_work["JD"] = pd.to_numeric(df_work[cols["JD"]], errors="coerce")
            df_work["Magnitude"] = pd.to_numeric(df_work[cols["Magnitude"]], errors="coerce")
            df_work["Uncertainty"] = pd.to_numeric(df_work[cols["Uncertainty"]], errors="coerce")
            df_work = df_work.dropna(subset=["JD", "Magnitude", "Uncertainty"])

            if df_work.empty:
                raise ValueError("No valid rows after filtering/coercion.")

            # Sort by time
            df_work = df_work.sort_values("JD")
            time   = df_work["JD"].to_numpy(float)
            Mag    = df_work["Magnitude"].to_numpy(float)
            Magerr = df_work["Uncertainty"].to_numpy(float)
            band_labels = df_work["Band"].astype(str).str.upper().to_numpy() if "Band" in df_work.columns else None
            print(f"[loader] Loaded {len(time)} points (headered).")
            return time, Mag, Magerr, band_labels
        else:
            raise ValueError("Headered columns not found.")
    except Exception:
        # Fallback: plain numeric 3 columns
        arr = np.loadtxt(path)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError("Numeric fallback expects at least 3 columns: JD, Mag, Magerr.")
        time   = arr[:, 0].astype(float)
        Mag    = arr[:, 1].astype(float)
        Magerr = arr[:, 2].astype(float)
        order = np.argsort(time)
        time, Mag, Magerr = time[order], Mag[order], Magerr[order]
        print(f"[loader] Loaded {len(time)} points (numeric fallback).")
        return time, Mag, Magerr, None

# ======================================================
# UNCERTAINTY HELPERS
# ======================================================
def bracket_peak_gaps(time, tpeak_JD):
    """
    Find the nearest observed datum to t_peak and bracket t_peak by the
    adjacent observations. Matches the logic in your second snippet.
    """
    idx = np.argmin(np.abs(time - tpeak_JD))
    if idx == 0:
        gap_before = 0.0
        gap_after  = time[1] - time[0] if len(time) > 1 else 0.0
    elif idx == len(time) - 1:
        gap_before = time[-1] - time[-2] if len(time) > 1 else 0.0
        gap_after  = 0.0
    else:
        gap_before = time[idx] - time[idx - 1]
        gap_after  = time[idx + 1] - time[idx]
    return gap_before, gap_after

def bracket_manual_epoch(time, t_epoch):
    """
    Given any manual epoch (e.g., t2_JD), return the time distances to the
    nearest observation before and after that epoch.
    """
    # index of last time <= t_epoch
    i_lo = np.searchsorted(time, t_epoch, side="right") - 1
    i_hi = i_lo + 1

    if i_lo < 0:
        # epoch is before first point
        before = 0.0
        after  = time[0] - t_epoch
    elif i_hi >= len(time):
        # epoch is after last point
        before = t_epoch - time[-1]
        after  = 0.0
    else:
        before = t_epoch - time[i_lo]
        after  = time[i_hi] - t_epoch
    return before, after

# ======================================================
# MAIN
# ======================================================
time, Mag, Magerr, _ = load_lightcurve(file_path, target_bands=target_bands)
time_shifted = time - tpeak_JD

# t_peak bracket (same rules as your second snippet)
gap_before_peak, gap_after_peak = bracket_peak_gaps(time, tpeak_JD)

# Manual t2: bracket by nearest observations around your provided t2_JD
t2_bracket_before, t2_bracket_after = bracket_manual_epoch(time, t2_JD)

# Asymmetric t2 uncertainties using quadrature (match second snippet intent)
t2_err_before = np.sqrt(t2_bracket_before**2 + gap_before_peak**2)
t2_err_after  = np.sqrt(t2_bracket_after**2  + gap_after_peak**2)

# Central t2 in "days since peak"
t2_central = t2_JD - tpeak_JD
m_t2 = m_peak + 2.0

# ======================================================
# PLOT
# ======================================================
mask = (time_shifted >= -plot_window_days) & (time_shifted <= plot_window_days)

plt.figure(figsize=(16, 10))
ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_useOffset(False)
plt.gca().invert_yaxis()

# data + error bars
ax.errorbar(
    time_shifted[mask], Mag[mask], yerr=Magerr[mask],
    fmt=connect_style, color=marker_color, markersize=7, alpha=0.9,
    label="Photometry"
)

# reference lines
plt.axhline(m_peak, color='b', linestyle='--', label=f"m_peak = {m_peak:.3f}")
plt.axhline(m_t2,   color='r', linestyle='--', label=f"m_peak+2 = {m_t2:.3f}")
plt.axvline(0, color='k', linestyle='--', label="t_peak = 0")
plt.axvline(t2_central, color='purple', linestyle='--', label=f"t2 = {t2_central:.2f} d")

# show t_peak horizontal error (as vertical band) using axvspan
plt.axvspan(-gap_before_peak, gap_after_peak, color='red', alpha=0.25)

# show t2 asymmetric bracket region (cyan)
plt.axvspan(t2_central - t2_bracket_before, t2_central + t2_bracket_after, color='cyan', alpha=0.25)

# Axis formatting
ax.set_yticks(np.arange(0, 30, 1))
ax.set_yticks(np.arange(0, 30, 0.5), minor=True)

xmax = plot_window_days
ax.set_xticks(np.arange(-xmax, xmax + 1e-9, 20))
ax.set_xticks(np.arange(-xmax, xmax + 1e-9, 10), minor=True)

plt.ylim(23, 10)
plt.xlim(-plot_window_days, plot_window_days)
plt.xlabel('Days since peak brightness', fontsize=20)
plt.ylabel('Optical Brightness (mag)', fontsize=20)
plt.title(f"{os.path.basename(file_path)}", fontsize=20)
mpl.rcParams.update({'font.size': 16})
plt.legend(loc='upper right', fontsize=14)
plt.tight_layout()
plt.show()

# ======================================================
# PRINT OUT UNCERTAINTIES
# ======================================================
print("---- Asymmetric uncertainties (quadrature for t2) ----")
print(f"t_peak uncertainty: -{gap_before_peak:.5f} / +{gap_after_peak:.5f} days")
print(f"t2 uncertainty:     -{t2_err_before:.5f} / +{t2_err_after:.5f} days")
print(f"(Plotted t2 bracket used: -{t2_bracket_before:.5f} / +{t2_bracket_after:.5f} days)")
print(f"t2 (days since peak): {t2_central:.5f}")
print(f"t2 JD provided:      {t2_JD:.5f}")
