# By Jacob Ellerbrook — Multi-filter nova light-curve analysis (stable build)
# Requires: numpy, pandas, matplotlib

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons

# Optional: silence harmless "mean of empty slice" warnings from median on empty groups
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

# ======================================================
# User Inputs (edit these)
# ======================================================
file_path = r"C:\Users\Jmell\Dropbox\Research File\Nova_targets\SMCN_FOLDER\SMCN_2016_10a\SMCN_2016_10a.dat"

# Set your adopted values for the event:
tpeak_JD = 2457682.64139     # <-- EDIT: adopted peak JD
m_peak   = 10.847            # <-- EDIT: adopted peak magnitude
t2_JD    = 2457695.00000     # <-- EDIT: adopted t2 JD (when mag = m_peak + 2)

# Optional override for the "before" uncertainty on t_peak (days); set None to auto-compute
manual_tpeak_before = None   # e.g., 0.35 or None

# Axes ranges
x_lim = (-30, 180)                   # days relative to t_peak on x-axis
y_lim = (m_peak + 3.5, m_peak - 0.8) # inverted mag axis later

# ======================================================
# Load and clean file
# Expect columns (whitespace separated): JD  Filter  Mag  MagErr
# Lines starting with '#' are ignored.
# ======================================================
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Could not find file at: {file_path}")

df = pd.read_csv(
    file_path,
    sep=r'\s+',
    comment='#',
    header=None,
    names=['JD', 'Filter', 'Mag', 'MagErr'],
    engine='python'
)

# Coerce types
df['Filter'] = df['Filter'].astype(str).str.strip()
for col in ['JD', 'Mag', 'MagErr']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

before = len(df)
# Drop rows missing JD or Mag
df = df.dropna(subset=['JD', 'Mag']).copy()

# Remove obviously invalid magnitudes
df = df[df['Mag'] != 99.999].copy()

# Fill missing MagErr with per-filter median, then global median, then 0.0
if df['MagErr'].isna().any():
    df['MagErr'] = df.groupby('Filter')['MagErr'].transform(lambda s: s.fillna(s.median()))
    if df['MagErr'].isna().any():
        gmed = float(df['MagErr'].median()) if pd.notna(df['MagErr'].median()) else 0.0
        df['MagErr'] = df['MagErr'].fillna(gmed)

# Ensure finite, non-negative errors
err = df['MagErr'].to_numpy(copy=True)
err[~np.isfinite(err)] = 0.0
err[err < 0] = 0.0
df['MagErr'] = err

print(f"Loaded {before} rows; kept {len(df)} after cleaning.")
print("NaNs remaining per column:\n", df.isna().sum().to_dict())

# Derived column: days since peak
df['DaysAfterPeak'] = df['JD'] - tpeak_JD

# Filter list
filters = sorted(df['Filter'].unique().tolist())
filters_with_all = ['ALL'] + filters

# ======================================================
# Helper functions
# ======================================================
def compute_brackets_for_tpeak(time_array: np.ndarray, tpeak: float):
    """Return (gap_before, gap_after) around tpeak using nearest observations in time_array (days)."""
    if time_array.size == 0:
        return 0.0, 0.0

    idx = int(np.argmin(np.abs(time_array - tpeak)))
    n = len(time_array)

    if idx == 0 and n > 1:
        gap_before, gap_after = 0.0, time_array[1] - time_array[0]
    elif idx == n - 1 and n > 1:
        gap_before, gap_after = time_array[-1] - time_array[-2], 0.0
    elif n > 2:
        left  = time_array[idx-1] if idx-1 >= 0 else time_array[idx]
        right = time_array[idx+1] if idx+1 < n else time_array[idx]
        gap_before, gap_after = max(0.0, tpeak - left), max(0.0, right - tpeak)
    else:
        gap_before, gap_after = 0.0, 0.0

    if manual_tpeak_before is not None:
        try:
            gap_before = float(manual_tpeak_before)
        except Exception:
            pass
    return gap_before, gap_after


def compute_t2_uncertainties(time_array: np.ndarray, t2jd: float, gap_before_peak: float, gap_after_peak: float):
    """Return (t2_err_before, t2_err_after) combining local t2 bracket with t_peak brackets (quadrature)."""
    if time_array.size == 0:
        return 0.0, 0.0

    idx_t2 = int(np.argmin(np.abs(time_array - t2jd)))
    n = len(time_array)

    if idx_t2 == 0 and n > 1:
        t2_gap_before, t2_gap_after = 0.0, time_array[1] - time_array[0]
    elif idx_t2 == n - 1 and n > 1:
        t2_gap_before, t2_gap_after = time_array[-1] - time_array[-2], 0.0
    elif n > 2:
        left  = time_array[idx_t2-1] if idx_t2-1 >= 0 else time_array[idx_t2]
        right = time_array[idx_t2+1] if idx_t2+1 < n else time_array[idx_t2]
        t2_gap_before, t2_gap_after = max(0.0, t2jd - left), max(0.0, right - t2jd)
    else:
        t2_gap_before, t2_gap_after = 0.0, 0.0

    t2_err_before = np.sqrt(t2_gap_before**2 + gap_before_peak**2)
    t2_err_after  = np.sqrt(t2_gap_after**2  + gap_after_peak**2)
    return t2_err_before, t2_err_after


def format_unc_text(tb: float, ta: float, t2c: float, t2b: float, t2a: float) -> str:
    return (
        f"t_peak uncertainty: -{tb:.3f} / +{ta:.3f} d\n"
        f"t2 @ {t2c:.3f} d  uncertainty: -{t2b:.3f} / +{t2a:.3f} d"
    )


def set_errorbar_visibility(eb, visible: bool):
    """Toggle visibility for a Matplotlib ErrorbarContainer safely."""
    if eb is None:
        return
    # Main line
    try:
        eb[0].set_visible(visible)
    except Exception:
        pass
    # Caps (list/tuple)
    if len(eb) > 1 and eb[1] is not None:
        for cap in eb[1]:
            try:
                cap.set_visible(visible)
            except Exception:
                pass
    # Bars (list/tuple)
    if len(eb) > 2 and eb[2] is not None:
        for bar in eb[2]:
            try:
                bar.set_visible(visible)
            except Exception:
                pass

# ======================================================
# Figure / UI
# ======================================================
mpl.rcParams.update({'font.size': 13})
fig, ax = plt.subplots(figsize=(16, 9))
plt.subplots_adjust(left=0.22, right=0.97, bottom=0.1, top=0.92)

# RadioButtons (left panel) for selecting filter or ALL
rax = plt.axes([0.03, 0.35, 0.15, 0.5])
radio = RadioButtons(rax, labels=filters_with_all, active=0)

# Plot all filters initially (we'll toggle visibility below)
errorbar_artists = {}
for f in filters:
    sub = df[df['Filter'] == f]
    eb = ax.errorbar(
        sub['DaysAfterPeak'].values,
        sub['Mag'].values,
        yerr=sub['MagErr'].values,
        fmt='o', label=f"{f}-band", alpha=0.85, markersize=6
    )
    errorbar_artists[f] = eb

# Horizontal lines: m_peak and m_peak + 2
ax.plot([-1e6, 1e6], [m_peak, m_peak], linestyle='--', color='k', label='m_peak')
ax.plot([-1e6, 1e6], [m_peak + 2.0, m_peak + 2.0], linestyle='--', color='k', label='m_peak + 2')

# Vertical markers for t_peak (x=0) and t2 (x = t2 - t_peak)
v_tpeak = ax.axvline(0.0, linestyle='--', color='k', label='t_peak = 0')
t2_center_days = t2_JD - tpeak_JD
v_t2 = ax.axvline(t2_center_days, linestyle='--', color='k', label=f"t2 = {t2_center_days:.2f} d")

# Initialize shaded uncertainty spans (created on first update)
span_tpeak = None
span_t2 = None

# Text box for uncertainty summary
txt = ax.text(
    0.02, 0.02, "", transform=ax.transAxes, fontsize=12, va='bottom', ha='left',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
)

# Axes styling
ax.invert_yaxis()
ax.set_xlim(*x_lim)
ax.set_ylim(*y_lim)
ax.set_xlabel("Days Since Peak (days)")
ax.set_ylabel("Optical Brightness (mag)")
title_name = os.path.splitext(os.path.basename(file_path))[0].replace('_', '-')
ax.set_title(f"{title_name} — multi-filter analysis")
ax.legend(loc='upper right', fontsize=11)

# ======================================================
# Update routine
# ======================================================
def set_filter_visibility(selected: str):
    global span_tpeak, span_t2

    # Toggle visibility
    if selected == 'ALL':
        active_filters = filters[:]
        for f in filters:
            set_errorbar_visibility(errorbar_artists[f], True)
    else:
        active_filters = [selected]
        for f in filters:
            set_errorbar_visibility(errorbar_artists[f], f == selected)

    # Recompute uncertainties only on the active subset (or all)
    sub = df[df['Filter'].isin(active_filters)].copy()
    times = np.sort(sub['JD'].values)

    gap_before, gap_after = compute_brackets_for_tpeak(times, tpeak_JD)
    t2_before, t2_after   = compute_t2_uncertainties(times, t2_JD, gap_before, gap_after)

    # Remove and recreate shaded spans (robust across Matplotlib versions)
    if span_tpeak is not None:
        try:
            span_tpeak.remove()
        except Exception:
            pass
    span_tpeak = ax.axvspan(-gap_before, +gap_after, color='red', alpha=0.22)

    t2_center_days = t2_JD - tpeak_JD
    if span_t2 is not None:
        try:
            span_t2.remove()
        except Exception:
            pass
    span_t2 = ax.axvspan(t2_center_days - t2_before, t2_center_days + t2_after,
                         color='cyan', alpha=0.22)

    # Keep v_t2 in sync (value doesn't change; included for completeness)
    v_t2.set_xdata([t2_center_days, t2_center_days])

    # Update info text
    txt.set_text(format_unc_text(gap_before, gap_after, t2_center_days, t2_before, t2_after))

    # Console log
    print("-------------------------------------------------")
    print(f"Active filters: {active_filters}")
    print(f"t_peak uncertainty: -{gap_before:.5f} / +{gap_after:.5f} days")
    print(f"t2 (at {t2_center_days:.5f} d) uncertainty: -{t2_before:.5f} / +{t2_after:.5f} days")

    fig.canvas.draw_idle()

# Initialize with ALL
set_filter_visibility('ALL')

def on_radio_clicked(label: str):
    set_filter_visibility(label)

radio.on_clicked(on_radio_clicked)

plt.tight_layout()
plt.show()
