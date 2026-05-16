import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Compare your AstroImageJ W UMa light curve with AAVSO data
# ============================================================

# -----------------------------
# File names
# -----------------------------

my_file = r"C:\Users\Jmell\Dropbox\Research File\Table.tbl"
aavso_file = r"C:\Users\Jmell\Downloads\observations_20260424_181640\observations_20260424_181640.csv"

# -----------------------------
# W UMa known period
# -----------------------------

WUMA_PERIOD_DAYS = 0.3336352

# For your data, using your first observation as phase zero is okay
# for comparison of shape.
# If you want literature phase, use WUMA_T0 instead.
WUMA_T0 = None

# Literature epoch, optional
WUMA_T0_LIT = 2451952.34017

# -----------------------------
# Your AstroImageJ columns
# -----------------------------

my_time_col = "BJD_TDB"
my_mag_col = "Source_AMag_T1"
my_mag_err_col = "Source_AMag_Err_T1"

# -----------------------------
# AAVSO columns
# -----------------------------

aavso_time_col = "jd"
aavso_mag_col = "mag"
aavso_err_col = "uncertainty"
aavso_band_col = "band"

# Visual AAVSO estimates often do not include formal uncertainties.
# This assumed error is used only for plotting error bars.
AAVSO_VISUAL_ERROR = 0.04

# ============================================================
# Helper functions
# ============================================================

def phase_fold(time, period, t0):
    """
    Convert time to orbital phase.
    """
    return ((time - t0) / period) % 1.0


def normalize_magnitude(mag):
    """
    Normalize magnitude by subtracting the median.
    
    This keeps the shape of the light curve but removes the absolute
    magnitude offset, making two datasets easier to compare.
    """
    return mag - np.nanmedian(mag)


def sort_by_phase(phase, mag, err=None):
    """
    Sort arrays by orbital phase for cleaner plotting.
    """
    order = np.argsort(phase)

    if err is None:
        return phase[order], mag[order]
    else:
        return phase[order], mag[order], err[order]


# ============================================================
# Load your AstroImageJ data
# ============================================================

my_df = pd.read_csv(my_file, sep="\t")
my_df = my_df.replace([np.inf, -np.inf], np.nan)
my_df = my_df.dropna(subset=[my_time_col, my_mag_col, my_mag_err_col])

my_time = my_df[my_time_col].to_numpy(dtype=float)
my_mag = my_df[my_mag_col].to_numpy(dtype=float)
my_err = my_df[my_mag_err_col].to_numpy(dtype=float)

# Sort by time
order = np.argsort(my_time)
my_time = my_time[order]
my_mag = my_mag[order]
my_err = my_err[order]

# Use your first observation as phase zero unless a literature T0 is chosen
if WUMA_T0 is None:
    my_t0 = my_time[0]
else:
    my_t0 = WUMA_T0

my_phase = phase_fold(my_time, WUMA_PERIOD_DAYS, my_t0)

# Normalize your magnitude
my_mag_norm = normalize_magnitude(my_mag)

my_phase, my_mag_norm, my_err = sort_by_phase(
    my_phase,
    my_mag_norm,
    my_err
)

# ============================================================
# Load AAVSO data
# ============================================================

aavso_df = pd.read_csv(aavso_file)
aavso_df = aavso_df.replace([np.inf, -np.inf], np.nan)

# Keep only real detections, not "fainter than" observations
if "fainterthan" in aavso_df.columns:
    aavso_df = aavso_df[aavso_df["fainterthan"] == False]

# Drop missing magnitudes/times
aavso_df = aavso_df.dropna(subset=[aavso_time_col, aavso_mag_col])

aavso_time = aavso_df[aavso_time_col].to_numpy(dtype=float)
aavso_mag = aavso_df[aavso_mag_col].to_numpy(dtype=float)

# If AAVSO uncertainty column is empty, use assumed visual error
if aavso_err_col in aavso_df.columns:
    aavso_err = aavso_df[aavso_err_col].to_numpy(dtype=float)

    # Replace NaN uncertainties with assumed visual uncertainty
    aavso_err = np.where(
        np.isfinite(aavso_err),
        aavso_err,
        AAVSO_VISUAL_ERROR
    )
else:
    aavso_err = np.full_like(aavso_mag, AAVSO_VISUAL_ERROR)

# For AAVSO, fold using the same phase zero as your data for shape comparison.
# This aligns the overall phase convention to your observing run.
aavso_phase = phase_fold(aavso_time, WUMA_PERIOD_DAYS, my_t0)

# Normalize AAVSO magnitudes
aavso_mag_norm = normalize_magnitude(aavso_mag)

aavso_phase, aavso_mag_norm, aavso_err = sort_by_phase(
    aavso_phase,
    aavso_mag_norm,
    aavso_err
)

# ============================================================
# Print summaries
# ============================================================

print("\nDATA SUMMARY")
print("------------")
print(f" data points: {len(my_mag_norm)}")
print(f"AAVSO data points: {len(aavso_mag_norm)}")
print(f" time range: {np.min(my_time):.5f} to {np.max(my_time):.5f}")
print(f"AAVSO time range: {np.min(aavso_time):.5f} to {np.max(aavso_time):.5f}")
print(f"Period used: {WUMA_PERIOD_DAYS:.7f} days")
print(f"AAVSO bands included: {sorted(aavso_df[aavso_band_col].dropna().unique())}")

print("\nNORMALIZATION")
print("-------------")
print("Both datasets were normalized by subtracting their own median magnitude.")
print("So the y-axis is relative magnitude, not absolute magnitude.")


# ============================================================
# Plot 1: Your original light curve
# ============================================================

plt.figure(figsize=(10, 5))

plt.errorbar(
    my_time - my_time[0],
    my_mag - np.nanmedian(my_mag),
    yerr=my_err,
    fmt="o",
    markersize=4,
    capsize=2,
    label=" AstroImageJ data"
)

plt.gca().invert_yaxis()
plt.xlabel("Time Since First Observation [days]")
plt.ylabel("Relative Magnitude")
plt.title(" W UMa Light Curve")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# Plot 2: AAVSO original light curve
# ============================================================

plt.figure(figsize=(10, 5))

plt.errorbar(
    aavso_time - np.min(aavso_time),
    aavso_mag_norm,
    yerr=aavso_err,
    fmt="o",
    markersize=4,
    capsize=2,
    label="AAVSO Visual data"
)

plt.gca().invert_yaxis()
plt.xlabel("Time Since First AAVSO Observation [days]")
plt.ylabel("Relative Magnitude")
plt.title("AAVSO W UMa Light Curve")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# Plot 3: Phase-folded comparison
# ============================================================

plt.figure(figsize=(11, 6))

# Your data
plt.errorbar(
    my_phase,
    my_mag_norm,
    yerr=my_err,
    fmt="o",
    markersize=5,
    capsize=2,
    label=" AstroImageJ data"
)

plt.errorbar(
    my_phase + 1.0,
    my_mag_norm,
    yerr=my_err,
    fmt="o",
    markersize=5,
    capsize=2
)

# AAVSO data
plt.errorbar(
    aavso_phase,
    aavso_mag_norm,
    yerr=aavso_err,
    fmt=".",
    markersize=8,
    alpha=0.6,
    capsize=2,
    label=f"AAVSO Visual data, assumed error = {AAVSO_VISUAL_ERROR:.2f} mag"
)

plt.errorbar(
    aavso_phase + 1.0,
    aavso_mag_norm,
    yerr=aavso_err,
    fmt=".",
    markersize=8,
    alpha=0.6,
    capsize=2
)

plt.gca().invert_yaxis()
plt.xlabel("Orbital Phase")
plt.ylabel("Relative Magnitude")
plt.title("Phase-Folded W UMa Light Curve:  Data vs. AAVSO")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# Plot 4: Binned AAVSO curve + your data
# ============================================================

# Bin AAVSO data by phase so it is easier to compare visually
bins = np.linspace(0, 1, 31)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

aavso_bin_mag = []
aavso_bin_err = []

for left, right in zip(bins[:-1], bins[1:]):
    mask = (aavso_phase >= left) & (aavso_phase < right)

    if np.sum(mask) > 0:
        aavso_bin_mag.append(np.nanmedian(aavso_mag_norm[mask]))
        aavso_bin_err.append(np.nanstd(aavso_mag_norm[mask]))
    else:
        aavso_bin_mag.append(np.nan)
        aavso_bin_err.append(np.nan)

aavso_bin_mag = np.array(aavso_bin_mag)
aavso_bin_err = np.array(aavso_bin_err)

plt.figure(figsize=(11, 6))

plt.errorbar(
    my_phase,
    my_mag_norm,
    yerr=my_err,
    fmt="o",
    markersize=5,
    capsize=2,
    label=" AstroImageJ data"
)

plt.errorbar(
    my_phase + 1.0,
    my_mag_norm,
    yerr=my_err,
    fmt="o",
    markersize=5,
    capsize=2
)

plt.errorbar(
    bin_centers,
    aavso_bin_mag,
    yerr=aavso_bin_err,
    fmt="s-",
    markersize=5,
    capsize=3,
    label="AAVSO Visual data, phase-binned"
)

plt.errorbar(
    bin_centers + 1.0,
    aavso_bin_mag,
    yerr=aavso_bin_err,
    fmt="s-",
    markersize=5,
    capsize=3
)

plt.gca().invert_yaxis()
plt.xlabel("Orbital Phase")
plt.ylabel("Relative Magnitude")
plt.title("Phase-Folded Comparison with Binned AAVSO Data")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()