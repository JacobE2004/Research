import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lombscargle

# ============================================================
# W Ursae Majoris Photometry Analysis
# Period Estimate + Phase Folding + Toy Inclination Fit
# ============================================================
#
# IMPORTANT:
# This code uses your photometry to estimate period and inclination
# in a simplified/toy-model way.
#
# This is NOT a full physical W UMa model.
# A true contact-binary solution should use PHOEBE or Wilson-Devinney.
#
# ============================================================


# ============================================================
# 1. USER SETTINGS
# ============================================================

file_path = r"C:\Users\Jmell\Dropbox\Research File\Table.tbl"

# Change this to "JD_UTC" if BJD_TDB is not in your file
time_col = "BJD_TDB"

mag_col = "Source_AMag_T1"
mag_err_col = "Source_AMag_Err_T1"

# Literature values for W Ursae Majoris
# These are used as comparisons and to set the toy model scale.
WUMA_PERIOD_LIT = 0.3336352      # days
WUMA_T0_LIT = 2451952.34017      # HJD, used only as a reference epoch
WUMA_INCLINATION_LIT = 88.4      # degrees

# Approximate published system values
# Used only to set the toy model geometry.
M1_LIT = 1.139   # solar masses
M2_LIT = 0.551   # solar masses
R1_LIT = 1.092   # solar radii
R2_LIT = 0.792   # solar radii
T1_LIT = 6450    # Kelvin
T2_LIT = 6170    # Kelvin

# W UMa has possible third light in detailed studies.
# You can set this to 0.0 if you do not want to include third light.
THIRD_LIGHT_FRACTION = 0.09

# Prevent tiny formal photometric errors from making chi-square unrealistic
ERROR_FLOOR_MAG = 0.01

# Period search range, in days
PERIOD_MIN = 0.15
PERIOD_MAX = 0.70
N_PERIOD_GRID = 20000

# Inclination search range, in degrees
INCLINATION_MIN = 55.0
INCLINATION_MAX = 90.0
N_INCLINATION_GRID = 71

# Phase shift search range
PHASE_SHIFT_MIN = -0.5
PHASE_SHIFT_MAX = 0.5
N_PHASE_SHIFT_GRID = 201

# Monte Carlo settings
N_MONTE_CARLO = 300

# Set random seed so results are repeatable
RANDOM_SEED = 42


# ============================================================
# 2. CONSTANTS
# ============================================================

G = 6.67430e-11
M_SUN = 1.98847e30
R_SUN = 6.957e8


# ============================================================
# 3. LOAD AND CLEAN DATA
# ============================================================

df = pd.read_csv(file_path, sep="\t")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=[time_col, mag_col, mag_err_col])

time = df[time_col].to_numpy(dtype=float)
mag = df[mag_col].to_numpy(dtype=float)
mag_err = df[mag_err_col].to_numpy(dtype=float)

order = np.argsort(time)
time = time[order]
mag = mag[order]
mag_err = mag_err[order]

sigma = np.maximum(mag_err, ERROR_FLOOR_MAG)

time_since_start = time - time[0]
baseline_days = time[-1] - time[0]
baseline_hours = baseline_days * 24

print("\nDATA SUMMARY")
print("------------")
print(f"Number of points: {len(time)}")
print(f"Time baseline: {baseline_days:.6f} days")
print(f"Time baseline: {baseline_hours:.2f} hours")
print(f"Magnitude range: {np.min(mag):.4f} to {np.max(mag):.4f}")


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def phase_coverage_fraction(phases):
    """
    Estimate fraction of full orbital phase covered by data.
    """
    p = np.sort(phases % 1.0)
    gaps = np.diff(np.r_[p, p[0] + 1.0])
    return 1.0 - np.max(gaps)


def estimate_period_lomb_scargle(time_days, magnitude,
                                  min_period=PERIOD_MIN,
                                  max_period=PERIOD_MAX,
                                  n_periods=N_PERIOD_GRID):
    """
    Estimate period using a Lomb-Scargle periodogram.
    """
    t = time_days - np.min(time_days)
    y = magnitude - np.mean(magnitude)

    periods = np.linspace(min_period, max_period, n_periods)
    angular_freqs = 2 * np.pi / periods

    power = lombscargle(t, y, angular_freqs, normalize=True)

    best_idx = np.argmax(power)
    best_period = periods[best_idx]
    best_power = power[best_idx]

    return best_period, best_power, periods, power


def circle_overlap_area(d, r_a, r_b):
    """
    Area of overlap between two circles of radii r_a and r_b,
    separated by projected distance d.
    """
    d = np.asarray(d, dtype=float)
    area = np.zeros_like(d)

    no_overlap = d >= (r_a + r_b)
    full_overlap = d <= abs(r_a - r_b)

    area[full_overlap] = np.pi * min(r_a, r_b) ** 2

    partial = (~no_overlap) & (~full_overlap)
    dp = np.maximum(d[partial], 1e-12)

    arg1 = (dp**2 + r_a**2 - r_b**2) / (2.0 * dp * r_a)
    arg2 = (dp**2 + r_b**2 - r_a**2) / (2.0 * dp * r_b)

    arg1 = np.clip(arg1, -1.0, 1.0)
    arg2 = np.clip(arg2, -1.0, 1.0)

    root_term = (
        (-dp + r_a + r_b)
        * (dp + r_a - r_b)
        * (dp - r_a + r_b)
        * (dp + r_a + r_b)
    )
    root_term = np.maximum(root_term, 0.0)

    area[partial] = (
        r_a**2 * np.arccos(arg1)
        + r_b**2 * np.arccos(arg2)
        - 0.5 * np.sqrt(root_term)
    )

    return area


# ============================================================
# 5. SET UP TOY BINARY GEOMETRY
# ============================================================

P_seconds_lit = WUMA_PERIOD_LIT * 86400.0

a_m = (
    G * (M1_LIT + M2_LIT) * M_SUN * P_seconds_lit**2
    / (4.0 * np.pi**2)
) ** (1.0 / 3.0)

a_rsun = a_m / R_SUN

r1 = R1_LIT / a_rsun
r2 = R2_LIT / a_rsun

surface_brightness_ratio = (T2_LIT / T1_LIT) ** 4

print("\nTOY MODEL SETUP")
print("----------------")
print(f"Literature comparison period: {WUMA_PERIOD_LIT:.7f} days")
print(f"Published comparison inclination: {WUMA_INCLINATION_LIT:.1f} degrees")
print(f"Estimated separation used for toy model: {a_rsun:.3f} solar radii")
print(f"R1/a = {r1:.3f}")
print(f"R2/a = {r2:.3f}")
print(f"Surface brightness ratio S2/S1 ≈ {surface_brightness_ratio:.3f}")
print(f"Third light fraction = {THIRD_LIGHT_FRACTION:.3f}")


def toy_binary_flux(phases, inclination_deg, phase_shift):
    """
    Simple eclipsing-binary toy model.

    Assumptions:
    - circular orbit
    - spherical stars
    - uniform surface brightness
    - simple overlap geometry
    - no limb darkening
    - no Roche/contact distortion
    - optional third light

    This is not a full W UMa contact-binary model.
    """
    shifted_phase = (phases + phase_shift) % 1.0
    theta = 2.0 * np.pi * shifted_phase

    inc = np.radians(inclination_deg)

    # Projected center-to-center separation in units of orbital separation a
    d_projected = np.sqrt(
        np.sin(theta) ** 2
        + (np.cos(theta) * np.cos(inc)) ** 2
    )

    overlap = circle_overlap_area(d_projected, r1, r2)

    S1 = 1.0
    S2 = surface_brightness_ratio

    F1 = np.pi * r1**2 * S1
    F2 = np.pi * r2**2 * S2
    F_binary_out = F1 + F2

    # third-light flux relative to binary flux
    F3 = THIRD_LIGHT_FRACTION / (1.0 - THIRD_LIGHT_FRACTION) * F_binary_out

    flux = np.full_like(phases, F_binary_out, dtype=float)

    # At phase 0, star 2 is assumed to be in front of star 1
    star2_in_front = np.cos(theta) > 0

    # star 2 blocks star 1
    flux[star2_in_front] -= S1 * overlap[star2_in_front]

    # star 1 blocks star 2
    flux[~star2_in_front] -= S2 * overlap[~star2_in_front]

    flux_total = flux + F3
    flux_out = F_binary_out + F3

    return flux_total / flux_out


def model_magnitude_shape(phases, inclination_deg, phase_shift):
    """
    Convert toy flux model to magnitude shape.
    This does not include the vertical magnitude offset yet.
    """
    flux = toy_binary_flux(phases, inclination_deg, phase_shift)
    flux = np.clip(flux, 1e-8, None)
    return -2.5 * np.log10(flux)


def best_vertical_offset(observed_mag, model_shape, uncertainty):
    """
    Best additive magnitude offset for a fixed model shape.
    """
    weights = 1.0 / uncertainty**2
    offset = np.sum(weights * (observed_mag - model_shape)) / np.sum(weights)
    return offset


def chi_square_for_model(phases, observed_mag, uncertainty,
                         inclination_deg, phase_shift):
    """
    Chi-square after fitting only vertical magnitude offset.
    """
    shape = model_magnitude_shape(phases, inclination_deg, phase_shift)
    offset = best_vertical_offset(observed_mag, shape, uncertainty)
    model_mag = shape + offset
    chi2 = np.sum(((observed_mag - model_mag) / uncertainty) ** 2)
    return chi2, offset


def fit_inclination(phases, observed_mag, uncertainty,
                    inclination_grid, phase_shift_grid):
    """
    Search inclination and phase shift.
    Returns best inclination, phase shift, offset, chi-square,
    and the full best-chi-square-vs-inclination array.
    """
    results = []

    for inc in inclination_grid:
        best_chi2_for_inc = np.inf
        best_shift_for_inc = None
        best_offset_for_inc = None

        for shift in phase_shift_grid:
            chi2, offset = chi_square_for_model(
                phases,
                observed_mag,
                uncertainty,
                inc,
                shift
            )

            if chi2 < best_chi2_for_inc:
                best_chi2_for_inc = chi2
                best_shift_for_inc = shift
                best_offset_for_inc = offset

        results.append(
            [inc, best_chi2_for_inc, best_shift_for_inc, best_offset_for_inc]
        )

    results = np.array(results, dtype=float)

    best_idx = np.argmin(results[:, 1])

    best_inc = results[best_idx, 0]
    best_chi2 = results[best_idx, 1]
    best_shift = results[best_idx, 2]
    best_offset = results[best_idx, 3]

    return best_inc, best_shift, best_offset, best_chi2, results


# ============================================================
# 6. RAW LIGHT CURVE PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.errorbar(
    time_since_start,
    mag,
    yerr=mag_err,
    fmt="o",
    markersize=4,
    capsize=2,
    label="Observed data"
)

plt.gca().invert_yaxis()
plt.xlabel("Time Since First Observation [days]")
plt.ylabel("Magnitude")
plt.title("W Ursae Majoris: Magnitude vs Time")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 7. PERIOD SEARCH FROM YOUR DATA
# ============================================================

best_period, best_power, period_grid, power = estimate_period_lomb_scargle(
    time,
    mag
)

print("\nPERIOD SEARCH FROM YOUR DATA")
print("----------------------------")
print(f"Best estimated period = {best_period:.6f} days")
print(f"Best periodogram power = {best_power:.4f}")
print(f"Literature W UMa period = {WUMA_PERIOD_LIT:.7f} days")

if baseline_days < WUMA_PERIOD_LIT:
    print("\nWARNING")
    print("-------")
    print("Your time baseline is shorter than one full W UMa orbit.")
    print("The period estimate from this dataset alone may be unreliable.")

plt.figure(figsize=(10, 5))

plt.plot(period_grid, power, label="Lomb-Scargle power")
plt.axvline(
    best_period,
    linestyle="-",
    label=f"Best from data: {best_period:.4f} d"
)
plt.axvline(
    WUMA_PERIOD_LIT,
    linestyle="--",
    label=f"Literature: {WUMA_PERIOD_LIT:.4f} d"
)

plt.xlabel("Trial Period [days]")
plt.ylabel("Power")
plt.title("Period Search Check")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 8. PHASE FOLD USING ESTIMATED PERIOD
# ============================================================

phase_estimated = ((time - time[0]) / best_period) % 1.0
coverage_estimated = phase_coverage_fraction(phase_estimated)

print("\nPHASE COVERAGE")
print("--------------")
print(f"Phase coverage using estimated period: {coverage_estimated:.3f}")

plt.figure(figsize=(10, 6))

plt.errorbar(
    phase_estimated,
    mag,
    yerr=mag_err,
    fmt="o",
    markersize=4,
    capsize=2,
    label=f"Folded with estimated P = {best_period:.6f} d"
)

plt.errorbar(
    phase_estimated + 1.0,
    mag,
    yerr=mag_err,
    fmt="o",
    markersize=4,
    capsize=2
)

plt.gca().invert_yaxis()
plt.xlabel("Orbital Phase")
plt.ylabel("Magnitude")
plt.title("Phase-Folded Light Curve Using Estimated Period")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 9. PHASE FOLD USING LITERATURE PERIOD FOR COMPARISON
# ============================================================

phase_literature = ((time - WUMA_T0_LIT) / WUMA_PERIOD_LIT) % 1.0
coverage_literature = phase_coverage_fraction(phase_literature)

print(f"Phase coverage using literature period: {coverage_literature:.3f}")

plt.figure(figsize=(10, 6))

plt.errorbar(
    phase_literature,
    mag,
    yerr=mag_err,
    fmt="o",
    markersize=4,
    capsize=2,
    label=f"Folded with literature P = {WUMA_PERIOD_LIT:.7f} d"
)

plt.errorbar(
    phase_literature + 1.0,
    mag,
    yerr=mag_err,
    fmt="o",
    markersize=4,
    capsize=2
)

plt.gca().invert_yaxis()
plt.xlabel("Orbital Phase")
plt.ylabel("Magnitude")
plt.title("Phase-Folded Light Curve Using Literature Period")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 10. TOY INCLINATION FIT USING ESTIMATED PERIOD
# ============================================================

inclination_grid = np.linspace(
    INCLINATION_MIN,
    INCLINATION_MAX,
    N_INCLINATION_GRID
)

phase_shift_grid = np.linspace(
    PHASE_SHIFT_MIN,
    PHASE_SHIFT_MAX,
    N_PHASE_SHIFT_GRID
)

best_inc, best_shift, best_offset, best_chi2, inclination_results = fit_inclination(
    phase_estimated,
    mag,
    sigma,
    inclination_grid,
    phase_shift_grid
)

dof = len(mag) - 3
reduced_chi2 = best_chi2 / dof

print("\nTOY INCLINATION FIT USING ESTIMATED PERIOD")
print("------------------------------------------")
print(f"Best toy inclination = {best_inc:.2f} degrees")
print(f"Best phase shift = {best_shift:.4f}")
print(f"Reduced chi-square = {reduced_chi2:.2f}")
print(f"Published W UMa inclination = {WUMA_INCLINATION_LIT:.1f} degrees")

delta_chi2 = inclination_results[:, 1] - np.min(inclination_results[:, 1])

plt.figure(figsize=(10, 6))

plt.plot(
    inclination_results[:, 0],
    delta_chi2,
    marker="o",
    markersize=3
)

plt.axvline(
    best_inc,
    linestyle="-",
    label=f"Best toy fit: {best_inc:.1f} deg"
)

plt.axvline(
    WUMA_INCLINATION_LIT,
    linestyle="--",
    label=f"Published: {WUMA_INCLINATION_LIT:.1f} deg"
)

plt.xlabel("Inclination [degrees]")
plt.ylabel(r"$\Delta \chi^2$")
plt.title("Toy Inclination Search")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 11. BEST TOY MODEL PLOT
# ============================================================

phase_model = np.linspace(0, 2, 1200)
phase_model_wrapped = phase_model % 1.0

best_model_shape = model_magnitude_shape(
    phase_model_wrapped,
    best_inc,
    best_shift
)

best_model_mag = best_model_shape + best_offset

published_shape = model_magnitude_shape(
    phase_model_wrapped,
    WUMA_INCLINATION_LIT,
    best_shift
)

published_mag = published_shape + best_offset

plt.figure(figsize=(10, 6))

plt.errorbar(
    phase_estimated,
    mag,
    yerr=mag_err,
    fmt="o",
    markersize=4,
    capsize=2,
    label="Observed data"
)

plt.errorbar(
    phase_estimated + 1.0,
    mag,
    yerr=mag_err,
    fmt="o",
    markersize=4,
    capsize=2
)

plt.plot(
    phase_model,
    best_model_mag,
    linewidth=2,
    label=f"Best toy model: i = {best_inc:.1f} deg"
)

plt.plot(
    phase_model,
    published_mag,
    linestyle="--",
    linewidth=2,
    label=f"Published inclination shape: i = {WUMA_INCLINATION_LIT:.1f} deg"
)

plt.gca().invert_yaxis()
plt.xlabel("Orbital Phase")
plt.ylabel("Magnitude")
plt.title("Phase-Folded Data with Toy Inclination Model")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 12. MONTE CARLO PERIOD + INCLINATION UNCERTAINTY
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

period_results = []
inclination_results_mc = []
chi2_results_mc = []

print("\nRUNNING MONTE CARLO")
print("-------------------")
print(f"Number of trials: {N_MONTE_CARLO}")
print("This may take a little while depending on your computer.")

for n in range(N_MONTE_CARLO):

    # Perturb magnitudes by photometric uncertainties
    mag_perturbed = mag + rng.normal(0.0, sigma)

    # Estimate period for this perturbed dataset
    period_mc, power_mc, _, _ = estimate_period_lomb_scargle(
        time,
        mag_perturbed,
        min_period=PERIOD_MIN,
        max_period=PERIOD_MAX,
        n_periods=5000
    )

    # Fold with that trial period
    phase_mc = ((time - time[0]) / period_mc) % 1.0

    # Fit toy inclination
    inc_mc, shift_mc, offset_mc, chi2_mc, results_mc = fit_inclination(
        phase_mc,
        mag_perturbed,
        sigma,
        inclination_grid,
        phase_shift_grid
    )

    period_results.append(period_mc)
    inclination_results_mc.append(inc_mc)
    chi2_results_mc.append(chi2_mc)

period_results = np.array(period_results)
inclination_results_mc = np.array(inclination_results_mc)
chi2_results_mc = np.array(chi2_results_mc)

period_median = np.median(period_results)
period_low = np.percentile(period_results, 16)
period_high = np.percentile(period_results, 84)

inc_median = np.median(inclination_results_mc)
inc_low = np.percentile(inclination_results_mc, 16)
inc_high = np.percentile(inclination_results_mc, 84)

print("\nMONTE CARLO RESULTS")
print("-------------------")
print(
    f"Estimated period = {period_median:.6f} "
    f"+{period_high - period_median:.6f} "
    f"/ -{period_median - period_low:.6f} days"
)

print(
    f"Toy inclination = {inc_median:.2f} "
    f"+{inc_high - inc_median:.2f} "
    f"/ -{inc_median - inc_low:.2f} degrees"
)

print("\nCOMPARISON VALUES")
print("-----------------")
print(f"Literature period = {WUMA_PERIOD_LIT:.7f} days")
print(f"Literature inclination = {WUMA_INCLINATION_LIT:.1f} degrees")


# ============================================================
# 13. MONTE CARLO DISTRIBUTION PLOTS
# ============================================================

plt.figure(figsize=(9, 5))

plt.hist(period_results, bins=30)
plt.axvline(
    period_median,
    linestyle="-",
    label=f"Median = {period_median:.5f} d"
)
plt.axvline(
    WUMA_PERIOD_LIT,
    linestyle="--",
    label=f"Literature = {WUMA_PERIOD_LIT:.5f} d"
)

plt.xlabel("Estimated Period [days]")
plt.ylabel("Number of Monte Carlo Trials")
plt.title("Monte Carlo Period Distribution")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(9, 5))

plt.hist(inclination_results_mc, bins=30)
plt.axvline(
    inc_median,
    linestyle="-",
    label=f"Median = {inc_median:.1f} deg"
)
plt.axvline(
    WUMA_INCLINATION_LIT,
    linestyle="--",
    label=f"Published = {WUMA_INCLINATION_LIT:.1f} deg"
)

plt.xlabel("Toy-Model Inclination [degrees]")
plt.ylabel("Number of Monte Carlo Trials")
plt.title("Monte Carlo Inclination Distribution")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\nFINAL SUMMARY")
print("-------------")
print("This code used your photometry to:")
print("1. Plot magnitude vs. time")
print("2. Estimate the period with a Lomb-Scargle periodogram")
print("3. Phase-fold the light curve")
print("4. Fit a simplified toy-model inclination")
print("5. Use Monte Carlo trials to estimate uncertainty")
print()
print("Important scientific caution:")
print("The inclination value is a toy-model estimate, not a final physical solution.")
print("W UMa is a contact binary, so a real solution requires full binary modeling.")
print("Use PHOEBE or Wilson-Devinney for a publishable/contact-binary inclination.")