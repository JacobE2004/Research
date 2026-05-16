import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================================
# USER INPUT
# Put your spectra here
# group can be: "Galactic", "LMC", or "SMC"
# ==========================================================
spectra_files = [

    {
        "label": "LMCN-2026-03a",
        "group": "LMC",
        "file": r"C:\Users\Jmell\Dropbox\Research File\Nova_targets\LMCN_Folder\LMCN_2026_03a\tns_2026fbz_ascii_files\tns_2026fbz_2026-03-08_18-07-42_Lesedi_Mookodi_BlackGEM.txt"
    },

]

# ==========================================================
# REST WAVELENGTHS (Angstroms)
# ==========================================================
LINES = {
    "Hbeta": 4861.3,
    "HeII4686": 4685.7,
    "FeII5018": 5018.4,
    "Halpha": 6562.8,
    "OI7773": 7773.0,
}

C_KM_S = 299792.458

# ==========================================================
# HELPERS
# ==========================================================
def load_two_column_spectrum(file_path):
    data = np.loadtxt(file_path)
    wavelength = data[:, 0]
    flux = data[:, 1]
    return wavelength, flux

def normalize_flux(flux):
    med = np.median(flux)
    if med == 0:
        return flux.copy()
    return flux / med

def local_continuum(w, f, center, inner_window=20, outer_window=60):
    """
    Estimate local continuum using the left/right side regions around a line.
    Uses points in:
    [center-outer_window, center-inner_window] and
    [center+inner_window, center+outer_window]
    """
    left = (w >= center - outer_window) & (w <= center - inner_window)
    right = (w >= center + inner_window) & (w <= center + outer_window)

    continuum_points = np.concatenate([f[left], f[right]])

    if continuum_points.size < 4:
        return np.median(f)

    return np.median(continuum_points)

def line_peak_ratio(w, f_norm, center, search_window=20):
    """
    Returns peak flux / local continuum near a line.
    """
    mask = (w >= center - search_window) & (w <= center + search_window)
    if np.sum(mask) < 3:
        return np.nan, np.nan, np.nan

    local_w = w[mask]
    local_f = f_norm[mask]

    cont = local_continuum(w, f_norm, center)
    peak_idx = np.argmax(local_f)
    peak_flux = local_f[peak_idx]
    peak_wave = local_w[peak_idx]

    if cont == 0:
        return peak_wave, peak_flux, np.nan

    return peak_wave, peak_flux, peak_flux / cont

def line_fwhm(w, f_norm, center, search_window=40):
    """
    Rough FWHM estimate for an emission line.
    Measures width above local continuum at half-maximum.
    Returns:
      fwhm_angstrom, fwhm_km_s, peak_wave
    """
    mask = (w >= center - search_window) & (w <= center + search_window)
    if np.sum(mask) < 5:
        return np.nan, np.nan, np.nan

    local_w = w[mask]
    local_f = f_norm[mask]

    cont = local_continuum(w, f_norm, center)
    peak_idx = np.argmax(local_f)
    peak_flux = local_f[peak_idx]
    peak_wave = local_w[peak_idx]

    peak_height = peak_flux - cont
    if peak_height <= 0:
        return np.nan, np.nan, peak_wave

    half_level = cont + 0.5 * peak_height

    # Find left crossing
    left_w = np.nan
    for i in range(peak_idx, 0, -1):
        if local_f[i] >= half_level and local_f[i - 1] < half_level:
            x1, y1 = local_w[i - 1], local_f[i - 1]
            x2, y2 = local_w[i], local_f[i]
            if y2 != y1:
                left_w = x1 + (half_level - y1) * (x2 - x1) / (y2 - y1)
            else:
                left_w = x1
            break

    # Find right crossing
    right_w = np.nan
    for i in range(peak_idx, len(local_w) - 1):
        if local_f[i] >= half_level and local_f[i + 1] < half_level:
            x1, y1 = local_w[i], local_f[i]
            x2, y2 = local_w[i + 1], local_f[i + 1]
            if y2 != y1:
                right_w = x1 + (half_level - y1) * (x2 - x1) / (y2 - y1)
            else:
                right_w = x2
            break

    if np.isnan(left_w) or np.isnan(right_w):
        return np.nan, np.nan, peak_wave

    fwhm_ang = right_w - left_w
    fwhm_km_s = (fwhm_ang / center) * C_KM_S

    return fwhm_ang, fwhm_km_s, peak_wave

# ==========================================================
# PROCESS ALL SPECTRA
# ==========================================================
results = []
loaded_spectra = []

for spec in spectra_files:
    label = spec["label"]
    group = spec["group"]
    file_path = spec["file"]

    if not os.path.exists(file_path):
        print(f"Missing file: {file_path}")
        continue

    w, f = load_two_column_spectrum(file_path)
    f_norm = normalize_flux(f)

    loaded_spectra.append({
        "label": label,
        "group": group,
        "w": w,
        "f_norm": f_norm
    })

    # Peak ratios
    hb_wave, hb_peak, hb_ratio = line_peak_ratio(w, f_norm, LINES["Hbeta"])
    he_wave, he_peak, he_ratio = line_peak_ratio(w, f_norm, LINES["HeII4686"])
    fe_wave, fe_peak, fe_ratio = line_peak_ratio(w, f_norm, LINES["FeII5018"])
    ha_wave, ha_peak, ha_ratio = line_peak_ratio(w, f_norm, LINES["Halpha"])
    oi_wave, oi_peak, oi_ratio = line_peak_ratio(w, f_norm, LINES["OI7773"])

    # FWHM
    ha_fwhm_ang, ha_fwhm_kms, ha_peak_wave = line_fwhm(w, f_norm, LINES["Halpha"])
    hb_fwhm_ang, hb_fwhm_kms, hb_peak_wave = line_fwhm(w, f_norm, LINES["Hbeta"])

    # Ratios relative to Balmer lines
    heii_over_hb = he_ratio / hb_ratio if np.isfinite(he_ratio) and np.isfinite(hb_ratio) and hb_ratio != 0 else np.nan
    feii_over_hb = fe_ratio / hb_ratio if np.isfinite(fe_ratio) and np.isfinite(hb_ratio) and hb_ratio != 0 else np.nan
    oi_over_ha = oi_ratio / ha_ratio if np.isfinite(oi_ratio) and np.isfinite(ha_ratio) and ha_ratio != 0 else np.nan

    results.append({
        "label": label,
        "group": group,
        "Hbeta_peak_ratio": hb_ratio,
        "HeII4686_peak_ratio": he_ratio,
        "FeII5018_peak_ratio": fe_ratio,
        "Halpha_peak_ratio": ha_ratio,
        "OI7773_peak_ratio": oi_ratio,
        "HeII4686_over_Hbeta": heii_over_hb,
        "FeII5018_over_Hbeta": feii_over_hb,
        "OI7773_over_Halpha": oi_over_ha,
        "Halpha_FWHM_A": ha_fwhm_ang,
        "Halpha_FWHM_km_s": ha_fwhm_kms,
        "Hbeta_FWHM_A": hb_fwhm_ang,
        "Hbeta_FWHM_km_s": hb_fwhm_kms,
    })

# ==========================================================
# PRINT RESULTS TABLE
# ==========================================================
print("\n================ QUANTITATIVE COMPARISON =================")
header = (
    f"{'Label':25s} {'Group':10s} "
    f"{'HeII/Hb':>10s} {'FeII/Hb':>10s} {'OI/Ha':>10s} "
    f"{'Ha_FWHM(km/s)':>16s} {'Hb_FWHM(km/s)':>16s}"
)
print(header)
print("-" * len(header))

for r in results:
    print(
        f"{r['label'][:25]:25s} "
        f"{r['group']:10s} "
        f"{r['HeII4686_over_Hbeta']:10.3f} "
        f"{r['FeII5018_over_Hbeta']:10.3f} "
        f"{r['OI7773_over_Halpha']:10.3f} "
        f"{r['Halpha_FWHM_km_s']:16.1f} "
        f"{r['Hbeta_FWHM_km_s']:16.1f}"
    )

# ==========================================================
# PLOT 1: FULL NORMALIZED OVERLAY
# ==========================================================
plt.figure(figsize=(14, 7))

for spec in loaded_spectra:
    plt.plot(spec["w"], spec["f_norm"], linewidth=1.2, label=f"{spec['label']} ({spec['group']})")

for name, lam in LINES.items():
    plt.axvline(lam, linestyle="--", linewidth=0.8, alpha=0.6)
    plt.text(lam + 3, plt.ylim()[1] * 0.82, name, rotation=90, va="bottom", fontsize=14)

plt.xlabel("Wavelength (Å)", fontsize=20)
plt.ylabel("Normalized Flux", fontsize=20)
plt.title("Normalized Nova Spectra Comparison", fontsize=20)

plt.legend()
plt.tight_layout()
plt.show()

# ==========================================================
# PLOT 2: Hbeta REGION
# ==========================================================
plt.figure(figsize=(12, 6))

for spec in loaded_spectra:
    mask = (spec["w"] >= 4550) & (spec["w"] <= 5100)
    plt.plot(spec["w"][mask], spec["f_norm"][mask], linewidth=1.2, label=f"{spec['label']} ({spec['group']})")

for lam, name in [(4685.7, "He II 4686"), (4861.3, "Hβ"), (5018.4, "Fe II 5018")]:
    plt.axvline(lam, linestyle="--", linewidth=0.8, alpha=0.6)
    plt.text(lam + 3, plt.ylim()[1] * 0.82, name, rotation=90, va="bottom", fontsize=9)

plt.xlabel("Wavelength (Å)", fontsize=14)
plt.ylabel("Normalized Flux", fontsize=14)
plt.title("Hβ / He II / Fe II Region", fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================================
# PLOT 3: Halpha REGION
# ==========================================================
plt.figure(figsize=(12, 6))

for spec in loaded_spectra:
    mask = (spec["w"] >= 6400) & (spec["w"] <= 7850)
    plt.plot(spec["w"][mask], spec["f_norm"][mask], linewidth=1.2, label=f"{spec['label']} ({spec['group']})")

for lam, name in [(6562.8, "Hα"), (7773.0, "O I 7773")]:
    plt.axvline(lam, linestyle="--", linewidth=0.8, alpha=0.6)
    plt.text(lam + 3, plt.ylim()[1] * 0.82, name, rotation=90, va="bottom", fontsize=9)

plt.xlabel("Wavelength (Å)", fontsize=14)
plt.ylabel("Normalized Flux", fontsize=14)
plt.title("Hα / O I Region", fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()