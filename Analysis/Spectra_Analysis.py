import numpy as np
import matplotlib.pyplot as plt
import os

# ======================================================
# File path
# ======================================================
file_path = r"C:\Users\Jmell\Dropbox\Research File\Nova_targets\LMCN_Folder\LMCN_2026_03a\tns_2026fbz_ascii_files\tns_2026fbz_2026-03-08_18-07-42_Lesedi_Mookodi_BlackGEM.txt"

# ======================================================
# Load spectrum
# ======================================================
wavelength = []
flux = []

with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                wavelength.append(float(parts[0]))
                flux.append(float(parts[1]))
            except ValueError:
                continue

wavelength = np.array(wavelength)
flux = np.array(flux)

# ======================================================
# Common nova classification emission lines (Angstroms)
# ======================================================
emission_lines = {
    # Hydrogen Balmer
    "Hδ": 4101.7,
    "Hγ": 4340.5,
    "Hβ": 4861.3,
    "Hα": 6562.8,

    # Helium
    "He I 4471": 4471.5,
    "He II 4686": 4685.7,
    "He I 5876": 5875.6,
    "He I 6678": 6678.2,
    "He I 7065": 7065.2,

    # Nitrogen
    "N III 4640": 4640.0,
    "N II 5679": 5679.6,

    # Oxygen
    "O I 5577": 5577.3,
    "O I 6300": 6300.3,
    "O I 6364": 6363.8,
    "O I 7773": 7773.0,

    # Fe II lines often seen in Fe II novae
    "Fe II 4233": 4233.2,
    "Fe II 4924": 4923.9,
    "Fe II 5018": 5018.4,
    "Fe II 5169": 5169.0,
}

# ======================================================
# Plot
# ======================================================
plt.figure(figsize=(14, 7))
plt.plot(wavelength, flux, linewidth=1.0, label="Spectrum")

ymax = np.max(flux)

for name, lam in emission_lines.items():
    if wavelength.min() <= lam <= wavelength.max():
        plt.axvline(lam, linestyle="--", linewidth=0.9, alpha=0.8)
        plt.text(
            lam + 3,
            ymax * 0.82,
            name,
            rotation=90,
            va="bottom",
            fontsize=9
        )

plt.xlabel("Wavelength (Å)", fontsize=14)
plt.ylabel("Flux", fontsize=14)
plt.title("Spectrum with Common Nova Classification Emission Lines", fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Optional save
output_png = os.path.join(os.path.dirname(file_path), "nova_spectrum_emission_lines.png")
plt.savefig(output_png, dpi=300, bbox_inches="tight")

plt.show()

print(f"Saved plot to: {output_png}")