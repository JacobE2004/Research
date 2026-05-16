import os
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# Speed of light constant in km/s (used for Doppler calculations)
C_LIGHT = 299792.458

class NovaSpectrum:
    def __init__(self, file_path, wavelength_col=0, flux_col=1, rescale=False):
        self.file_path = file_path  # Path to the data file
        self.wavelength_col = wavelength_col  # Column index for wavelengths
        self.flux_col = flux_col  # Column index for flux values
        self.rescale = rescale  # Whether to normalize flux to 1
        self.data = None  # Placeholder for spectrum data (Pandas DataFrame)
        self.smoothed_flux = None  # Placeholder for smoothed flux data
        self.peaks = None  # Placeholder for peak indices

        # Known spectral lines with rest wavelengths
        self.known_lines = {
            "Hβ": 4861,
            "[O III]": 5007,
            "Hα": 6563,
            "[N II]": 6584,
            "[S II]": 6731,
            "Fe II 42-4923": 4923.92,
            "Fe II 42-5018": 5018.44,
            "Fe II 42-5169": 5169.03
        }

        self._load_data()  # Load the spectrum file

    def _load_data(self):
        # Load spectrum file as DataFrame with wavelength & flux columns
        df = pd.read_csv(self.file_path, delim_whitespace=True, comment="#", header=None)
        self.data = pd.DataFrame({
            "Wavelength": df.iloc[:, self.wavelength_col],  # Extract wavelengths
            "Flux": df.iloc[:, self.flux_col]  # Extract fluxes
        })

    def smooth(self, sigma=3):
        # Optional normalization if enabled
        if self.rescale:
            self.data["Flux"] /= self.data["Flux"].max()
        # Apply Gaussian smoothing to flux column
        self.data["SmoothedFlux"] = gaussian_filter1d(self.data["Flux"], sigma=sigma)
        self.smoothed_flux = self.data["SmoothedFlux"]  # Store result

    def find_peaks(self, height, distance=10):
        # Find local peaks in the smoothed flux
        if self.smoothed_flux is None:
            self.smooth()  # Ensure smoothing was done
        peaks, _ = find_peaks(self.smoothed_flux, height=height, distance=distance)
        self.peaks = peaks  # Store detected peak indices
        return self.data["Wavelength"].iloc[peaks], self.smoothed_flux.iloc[peaks]  # Return peak values

    def estimate_velocity(self, rest_wavelength, window=10):
        # Create a mask to select data points near the target rest wavelength
        mask = (self.data["Wavelength"] > rest_wavelength - window) & (self.data["Wavelength"] < rest_wavelength + window)
        if not mask.any():
            return None, None  # If no data found, exit

        x = self.data["Wavelength"][mask].values  # Wavelength slice
        y = self.data["Flux"][mask].values  # Flux slice

        # Define Gaussian function for fitting
        def gaussian(x, amp, cen, wid):
            return amp * np.exp(-(x - cen)**2 / (2 * wid**2))

        # Initial guess for fitting: amplitude, center, width
        p0 = [np.max(y), rest_wavelength, 1.0]

        try:
            # Perform Gaussian curve fit
            popt, _ = curve_fit(gaussian, x, y, p0=p0)
            cen = popt[1]  # Extract fitted center wavelength
            # Calculate radial velocity using Doppler formula
            v_rad = C_LIGHT * (cen - rest_wavelength) / rest_wavelength
            return cen, v_rad  # Return observed center and velocity
        except RuntimeError:
            return None, None  # Return None if fit fails

    def plot(self, match_tolerance=10, log_scale=True):
        # Ensure data is smoothed
        if self.smoothed_flux is None:
            self.smooth()

        # Plot raw and smoothed flux
        plt.figure(figsize=(14, 6))
        plt.plot(self.data["Wavelength"], self.data["Flux"], alpha=0.5, label="Raw Flux", color='gray')
        plt.plot(self.data["Wavelength"], self.smoothed_flux, label="Smoothed Flux", color='black')

        # Plot detected peaks
        if self.peaks is not None:
            wl_peaks = self.data["Wavelength"].iloc[self.peaks]
            flux_peaks = self.smoothed_flux.iloc[self.peaks]
            plt.scatter(wl_peaks, flux_peaks, color='red', s=20, label="Detected Peaks")

        results = []  # To store matched line results

        # Iterate over known spectral lines to match and compute Doppler shifts
        for label, wl_rest in self.known_lines.items():
            # Find peaks within match tolerance of rest wavelength
            matched_peak = self.data["Wavelength"].iloc[self.peaks][np.abs(self.data["Wavelength"].iloc[self.peaks] - wl_rest) < match_tolerance]
            if not matched_peak.empty:
                obs_wl, v_rad = self.estimate_velocity(wl_rest)  # Fit Gaussian & get velocity
                if obs_wl is not None:
                    # Plot matched peak position
                    plt.axvline(obs_wl, color='red', linestyle='-', alpha=0.7)
                    plt.text(obs_wl, plt.ylim()[1]*0.85, f"{label}\n{v_rad:.1f} km/s",
                             rotation=90, va='top', fontsize=9, color='red')
                    # Store result
                    results.append({
                        "Line": label,
                        "Rest Wavelength (Å)": wl_rest,
                        "Observed Wavelength (Å)": obs_wl,
                        "Radial Velocity (km/s)": v_rad
                    })

        # Plot vertical lines for rest wavelengths of known lines
        for label, wl_rest in self.known_lines.items():
            plt.axvline(wl_rest, linestyle='--', color='blue', alpha=0.5)
            plt.text(wl_rest, plt.ylim()[1]*0.95, label, rotation=90, va='top', fontsize=8, color='blue')

        # Apply logarithmic y-axis if requested
        if log_scale:
            plt.yscale("log")

        # Final plot labels and formatting
        plt.xlabel("Wavelength (Å)")
        plt.ylabel("Flux (log scale)" if log_scale else "Flux")
        plt.title("Spectral Line Doppler Shift Analysis")
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Save results to CSV if any lines were detected
        if results:
            df = pd.DataFrame(results)
            output_file = os.path.splitext(self.file_path)[0] + "_velocity_analysis.csv"
            df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")

    @staticmethod
    def launch_gui():
        # GUI function to browse for a spectrum file
        def select_file():
            file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
            if file_path:
                file_entry.delete(0, tk.END)
                file_entry.insert(0, file_path)

        # GUI function to run analysis on selected file
        def run_analysis():
            path = file_entry.get()
            if not path:
                messagebox.showerror("Error", "No file selected.")
                return
            try:
                spectrum = NovaSpectrum(path)
                spectrum.smooth()
                spectrum.find_peaks(height=0.05, distance=10)
                spectrum.plot()
            except Exception as e:
                messagebox.showerror("Analysis Error", str(e))

        # Create GUI window
        root = tk.Tk()
        root.title("Nova Spectrum Analyzer - Multi-Line Doppler Shift")

        # GUI widgets for file selection and actions
        tk.Label(root, text="Select Spectrum File:").pack(pady=5)
        file_entry = tk.Entry(root, width=60)
        file_entry.pack(pady=5)
        tk.Button(root, text="Browse", command=select_file).pack(pady=5)
        tk.Button(root, text="Analyze Spectrum", command=run_analysis).pack(pady=10)

        root.mainloop()  # Run the GUI event loop

if __name__ == "__main__":
    NovaSpectrum.launch_gui()  # Run the GUI if script is executed