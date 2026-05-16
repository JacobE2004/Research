import os
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

class NovaSpectrum:
    def __init__(self, file_path, wavelength_col=0, flux_col=1, rescale=False):
        self.file_path = file_path
        self.wavelength_col = wavelength_col
        self.flux_col = flux_col
        self.rescale = rescale  # if True, analysis is conducted on normalized flux
        self.data = None
        self.smoothed_flux = None
        self.peaks = None
        # Known emission lines (including Fe II 42 triplet)
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
        self._load_data()
    
    def _load_data(self):
        # Use comment="#" so that files with header/comment lines are handled correctly.
        try:
            df = pd.read_csv(self.file_path, delim_whitespace=True, comment="#", header=None)
            self.data = pd.DataFrame({
                "Wavelength": df.iloc[:, self.wavelength_col],
                "Flux": df.iloc[:, self.flux_col]
            })
        except Exception as e:
            raise ValueError(f"Error reading file: {e}")
    
    def smooth(self, sigma=3):
        # If rescaling is enabled, normalize the flux before smoothing.
        if self.rescale:
            norm_factor = self.data["Flux"].max()
            self.data["Flux"] = self.data["Flux"] / norm_factor
        self.data["SmoothedFlux"] = gaussian_filter1d(self.data["Flux"], sigma=sigma)
        self.smoothed_flux = self.data["SmoothedFlux"]
    
    def find_peaks(self, height, distance=10):
        if self.smoothed_flux is None:
            self.smooth()
        peaks, _ = find_peaks(self.smoothed_flux, height=height, distance=distance)
        self.peaks = peaks
        return self.data["Wavelength"].iloc[peaks], self.smoothed_flux.iloc[peaks]
    
    def plot(self, label_detected_lines=False, log_scale=True):
        if self.smoothed_flux is None:
            self.smooth()
        
        plt.figure(figsize=(14, 6))
        plt.plot(self.data["Wavelength"], self.data["Flux"], alpha=0.5, label="Raw Flux", color='gray')
        plt.plot(self.data["Wavelength"], self.smoothed_flux, label="Smoothed", color='black')
        ax = plt.gca()
        
        if self.peaks is not None:
            wl_peaks = self.data["Wavelength"].iloc[self.peaks]
            flux_peaks = self.smoothed_flux.iloc[self.peaks]
            plt.scatter(wl_peaks, flux_peaks, color='red', s=20, label="Detected Peaks")
            peaks_text = "Detected Peaks:\n" + ", ".join([f"{w:.1f}" for w in wl_peaks])
            ax.text(0.02, 0.98, peaks_text, transform=ax.transAxes,
                    ha="left", va="top",
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'),
                    fontsize=9)
        
        for label, wl in self.known_lines.items():
            plt.axvline(wl, linestyle='--', color='blue', alpha=0.5)
        known_lines_text = "\n".join([f"{label}: {wl} Å" for label, wl in self.known_lines.items()])
        ax.text(0.98, 0.98, known_lines_text, transform=ax.transAxes,
                ha="right", va="top",
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'),
                fontsize=9)
        
        if log_scale:
            plt.yscale("log")
            ylabel = "Flux (log scale)"
        else:
            ylabel = "Flux"
        plt.xlabel("Wavelength (Å)")
        plt.ylabel(ylabel)
        plt.title("Nova Spectrum")
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def integrate_line_flux(self, line_center, window=10):
        mask = (self.data["Wavelength"] > line_center - window) & (self.data["Wavelength"] < line_center + window)
        if not mask.any():
            return 0
        wavelength = self.data["Wavelength"][mask]
        flux = self.data["Flux"][mask]
        return np.trapz(flux, wavelength)
    
    def calculate_equivalent_width(self, line_center, window=10):
        side_window = 10
        left_mask = (self.data["Wavelength"] > line_center - window - side_window) & \
                    (self.data["Wavelength"] < line_center - window)
        right_mask = (self.data["Wavelength"] > line_center + window) & \
                     (self.data["Wavelength"] < line_center + window + side_window)
        continuum_flux = np.concatenate([self.data["Flux"][left_mask], self.data["Flux"][right_mask]])
        continuum_level = np.median(continuum_flux) if len(continuum_flux) > 0 else 1.0
        line_mask = (self.data["Wavelength"] > line_center - window) & (self.data["Wavelength"] < line_center + window)
        if not line_mask.any():
            return 0
        wavelength = self.data["Wavelength"][line_mask]
        flux = self.data["Flux"][line_mask]
        return np.trapz(1 - flux / continuum_level, wavelength)
    
    def estimate_velocity(self, line_center, window=10):
        c = 3e5
        mask = (self.data["Wavelength"] > line_center - window) & (self.data["Wavelength"] < line_center + window)
        if not mask.any():
            return None
        x = self.data["Wavelength"][mask].values
        y = self.data["Flux"][mask].values
        
        def gaussian(x, amp, cen, wid):
            return amp * np.exp(-(x - cen)**2 / (2 * wid**2))
        
        p0 = [np.max(y), line_center, 2.0]
        try:
            popt, _ = curve_fit(gaussian, x, y, p0=p0)
            fwhm = 2.355 * popt[2]
            return (fwhm / line_center) * c
        except RuntimeError:
            fwhm = 2.355 * p0[2]
            return (fwhm / line_center) * c

    def export_peaks_with_annotations_to_csv(self, output_path="peaks_analysis.csv", match_tolerance=20):
        if self.peaks is None:
            self.find_peaks(height=0.05, distance=10)
        peak_wavelengths = self.data["Wavelength"].iloc[self.peaks].values
        peak_fluxes = self.smoothed_flux.iloc[self.peaks].values
        
        rows = []
        for wl, flux in zip(peak_wavelengths, peak_fluxes):
            integrated_flux = self.integrate_line_flux(wl)
            eq_width = self.calculate_equivalent_width(wl)
            annotation = ""
            separation = ""
            velocity = ""
            for known_label, known_wl in self.known_lines.items():
                if abs(wl - known_wl) <= match_tolerance:
                    annotation = known_label
                    separation = wl - known_wl
                    velocity = self.estimate_velocity(wl)
                    break
            rows.append({
                "Peak Wavelength (Å)": wl,
                "Peak Flux": flux,
                "Integrated Flux": integrated_flux,
                "Equivalent Width (Å)": eq_width,
                "Separation (Å)": separation,
                "Estimated Velocity (km/s)": velocity if velocity is not None else "",
                "Annotation": annotation
            })
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Saved peaks analysis data to {output_path}")
    
    @staticmethod
    def launch_gui():
        def browse_dir():
            folder = filedialog.askdirectory()
            if folder:
                dir_entry.delete(0, tk.END)
                dir_entry.insert(0, folder)
        
        def update_file_list():
            folder = dir_entry.get()
            pattern = pattern_entry.get().strip()
            file_listbox.delete(0, tk.END)
            if folder:
                for fname in os.listdir(folder):
                    if fname.endswith(".txt") and (pattern in fname if pattern else True):
                        file_listbox.insert(tk.END, fname)
        
        def on_select(event):
            selection = file_listbox.curselection()
            if selection:
                selected = file_listbox.get(selection[0])
                full_path = os.path.join(dir_entry.get(), selected)
                selected_file_entry.delete(0, tk.END)
                selected_file_entry.insert(0, full_path)
        
        def analyze_spectrum():
            nonlocal spectrum
            file_path = selected_file_entry.get()
            if not os.path.isfile(file_path):
                messagebox.showerror("Error", "Please select a valid spectrum file.")
                return
            try:
                wave_col = int(wave_col_entry.get())
                flux_col = int(flux_col_entry.get())
                peak_thresh = float(peak_threshold_entry.get())
                # Set rescale=True so that analysis is done on normalized data.
                spectrum = NovaSpectrum(file_path, wavelength_col=wave_col, flux_col=flux_col, rescale=True)
                spectrum.smooth()
                spectrum.find_peaks(height=peak_thresh, distance=10)
                spectrum.plot(log_scale=True)
            except Exception as e:
                messagebox.showerror("Analysis Error", str(e))
        
        def run_export_csv():
            nonlocal spectrum
            if spectrum is None:
                messagebox.showerror("Error", "Please analyze a spectrum first.")
                return
            filename = filedialog.asksaveasfilename(defaultextension=".csv",
                                                    filetypes=[("CSV files", "*.csv")])
            if filename:
                try:
                    spectrum.export_peaks_with_annotations_to_csv(output_path=filename)
                    messagebox.showinfo("Export Complete", f"Peaks analysis saved to '{filename}'")
                except Exception as e:
                    messagebox.showerror("Export Error", str(e))
        
        spectrum = None
        root = tk.Tk()
        root.title("Nova Spectrum Analyzer")
        
        tk.Label(root, text="Select Spectrum Directory:").pack()
        dir_entry = tk.Entry(root, width=50)
        dir_entry.pack()
        tk.Button(root, text="Browse", command=browse_dir).pack()
        
        tk.Label(root, text="Enter File Pattern (e.g. 'bej') or Leave Blank:").pack()
        pattern_entry = tk.Entry(root, width=30)
        pattern_entry.pack()
        
        file_listbox = tk.Listbox(root, width=50, height=10)
        file_listbox.pack()
        file_listbox.bind('<<ListboxSelect>>', on_select)
        
        tk.Button(root, text="Refresh File List", command=update_file_list).pack()
        
        tk.Label(root, text="Selected File:").pack()
        selected_file_entry = tk.Entry(root, width=50)
        selected_file_entry.pack()
        
        tk.Label(root, text="Wavelength Column Index (default 0):").pack()
        wave_col_entry = tk.Entry(root, width=10)
        wave_col_entry.insert(0, "0")
        wave_col_entry.pack()
        
        tk.Label(root, text="Flux Column Index (default 1):").pack()
        flux_col_entry = tk.Entry(root, width=10)
        flux_col_entry.insert(0, "1")
        flux_col_entry.pack()
        
        tk.Label(root, text="Peak Height Threshold (default 0.05 for normalized data):").pack()
        peak_threshold_entry = tk.Entry(root, width=10)
        peak_threshold_entry.insert(0, "0.05")
        peak_threshold_entry.pack()
        
        tk.Button(root, text="Analyze Spectrum", command=analyze_spectrum).pack()
        tk.Button(root, text="Export CSV", command=run_export_csv).pack()
        
        root.mainloop()

if __name__ == "__main__":
    NovaSpectrum.launch_gui()
