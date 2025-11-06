# Import GUI and plotting libraries
import tkinter as tk  # Tkinter core module
from tkinter import filedialog, messagebox  # File dialogs and error dialogs
import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation
from matplotlib.figure import Figure  # Base Figure class
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # Embed Matplotlib with toolbar

# Class to perform nova analysis and plotting
class NovaAnalyzer:
    def __init__(self, tpeak, tmax_jds, t2_jds, fig, ax, canvas):
        # Julian Date of peak brightness
        self.tpeak = tpeak
        # t_max uncertainty bracket (JD before, central, after)
        self.tmax_jds = tmax_jds
        # t2 uncertainty bracket (JD before, central, after)
        self.t2_jds = t2_jds
        # Matplotlib objects
        self.fig = fig
        self.ax = ax
        self.canvas = canvas

    def compute_uncertainty(self, jd1, jd2, jd3):
        # Compute asymmetric uncertainties around central JD
        return jd2 - jd1, jd3 - jd2

    def plot_generic(self, days, mags, mag_errs=None, label="Data"):
        # Clear axes and invert y-axis for magnitude
        self.ax.clear()
        self.ax.invert_yaxis()
        # Plot data points
        if mag_errs is not None:
            self.ax.errorbar(days, mags, yerr=mag_errs, fmt='o', label=label)
        else:
            self.ax.plot(days, mags, 'o', label=label)
        # Compute and overlay uncertainty regions
        tmax_b, tmax_a = self.compute_uncertainty(*self.tmax_jds)
        t2_b, t2_a = self.compute_uncertainty(*self.t2_jds)
        t2_central = self.t2_jds[1] - self.tpeak
        self.ax.axvspan(-tmax_b, tmax_a, color='red', alpha=0.3,
                        label=f"t_peak unc: -{tmax_b:.2f}/+{tmax_a:.2f} d")
        self.ax.axvspan(t2_central - t2_b, t2_central + t2_a, color='blue', alpha=0.3,
                        label=f"t2 unc: -{t2_b:.2f}/+{t2_a:.2f} d")
        # Vertical lines for t_peak and t2
        self.ax.axvline(0, color='k', linestyle='--', label='t_peak')
        self.ax.axvline(t2_central, color='k', linestyle='--', label='t2')
        # Labels and legend
        self.ax.set_xlabel('Days since peak brightness')
        self.ax.set_ylabel('Magnitude')
        self.ax.set_title('Nova Light Curve')
        self.ax.legend()
        # Adjust layout and draw
        self.fig.tight_layout()
        self.canvas.draw()

    # Methods for different file types
    def analyze_aavso(self, file_path):
        # Load AAVSO data and plot
        df = pd.read_csv(file_path, usecols=[0,1], names=['HJD','Mag_raw'], comment='#')
        df['HJD'] = pd.to_numeric(df['HJD'], errors='coerce')
        df['Upper'] = df['Mag_raw'].astype(str).str.startswith('<')
        df['Mag'] = pd.to_numeric(df['Mag_raw'].astype(str).str.replace('<',''), errors='coerce')
        df = df.dropna(subset=['HJD','Mag'])
        df['Days'] = df['HJD'] - self.tpeak
        det = df[~df['Upper']]
        self.plot_generic(det['Days'], det['Mag'], label='AAVSO')

    def analyze_csv(self, file_path):
        # Load CSV data and plot
        df = pd.read_csv(file_path)
        df['HJD'] = pd.to_numeric(df['HJD'], errors='coerce')
        df['mag'] = pd.to_numeric(df['mag'], errors='coerce')
        df['mag_err'] = pd.to_numeric(df['mag_err'], errors='coerce')
        df = df.dropna(subset=['HJD','mag','mag_err'])
        df['Days'] = df['HJD'] - self.tpeak
        self.plot_generic(df['Days'], df['mag'], df['mag_err'], label='CSV')

    def analyze_dat(self, file_path):
        # Load DAT data and plot
        arr = np.loadtxt(file_path)
        days = arr[:,0] - self.tpeak
        mags = arr[:,1]
        errs = arr[:,2]
        self.plot_generic(days, mags, errs, label='DAT')

    def analyze_multi(self, file_path):
        # Load multi-filter data and plot
        data = []
        with open(file_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        jd = float(parts[0]) + 2450000.0
                        filt = parts[1]
                        mag = float(parts[2])
                        err = float(parts[3])
                        data.append([jd, filt, mag, err])
                    except:
                        continue
        df = pd.DataFrame(data, columns=['JD','Filter','Mag','Err'])
        df['Days'] = df['JD'] - self.tpeak
        self.ax.clear(); self.ax.invert_yaxis()
        for filt in df['Filter'].unique():
            sub = df[df['Filter']==filt]
            self.ax.errorbar(sub['Days'], sub['Mag'], yerr=sub['Err'], fmt='o', label=filt)
        tmax_b, tmax_a = self.compute_uncertainty(*self.tmax_jds)
        t2_b, t2_a = self.compute_uncertainty(*self.t2_jds)
        t2_central = self.t2_jds[1] - self.tpeak
        self.ax.axvspan(-tmax_b, tmax_a, color='red', alpha=0.3, label='t_peak_unc')
        self.ax.axvspan(t2_central - t2_b, t2_central + t2_a, color='blue', alpha=0.3, label='t2_unc')
        self.ax.axvline(0, color='k', ls='--'); self.ax.axvline(t2_central, color='k', ls='--')
        self.ax.set_xlabel('Days since peak'); self.ax.set_ylabel('Magnitude')
        self.ax.set_title('Multi-Filter'); self.ax.legend()
        self.fig.tight_layout(); self.canvas.draw()

# --- GUI Setup ---
root = tk.Tk(); root.title("Nova Analysis Menu")

# Variables and frames
analyzer = None
controls = tk.Frame(root); controls.grid(row=0, column=0, sticky='nw')
plot_frame = tk.Frame(root); plot_frame.grid(row=0, column=1, rowspan=15)

# File type selection
tk.Label(controls, text="File type:").grid(row=0, column=0, sticky='w')
filetype_var = tk.StringVar(value="AAVSO")
tk.OptionMenu(controls, filetype_var, "AAVSO","CSV","DAT","MULTI-FILTER").grid(row=0, column=1)

# File browser
tk.Label(controls, text="Data file:").grid(row=1, column=0, sticky='w')
file_entry = tk.Entry(controls, width=40); file_entry.grid(row=1, column=1)
tk.Button(controls, text="Browse", command=lambda: file_entry.insert(0, filedialog.askopenfilename())).grid(row=1, column=2)

# t_peak and uncertainty inputs
tk.Label(controls, text="t_peak JD:").grid(row=2, column=0, sticky='w')
tpeak_entry = tk.Entry(controls, width=12); tpeak_entry.grid(row=2, column=1)

tk.Label(controls, text="t_max JD1,JD2,JD3:").grid(row=3, column=0, sticky='w')
tmax_jd1 = tk.Entry(controls, width=6); tmax_jd1.grid(row=3, column=1, sticky='w')
tmax_jd2 = tk.Entry(controls, width=6); tmax_jd2.grid(row=3, column=1)
tmax_jd3 = tk.Entry(controls, width=6); tmax_jd3.grid(row=3, column=1, sticky='e')

tk.Label(controls, text="t2 JD1,JD2,JD3:").grid(row=4, column=0, sticky='w')
t2_jd1 = tk.Entry(controls, width=6); t2_jd1.grid(row=4, column=1, sticky='w')
t2_jd2 = tk.Entry(controls, width=6); t2_jd2.grid(row=4, column=1)
t2_jd3 = tk.Entry(controls, width=6); t2_jd3.grid(row=4, column=1, sticky='e')

# Axis limits
tk.Label(controls, text="Xmin:").grid(row=5, column=0, sticky='w')
xmin_entry = tk.Entry(controls, width=6); xmin_entry.grid(row=5, column=1, sticky='w')
tk.Label(controls, text="Xmax:").grid(row=5, column=2, sticky='w')
xmax_entry = tk.Entry(controls, width=6); xmax_entry.grid(row=5, column=3)

tk.Label(controls, text="Ymin:").grid(row=6, column=0, sticky='w')
ymin_entry = tk.Entry(controls, width=6); ymin_entry.grid(row=6, column=1, sticky='w')
tk.Label(controls, text="Ymax:").grid(row=6, column=2, sticky='w')
ymax_entry = tk.Entry(controls, width=6); ymax_entry.grid(row=6, column=3)

# Matplotlib Figure and Canvas
fig = Figure(figsize=(6,4))
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack()
# Add interactive toolbar
toolbar = NavigationToolbar2Tk(canvas, plot_frame)
toolbar.update()
canvas._tkcanvas.pack()

# Save plot function
def save_plot():
    file = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG Image','*.png'),('PDF','*.pdf')])
    if file:
        fig.savefig(file)
        messagebox.showinfo("Saved", f"Plot saved to {file}")

# Analysis runner
def run_analysis():
    global analyzer
    try:
        tpeak = float(tpeak_entry.get())
        tmax = (float(tmax_jd1.get()),float(tmax_jd2.get()),float(tmax_jd3.get()))
        t2 = (float(t2_jd1.get()),float(t2_jd2.get()),float(t2_jd3.get()))
        path = file_entry.get()
        if not path: raise ValueError("No file selected")
        analyzer = NovaAnalyzer(tpeak, tmax, t2, fig, ax, canvas)
        ftype = filetype_var.get()
        if ftype=="AAVSO": analyzer.analyze_aavso(path)
        elif ftype=="CSV": analyzer.analyze_csv(path)
        elif ftype=="DAT": analyzer.analyze_dat(path)
        else: analyzer.analyze_multi(path)
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Update axis limits
def update_axes():
    try:
        ax.set_xlim(float(xmin_entry.get()), float(xmax_entry.get()))
        ax.set_ylim(float(ymax_entry.get()), float(ymin_entry.get()))
        canvas.draw()
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Buttons
tk.Button(controls, text="Run Analysis", command=run_analysis).grid(row=7, column=1, pady=5)
tk.Button(controls, text="Update Axes", command=update_axes).grid(row=8, column=1)
tk.Button(controls, text="Save Plot", command=save_plot).grid(row=9, column=1)

# Start GUI loop
root.mainloop()
