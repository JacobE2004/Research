import pandas as pd
import math
import numpy as np
import os
import re


file_path = r"C:\Users\Jmell\Dropbox\LMCN_Spreadsheet(in).csv"

# Check if the file exists before proceeding
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit()

try:
    df = pd.read_csv(file_path, encoding='latin1')
    print("File found. Proceeding...")
except Exception as e:
    print("Error reading CSV:", e)
    exit()

rename_map = {
    "t_peak Uncer (indays)": "t_peak",
    "distance": "distance"

}
# Rename columns based on the provided mapping
df.rename(columns=rename_map, inplace=True)

# Check required columns
required_columns = {"t_peak", "distance"}
if not required_columns.issubset(df.columns):
    print(f"Error: Missing required columns {required_columns - set(df.columns)}")
    exit()

# Clean t_peak values (remove ± and uncertainty)
def clean_value(val):
    if pd.isna(val):
        return np.nan
    return re.sub(r"\(±\).*", "", str(val))

df["t_peak"] = df["t_peak"].apply(clean_value)
df["distance"] = pd.to_numeric(df["distance"], errors='coerce')

# Define redshift
redshift = 0.113

# Compute absolute magnitude
def compute_absolute_mag(row):
    try:
        m = float(row['t_peak'])
        d = float(row['distance'])
        if d - 1 <= 0:
            return np.nan
        return (m - redshift) - 5 * math.log10(d - 1)
    except (ValueError, TypeError):
        return np.nan

df['computed_absolute_mag'] = df.apply(compute_absolute_mag, axis=1)

# Add Absolute peak mag column at 10th position if needed
cols = df.columns.tolist()
if "Absolute peak mag" not in df.columns:
    if len(cols) >= 10:
        cols.insert(10, 'Absolute peak mag')
        df['Absolute peak mag'] = df['computed_absolute_mag']
        df = df[cols]
    else:
        df["Absolute peak mag"] = df['computed_absolute_mag']
else:
    df["Absolute peak mag"] = df['computed_absolute_mag']

print("\nResults with Computed Absolute Magnitudes:")
print(df[['Name', 't_peak', 'distance', 'Absolute peak mag']].head(10))

# Save cleaned CSV
output_path = r"C:\Users\Jmell\Dropbox\LMCN_Spreadsheet_cleaned.csv"
df.to_csv(output_path, index=False, encoding='latin1')
print(f"\nCSV file updated successfully: {output_path}")