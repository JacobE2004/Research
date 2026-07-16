import pandas as pd
import numpy as np

def calculate_absolute_magnitude(apparent_mag, distance_pc, error=None):
    """
    Calculate absolute magnitude using: M = m - 5*log10(distance) + 5

    Returns tuple: (M_value, M_lower, M_upper) if error provided, else just M_value
    """
    if pd.isna(apparent_mag) or pd.isna(distance_pc):
        if error is not None:
            return np.nan, np.nan, np.nan
        return np.nan

    try:
        if isinstance(distance_pc, str):
            distance_pc = float(distance_pc.replace(',', ''))
        else:
            distance_pc = float(distance_pc)

        apparent_mag = float(apparent_mag)
        absolute_mag = apparent_mag - 5 * np.log10(distance_pc) + 5

        if error is not None and not pd.isna(error):
            error = float(error)
            abs_mag_lower = absolute_mag - error
            abs_mag_upper = absolute_mag + error
            return round(absolute_mag, 3), round(abs_mag_lower, 3), round(abs_mag_upper, 3)

        return round(absolute_mag, 3)
    except (ValueError, TypeError):
        if error is not None:
            return np.nan, np.nan, np.nan
        return np.nan

def calculate_novae_absolute_magnitude(input_csv, output_csv,
                                       mag_col='peak mag',
                                       distance_col='distance(pc)',
                                       error_col='Error (±)',
                                       output_col='Absolute peak mag'):
    """
    Calculate absolute magnitude for novae data with error bounds.

    Parameters:
    -----------
    input_csv : str
        Path to input CSV file
    output_csv : str
        Path to save output CSV file
    mag_col : str
        Column name for apparent magnitude (default: 'peak mag')
    distance_col : str
        Column name for distance in parsecs (default: 'distance(pc)')
    error_col : str
        Column name for magnitude error (default: 'Error (±)')
    output_col : str
        Column name for output absolute magnitude (default: 'Absolute peak mag')
    """
    df = pd.read_csv(input_csv, encoding='latin-1')

    results = df.apply(
        lambda row: calculate_absolute_magnitude(
            row[mag_col],
            row[distance_col],
            row[error_col] if error_col in df.columns else None
        ),
        axis=1
    )

    if error_col in df.columns:
        df[output_col] = results.apply(lambda x: x[0] if isinstance(x, tuple) else x)
        df[f'{output_col} (lower)'] = results.apply(lambda x: x[1] if isinstance(x, tuple) else np.nan)
        df[f'{output_col} (upper)'] = results.apply(lambda x: x[2] if isinstance(x, tuple) else np.nan)
    else:
        df[output_col] = results

    df.to_csv(output_csv, index=False)

    print(f"✓ Absolute magnitude calculated and saved to: {output_csv}")
    print(f"\nSample results:")
    display_cols = [mag_col, distance_col, output_col]
    if error_col in df.columns:
        display_cols.extend([f'{output_col} (lower)', f'{output_col} (upper)'])
    print(df[[col for col in display_cols if col in df.columns]].head(10))
    print(f"\nTotal rows processed: {len(df)}")
    print(f"Rows with calculated magnitude: {df[output_col].notna().sum()}")

def calculate_absolute_magnitude_interactive(apparent_mag, distance_pc, reddening=0, error=None):
    """
    Interactive calculator for absolute magnitude.
    
    M = (m - A_v) - 5*log10(d) + 5
    
    Parameters:
    -----------
    apparent_mag : float
        Apparent magnitude (m)
    distance_pc : float
        Distance in parsecs
    reddening : float
        Extinction/reddening correction A_v (default: 0)
    error : float
        Uncertainty in magnitude (optional)
    
    Returns:
    --------
    tuple or float: (M_value, M_lower, M_upper) if error provided, else M_value
    """
    if pd.isna(apparent_mag) or pd.isna(distance_pc):
        if error is not None:
            return np.nan, np.nan, np.nan
        return np.nan
    
    try:
        apparent_mag = float(apparent_mag)
        distance_pc = float(distance_pc)
        reddening = float(reddening) if reddening else 0
        
        # Apply reddening correction
        corrected_mag = apparent_mag - reddening
        
        # Calculate absolute magnitude
        absolute_mag = corrected_mag - 5 * np.log10(distance_pc) + 5
        
        if error is not None and not pd.isna(error):
            error = float(error)
            abs_mag_lower = absolute_mag - error
            abs_mag_upper = absolute_mag + error
            return round(absolute_mag, 3), round(abs_mag_lower, 3), round(abs_mag_upper, 3)
        
        return round(absolute_mag, 3)
    except (ValueError, TypeError) as e:
        print(f"Error calculating absolute magnitude: {e}")
        if error is not None:
            return np.nan, np.nan, np.nan
        return np.nan

def interactive_input_calculator():
    """
    Interactive menu for calculating absolute magnitude by inputting values.
    """
    print("\n" + "="*60)
    print("ABSOLUTE MAGNITUDE CALCULATOR")
    print("="*60)
    print("\nFormula: M = (m - A_v) - 5*log10(d) + 5")
    print("  where: m = apparent magnitude")
    print("         A_v = extinction/reddening correction")
    print("         d = distance in parsecs")
    print("="*60 + "\n")
    
    while True:
        try:
            # Get apparent magnitude
            apparent_mag = float(input("Enter apparent magnitude (m): "))
            
            # Get distance in parsecs
            distance_pc = float(input("Enter distance (parsecs): "))
            
            # Get reddening (optional)
            reddening_input = input("Enter extinction/reddening A_v [default=0]: ").strip()
            reddening = float(reddening_input) if reddening_input else 0
            
            # Get error (optional)
            error_input = input("Enter magnitude error/uncertainty [optional, press Enter to skip]: ").strip()
            error = float(error_input) if error_input else None
            
            # Calculate
            result = calculate_absolute_magnitude_interactive(apparent_mag, distance_pc, reddening, error)
            
            # Display results
            print("\n" + "-"*60)
            print("RESULTS:")
            print("-"*60)
            print(f"  Apparent magnitude (m):     {apparent_mag}")
            print(f"  Distance (pc):              {distance_pc}")
            print(f"  Extinction/reddening (A_v): {reddening}")
            
            if error is not None:
                abs_mag, lower, upper = result
                print(f"\n  Absolute Magnitude (M):     {abs_mag}")
                print(f"  Lower bound:                {lower}")
                print(f"  Upper bound:                {upper}")
                print(f"  Error range:                {lower} to {upper}")
            else:
                print(f"\n  Absolute Magnitude (M):     {result}")
            
            print("-"*60 + "\n")
            
            # Ask if user wants to calculate another
            again = input("Calculate another? (y/n): ").strip().lower()
            if again not in ['y', 'yes']:
                print("\nThank you for using the Absolute Magnitude Calculator!")
                break
                
        except ValueError as e:
            print(f"\n⚠ Invalid input: {e}")
            print("Please enter numeric values.\n")
        except Exception as e:
            print(f"\n⚠ Unexpected error: {e}")
            print("Please try again.\n")

if __name__ == "__main__":
    # Uncomment the line below to use the interactive calculator
    # interactive_input_calculator()
    
    # Or use the CSV batch processing below:
    input_file = r"C:\Users\Jmell\Dropbox\Research File\LMCN_Spreadsheet 2025-10-16 (1)(in).csv"
    output_file = r"C:\Users\Jmell\Dropbox\LMCN_Spreadsheet_with_absolute_mag.csv"

    calculate_novae_absolute_magnitude(input_file, output_file)
