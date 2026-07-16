import numpy as np

def calculate_absolute_magnitude(apparent_mag, distance_pc, reddening=0):
    """
    Calculate absolute magnitude: M = (m - A_v) - 5*log10(d) + 5
    """
    apparent_mag = float(apparent_mag)
    distance_pc = float(distance_pc)
    reddening = float(reddening) if reddening else 0
    
    # Apply reddening correction and calculate
    corrected_mag = apparent_mag - reddening
    absolute_mag = corrected_mag - 5 * np.log10(distance_pc) + 5
    


# Input values for each nova
apparent_magnitude = 11.296
distance_parsecs = 50000
reddening = 0.348


# Calculate
result = calculate_absolute_magnitude(apparent_magnitude, distance_parsecs, reddening)

# Print result
if isinstance(result, tuple):
    abs_mag, lower, upper = result
    print(f"\nAbsolute Magnitude: {abs_mag}")
    print(f"Range: {lower} to {upper}")
else:
    print(f"\nAbsolute Magnitude: {result}")

