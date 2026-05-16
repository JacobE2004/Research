import numpy as np

# Frequencies read from your 10 scope snapshots (Hz)
f_hz = np.array([8333, 9554,  9615,  9868, ], dtype=float)

# Pulses per revolution (from your ~100 rps cluster estimate)
N = 97

f_mean = np.mean(f_hz)
f_std  = np.std(f_hz, ddof=1)  # sample std dev

rps_mean = f_mean / N
rps_std  = f_std / N

print("Mean frequency (Hz):", f_mean)
print("Std frequency (Hz): ", f_std)
print("Mean RPS:", rps_mean)
print("Std RPS: ", rps_std)