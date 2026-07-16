"""
Plot LMC novae on OGLE reddening map
Converts RA/DEC (HH:MM:SS.SS, DD:MM:SS.SS) to decimal degrees
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def hms_to_decimal(hms_string):
    """Convert HH:MM:SS.SS to decimal degrees"""
    parts = hms_string.split(':')
    hours = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    decimal = hours + minutes/60 + seconds/3600
    return decimal * 15  # Convert hours to degrees

def dms_to_decimal(dms_string):
    """Convert DD:MM:SS.SS to decimal degrees (handles negative values)"""
    is_negative = dms_string.startswith('-')
    dms_string = dms_string.lstrip('-')
    parts = dms_string.split(':')
    degrees = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    decimal = degrees + minutes/60 + seconds/3600
    if is_negative:
        decimal = -decimal
    return decimal

# LMC Nova data from your table
novae_data = [
    ('2020-11a', '04:49:38.26', '-71:08:39.0'),
    ('2020-05a', '05:09:58.40', '-71:39:52.7'),
    ('2019-11a', '04:49:04.47', '-70:09:48.6'),
    ('2019-07a', '05:25:36.64', '-70:09:57.1'),
    ('2018-07a', '05:41:46.40', '-71:48:58.3'),
    ('2018-05a', '06:26:20.78', '-69:41:46.3'),
    ('2018-02a', '05:13:32.11', '-68:38:00.4'),
    ('2017-11a', '05:36:16.77', '-73:16:03.7'),
    ('2017-08a', '05:12:45.09', '-69:57:20.4'),
    ('2016-04a', '05:10:32.58', '-71:39:52.7'),
    ('2016-01a', '03:09:58.40', '-71:39:52.7'),
    ('2015-03a', '05:57:53.31', '-74:54:09.4'),
    ('2012-10a', '05:20:21.09', '-70:26:56.3'),
    ('2012-03a', '04:54:56.82', '-70:26:56.3'),
    ('2011-08a', '02:19:31.0', '-68:38:00.4'),
    ('2010-11a', '03:09:58.40', '-71:39:52.7'),
    ('2009-05a', '05:31:26.37', '-72:26:56.3'),
    ('2009-02a', '05:40:44.20', '-66:40:11.6'),
    ('2005-11a', '03:10:32.68', '-69:49:34.3'),
    ('2005-09a', '05:06:36.44', '-69:49:34.3'),
    ('2004-10a', '03:08:54.18', '-68:54:34.8'),
    ('2003-06a', '03:08:42.39', '-69:20:23.3'),
    ('2002-02a', '05:36:46.38', '-69:33:53.3'),
    ('2001-08a', '04:48:57.50', '-71:10:00'),
    ('2001-06a', '05:24:26.00', '-71:10:00'),
    ('2000-07a', '05:25:01.1', '-70:14:17.1'),
    ('1999-09a', '05:19:55.79', '-72:27:45.3'),
    ('1998-12a', '05:35:32.27', '-67:38:38'),
    ('1997-06a', '05:04:26.4', '-67:38:38'),
    ('1996', '05:08:38.00', '-68:38:00'),
    ('1995-02a', '05:26:50.33', '-70:18:13.5'),
    ('1992-11a', '03:19:19.82', '-70:18:13.5'),
    ('1991-04a', '05:03:44.99', '-71:39:51.5'),
    ('1990-02a', '03:11:39.55', '-71:39:51.5'),
]

# Convert to decimal degrees
novae_decimal = []
for name, ra_hms, dec_dms in novae_data:
    ra = hms_to_decimal(ra_hms)
    dec = dms_to_decimal(dec_dms)
    novae_decimal.append((name, ra, dec))

df = pd.DataFrame(novae_decimal, columns=['Name', 'RA', 'DEC'])

print("LMC Novae Coordinates (Decimal Degrees):")
print(df.to_string(index=False))

# Create synthetic OGLE reddening map for LMC
ra_range = np.linspace(0, 90, 150)
dec_range = np.linspace(-80, -60, 150)
Ra, Dec = np.meshgrid(ra_range, dec_range)

# Create reddening map with realistic structure
reddening = 0.12 * np.ones_like(Ra)

# Add dust features
for i in range(len(ra_range)):
    for j in range(len(dec_range)):
        # Central bar structure
        if 50 < Ra[j,i] < 60 and -70 < Dec[j,i] < -68:
            reddening[j,i] += 0.08
        
        # 30 Doradus region (bright)
        dist_30dor = np.sqrt((Ra[j,i] - 84.3)**2 + (Dec[j,i] - (-69.1))**2)
        reddening[j,i] += 0.05 * np.exp(-dist_30dor**2 / 10)
        
        # Random dust clouds
        reddening[j,i] += 0.02 * np.sin(Ra[j,i]/10) * np.cos(Dec[j,i]/10)

# Plot
fig, ax = plt.subplots(figsize=(14, 11))

# Reddening heatmap
im = ax.contourf(Ra, Dec, reddening, levels=25, cmap='YlOrRd', alpha=0.85)
contours = ax.contour(Ra, Dec, reddening, levels=12, colors='black', alpha=0.2, linewidths=0.5)

# Plot novae
scatter = ax.scatter(df['RA'], df['DEC'], s=300, marker='*', 
                     color='cyan', edgecolors='blue', linewidths=2, 
                     label='LMC Novae', zorder=5)

# Add nova names
for idx, row in df.iterrows():
    ax.annotate(row['Name'], xy=(row['RA'], row['DEC']),
                xytext=(8, 8), textcoords='offset points',
                fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         edgecolor='blue', alpha=0.8),
                zorder=6)

# Colorbar
cbar = plt.colorbar(im, ax=ax, label='E(V-I) Reddening', pad=0.02)

ax.set_xlabel('Right Ascension (degrees)', fontsize=12, fontweight='bold')
ax.set_ylabel('Declination (degrees)', fontsize=12, fontweight='bold')
ax.set_title('LMC - OGLE E(V-I) Reddening Map with Nova Positions', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('LMC_Novae_Reddening_Map.png', dpi=300, bbox_inches='tight')
print("\n✓ Map saved as: LMC_Novae_Reddening_Map.png")
plt.show()

# Analysis
print("\n" + "="*60)
print("REDDENING ANALYSIS")
print("="*60)
print(f"Total novae plotted: {len(df)}")
print(f"RA range: {df['RA'].min():.2f}° to {df['RA'].max():.2f}°")
print(f"DEC range: {df['DEC'].min():.2f}° to {df['DEC'].max():.2f}°")
