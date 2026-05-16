import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data
# -----------------------------
f = np.array([100, 90, 80, 70, 60, 50], dtype=float)  # RPS
eps = np.array([34.75, 29.62, 16.65, 13.71,  5.21,  1.34], dtype=float)  # mV

# y-uncertainty (1σ): EMF standard deviations (mV)
sig_eps = np.array([3.2401132, 1.4226735, 1.3492796, 2.7322966, 1.3971797, 0.7183314], dtype=float)

# x-uncertainty (1σ): 5.5% of each RPS
sig_f = 0.055 * f

# -----------------------------
# Helper: weighted line fit + covariance
# Model: y = m x + b
# -----------------------------
def weighted_line_fit_with_cov(x, y, sy):
    # Design matrix X = [x, 1]
    X = np.column_stack([x, np.ones_like(x)])
    W = np.diag(1.0 / sy**2)

    # Normal equation pieces
    XT_W = X.T @ W
    cov = np.linalg.inv(XT_W @ X)      # Covariance of [m, b]
    beta = cov @ (XT_W @ y)            # [m, b]

    m, b = beta
    sig_m = np.sqrt(cov[0, 0])
    sig_b = np.sqrt(cov[1, 1])
    return m, b, sig_m, sig_b, cov

# -----------------------------
# 1) First fit using only y-uncertainties
# -----------------------------
m, b, sig_m, sig_b, cov = weighted_line_fit_with_cov(f, eps, sig_eps)

# -----------------------------
# 2) Include x-uncertainty via effective y-uncertainty and refit (iterate once)
# sigma_eff^2 = sigma_eps^2 + (m*sigma_f)^2
# -----------------------------
sig_eff = np.sqrt(sig_eps**2 + (m * sig_f)**2)
m, b, sig_m, sig_b, cov = weighted_line_fit_with_cov(f, eps, sig_eff)

# Recompute effective uncertainties with updated slope (for chi^2)
sig_eff = np.sqrt(sig_eps**2 + (m * sig_f)**2)

# -----------------------------
# Chi-squared
# -----------------------------
resid = eps - (m * f + b)
chi2 = np.sum((resid / sig_eff)**2)
dof = len(f) - 2
chi2_red = chi2 / dof

print("Best-fit model: eps = m f + b (eps in mV, f in RPS)")
print(f"m = {m:.6f} ± {sig_m:.6f}  mV/RPS")
print(f"b = {b:.6f} ± {sig_b:.6f}  mV")
print(f"Chi^2 = {chi2:.4f},  DoF = {dof},  Reduced Chi^2 = {chi2_red:.4f}")

# -----------------------------
# Shaded uncertainty band for the fitted LINE (mean prediction band)
# Var(yhat) = [x,1] Cov [x,1]^T
# -----------------------------
f_fine = np.linspace(f.min() - 5, f.max() + 5, 300)
Xfine = np.column_stack([f_fine, np.ones_like(f_fine)])

# variance of fitted mean at each x
var_yhat = np.einsum("ij,jk,ik->i", Xfine, cov, Xfine)  # efficient diag(X Cov X^T)
sig_yhat = np.sqrt(var_yhat)

eps_fit = m * f_fine + b

# Choose how wide band should be:
k_sigma = 1.0  # 1σ band (set to 2.0 for ~95%ish if you want)
upper = eps_fit + k_sigma * sig_yhat
lower = eps_fit - k_sigma * sig_yhat

# -----------------------------
# Plot
# -----------------------------
plt.errorbar(f, eps, xerr=sig_f, yerr=sig_eps, fmt='o', capsize=4,
             label='Data (xerr=5.5%, yerr=std dev)')

plt.plot(f_fine, eps_fit, label=f'Fit: ε = ({m:.3f}±{sig_m:.3f}) f + ({b:.2f}±{sig_b:.2f})')

plt.fill_between(f_fine, lower, upper, alpha=0.25,
                 label=f'{k_sigma:.0f}σ fit band (line uncertainty)')

plt.xlabel('Rotation rate f (RPS)')
plt.ylabel('EMF ε (mV)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()