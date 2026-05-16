import sympy as sp

# Define the variable
theta = sp.symbols('theta')

# Compute the integral of sin(2theta) from 0 to pi
integral_theta = sp.integrate(sp.sin(2*theta), (theta, 0, sp.pi))

# Final result: (1/4) * (2pi) * (1/2) * integral_theta
result = (1/4) * (2*sp.pi) * (1/2) * integral_theta
result.evalf()