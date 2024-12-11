import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.integrate import quad
from hilbert_transform_cdf_derivative import hilbert_transform_cdf_derivative
from scipy.interpolate import CubicSpline

# the characteristic function of the normal distribution
def characteristic_function_normal(z, mu, sigma):
    return np.exp(1j * mu * z - 0.5 * (sigma ** 2) * (z ** 2))

"""def hilbert_transform_cdf(x, mu, sigma, h, M):
    # Discretized Hilbert Transform summation
    summation = 0
    for m in range(-M, M + 1):
        z = (m - 0.5) * h
        term = np.exp(-1j * z * x) * characteristic_function_normal(z, mu, sigma) / (((m - 0.5) * np.pi))
        summation += term

    # Final calculation for the CDF approximation with corrected sign
    F_approx = 0.5 + summation * (1j/2.0)
    return F_approx.real"""

def hilbert_transform_cdf(x, mu, sigma, h, M):
    # Discretized Hilbert Transform summation
    summation = 0
    for m in range(1, M+1):
        z = (m - 0.5) * h
        term = np.exp(-1j * z * x) * characteristic_function_normal(z, mu, sigma) / (((m - 0.5) * np.pi))
        summation += term

    # Final calculation for the CDF approximation with corrected sign
    F_approx = 0.5 - summation.imag
    return F_approx

"""# find the inverse CDF using root-finding (bisection)
def inverse_cdf_hilbert_bis(u, mu, sigma, h, M, x_min=-10, x_max=10):
   
    # define the equation F(x) - u = 0
    def equation(x):
        return hilbert_transform_cdf(x, mu, sigma, h, M) - u
    
    # use bisection to solve the equation
    return bisect(equation, x_min, x_max)"""
#    Find the inverse CDF using Newton's method.

def inverse_cdf_newton(u, mu, sigma, h, M, x0=0, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        # Compute the value of the function and its derivative
        fx = hilbert_transform_cdf(x, mu, sigma, h, M) - u
        f_prime_x = hilbert_transform_cdf_derivative(x, mu, sigma, h, M)
        
        # Check if the derivative is zero to prevent division by zero
        if np.abs(f_prime_x) < 1e-10:
            raise ValueError(f"Derivative too small at iteration {i}, x = {x}")
        
        # Update the value of x
        x_new = x - fx / f_prime_x
        
        # Check for convergence
        if np.abs(x_new - x) < tol:
            return x_new
        
        x = x_new
    
    raise ValueError("Newton's method did not converge within the maximum number of iterations")



mu, sigma = 0, 1  # Standard normal parameters
h, M = 0.1, 50    # Hilbert transform parameters
num_intervals = 8



#################################

# Approximate using cubic spline interpolation.
def piecewise_cubic_spline_approximation(u_vals, Phi_inverse_vals, u_samples):

    # Create the cubic spline for (0, 0.5] using u_vals and their inverse CDF values
    spline = CubicSpline(u_vals, Phi_inverse_vals)

    results = []
    for u in u_samples:
        reflect = False

        # Reflect u about 0.5 for simplicity
        if u > 0.5:
            u = 1.0 - u
            reflect = True

        # Evaluate the cubic spline
        z = spline(u)

        # Reflect the value if necessary
        if reflect:
            z = -z

        results.append(z)

    return np.array(results)

# Define the interval edges for (0, 0.5]
r = 0.5
epsilon = 1e-10
u_vals = [r ** k for k in range(num_intervals + 1)][::-1] 

# Compute the exact inverse CDF values for these edges
Phi_inverse_vals = [inverse_cdf_newton(u, mu, sigma, h, M) for u in u_vals]

# Generate uniform samples for testing
u_samples = np.random.uniform(0, 1, 1000)

# Compute piecewise cubic spline approximation
z_approx = piecewise_cubic_spline_approximation(u_vals, Phi_inverse_vals, u_samples)

# Compare to the exact Gaussian inverse CDF
z_exact = norm.ppf(u_samples)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.scatter(u_samples, z_exact, label="Exact $\Phi^{-1}(u)$", alpha=0.5, s=10)
plt.scatter(u_samples, z_approx, label="Piecewise Cubic Spline Approximation", alpha=0.2, s=5, color="red")

plt.legend()
plt.xlabel("Uniform Samples ($u$)")
plt.ylabel("Approximation / Exact ($z$)")
plt.title("Piecewise Cubic Spline Approximation of Gaussian Inverse CDF")
plt.grid()
plt.show()

# Plot exact values
plt.figure(figsize=(12, 6))
plt.scatter(u_samples, z_exact, label="Exact $\Phi^{-1}(u)$", alpha=0.4, s=5)

# Plot the cubic spline approximations for \( (0, 0.5] \) and reflections
colors = plt.cm.viridis(np.linspace(0, 1, len(u_vals) - 1))  # Assign unique colors to each interval
for i in range(len(u_vals) - 1):
    u1, u2 = u_vals[i], u_vals[i + 1]
    
    # Skip intervals outside \( (0, 0.5] \)
    if u2 > 0.5:
        continue

    # Generate points for the interval in \( (0, 0.5] \)
    u_interval = np.linspace(u1, u2, 100)
    z_approx_interval = piecewise_cubic_spline_approximation(u_vals, Phi_inverse_vals, u_interval)

    # Plot the interval's approximation for \( (0, 0.5] \)
    plt.plot(u_interval, z_approx_interval, color=colors[i], label=f"Interval {i+1}: [{u1:.6f}, {u2:.6f}]")

    # Plot the mirrored interval's approximation for \( (0.5, 1] \)
    reflected_u_interval = 1 - u_interval
    reflected_z_approx = -z_approx_interval
    plt.plot(reflected_u_interval, reflected_z_approx, color=colors[i], linestyle="--")

# Add labels, legend, and title
plt.xlabel("Uniform Samples ($u$)")
plt.ylabel("Approximation / Exact ($z$)")
plt.title("Piecewise Cubic Spline Approximation Using Symmetric Intervals")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.tight_layout()
plt.show()
