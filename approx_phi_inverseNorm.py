import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import bisect
from hilbert_transform_cdf_derivative import hilbert_transform_cdf_derivative

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


# compute slope and intercept for piecewise linear approximation using the Hilbert transform-based inverse CDF.
def compute_L2_linear_coefficients(num_intervals, mu, sigma, h, M):

    r = 0.5  # shrinking factor
    u_vals = [r ** k for k in range(num_intervals + 1)]  # Interval edges
    slopes = []
    intercepts = []
    
    for i in range(num_intervals):
        u1, u2 = u_vals[i], u_vals[i + 1]
        
        # Numerical integrals using the inverse CDF approximation
        integral_Phi = quad(lambda u: inverse_cdf_newton(u, mu, sigma, h, M), u1, u2)[0]
        integral_u_Phi = quad(lambda u: u * inverse_cdf_newton(u, mu, sigma, h, M), u1, u2)[0]
        integral_u2 = quad(lambda u: u**2, u1, u2)[0]
        
        # Compute slope (a) and intercept (b)
        midpoint = (u1 + u2) / 2
        slope = (integral_u_Phi - midpoint * integral_Phi) / (integral_u2 - midpoint**2 * (u2 - u1))
        intercept = (1 / (u2 - u1)) * integral_Phi - slope * midpoint
        
        slopes.append(slope)
        intercepts.append(intercept)
    
    return np.array(intercepts), np.array(slopes)

mu, sigma = 0, 1  # Standard normal parameters
h, M = 0.1, 50    # Hilbert transform parameters
num_intervals = 8

intercepts, slopes = compute_L2_linear_coefficients(num_intervals, mu, sigma, h, M)
print("Intercepts:", intercepts)
print("Slopes:", slopes)


#################################

def piecewise_linear_approximation_code2(u_vals, intercepts, slopes):

    results = []
    table_size = len(intercepts)
    for u in u_vals:
        reflect = False

        # Reflect u about 0.5 for simplicity
        if u >= 0.5:
            u = 1.0 - u
            reflect = True

        # Determine the interval index
        interval = max(0, int(np.floor(-np.log2(u))))
        interval = min(interval, table_size - 1)  # cap index to table size

        # Approximate z value using linear coefficients
        z = intercepts[interval] + slopes[interval] * u

        # Reflect the value if necessary
        if reflect:
            z = -z

        results.append(z)
    return np.array(results)


# number of intervals for the piecewise linear approximation
num_intervals = 8

# generate uniform samples for testing
u_samples = np.random.uniform(0, 1, 1000)

# compute piecewise linear approximation
z_approx = piecewise_linear_approximation_code2(u_samples, intercepts, slopes)

# compare to the exact Gaussian inverse CDF
z_exact = norm.ppf(u_samples)

#Plotting the results
plt.figure(figsize=(12, 6))
plt.scatter(u_samples, z_exact, label="Exact $\Phi^{-1}(u)$", alpha=0.5, s=10)
plt.scatter(u_samples, z_approx, label="Piecewise Linear Approximation", alpha=0.2, s=5, color="red")

plt.legend()
plt.xlabel("Uniform Samples ($u$)")
plt.ylabel("Approximation / Exact ($z$)")
plt.title("Piecewise Linear Approximation of Gaussian Inverse CDF")
plt.grid()
plt.show()

r = 0.5
epsilon = 1e-10
u_vals = [r ** k for k in range(num_intervals + 1)]


# plot exact values
plt.figure(figsize=(12, 6))
plt.scatter(u_samples, z_exact, label="Exact $\Phi^{-1}(u)$", alpha=0.4, s=5)

# Plot approximations interval by interval for \( (0, 0.5] \) and reflections
colors = plt.cm.viridis(np.linspace(0, 1, num_intervals))  # Assign unique colors to each interval
for i in range(num_intervals):
    u1, u2 = u_vals[i], u_vals[i + 1]
    slope, intercept = slopes[i], intercepts[i]
    
    # skip intervals outside \( (0, 0.5] \)
    if u2 >= 0.5:
        continue

    # generate points for the interval in \( (0, 0.5] \)
    u_interval = np.linspace(u1, u2, 100)
    z_approx_interval = slope * u_interval + intercept

    # plot the interval's approximation for \( (0, 0.5] \)
    plt.plot(u_interval, z_approx_interval, color=colors[i], label=f"Interval {i+1}: [{u1:.6f}, {u2:.6f}]")

    # plot the mirrored interval's approximation for \( (0.5, 1] \)
    reflected_u_interval = 1 - u_interval
    reflected_z_approx = -z_approx_interval
    plt.plot(reflected_u_interval, reflected_z_approx, color=colors[i], linestyle="--")

plt.xlabel("Uniform Samples ($u$)")
plt.ylabel("Approximation / Exact ($z$)")
plt.title("Piecewise Linear Approximation Using Symmetric Intervals")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.tight_layout()
plt.show()

