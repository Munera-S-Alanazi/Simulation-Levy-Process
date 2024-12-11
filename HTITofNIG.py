import cmath
import numpy as np
import matplotlib.pyplot as plt
import cmath
from ParametersOfNIG import f, df_dz, find_K, find_M_h, search_x0_xK, compute_expression, compute_norms, find_M_h_diff

# The characteristic function of NIG process as in Cont & Tankov book
def nigCF(t, z, theta, sigma, kappa):
    i = complex(0.0, 1.0)  # Define the imaginary unit
    term1 = 1.0 / kappa
    term2 = 1.0 - 2.0 * i * z * theta * kappa + kappa * sigma**2 * z**2
    return cmath.exp(t * (term1 - term1 * cmath.sqrt(term2)))


# Step 1: Implement the Hilbert Transform to approximate the CDF
def hilbert_transform_cdf(x, t, theta, sigma, kappa, h, M):
    # Discretized Hilbert Transform summation
    summation = 0
    for m in range(-M, M + 1):
        z = (m - 0.5) * h
        term = np.exp(-1j * z * x) * nigCF(t, z, theta, sigma, kappa) / (((m - 0.5) * np.pi))
        summation += term

    # Final calculation for the CDF approximation with corrected sign
    F_approx = 0.5 + summation * (1j/2.0)
    return F_approx.real

# Step 2: tabulate the CDF using Hilbert Transform
def tabulate_cdf_using_hilbert(t, theta, sigma, kappa, x_min, x_max, K, h, M):
    x_values = np.linspace(x_min, x_max, K + 1)
    cdf_values = [hilbert_transform_cdf(x, t, theta, sigma, kappa, h, M) for x in x_values]
    # ensure CDF is monotonically increasing
    cdf_values = np.maximum.accumulate(cdf_values)
    return x_values, cdf_values

# Step 3: inverse Transform Sampling with Tabulated CDF
def inverse_transform_sampling(cdf_values, x_values, U):
    # Initialize the interval index
    k = 0 # initial step
    K = len(cdf_values) - 1 # K

    # find the correct interval using a loop (binary search)
    while k < K and cdf_values[k + 1] <= U:
        k += 1

    # if U is out of bounds, return the boundary value
    if U <= cdf_values[0]:
        return x_values[0]
    elif U >= cdf_values[K]:
        return x_values[K]

    # Linear interpolation to approximate the inverse CDF
    x_k = x_values[k]
    x_k1 = x_values[k + 1]
    F_k = cdf_values[k]
    F_k1 = cdf_values[k + 1]
    return x_k + (x_k1 - x_k) * (U - F_k) / (F_k1 - F_k)

# Validate the real component of the characteristic function
num_samples = 1000000 #number of uniform random samples for inverse transform sampling
theta = 0.2 #Drift
sigma = 0.2 #Volatility
kappa = 0.1 #Variance rate 
t = 4.0
c = t * (sigma / np.sqrt(kappa))
decay_kappa = np.exp(t / kappa)
nu = 1 
d_plus =  (theta*kappa +np.sqrt((theta**2)*(kappa**2) +(kappa *sigma**2)))/ (kappa*sigma**2)
d_minus = (theta*kappa - np.sqrt((theta**2)*(kappa**2) +(kappa *sigma**2)))/ (kappa*sigma**2)
norm_plus, norm_minus = compute_norms(nigCF, d_plus, d_minus, t, theta, sigma, kappa)
#print(f"norm + = {norm_plus}")
#print(f"norm - = {norm_minus}")
zValues = np.linspace(0, 4, 50)
# Generate uniform random samples and transform them\
uniform_samples = np.random.rand(num_samples)
# Theoretical characteristic function (real part only)
 # find optimal M and h for the given epsilon
n_f = 0  # Number of non-differentiable points
h_vals = np.linspace(5, 0.01, 40)  # Array of h values to try
M_vals = np.arange(1, 200, 1)  # Array of M values to try
    
cf_real_theoretical = [nigCF(t, z, theta, sigma, kappa) for z in zValues]
real_parts_theoretical = [val.real for val in cf_real_theoretical]

# Define the epsilon values to test
epsilon_values = [1e-2, 1e-3, 1e-4]
x_min_values = []
x_max_values = []
K_values = []
M_values = []
h_values = []

if __name__ == "__main__":
# Loop over each epsilon value
 for epsilon in epsilon_values:
    errors = []
    print(f"\nTesting for epsilon = {epsilon}")
    # find x_min, x_max, and K for the given epsilon
    x_min, x_max = search_x0_xK(f, nigCF, d_plus, d_minus, t, theta, sigma, kappa, epsilon)
    x_min_values.append(x_min)
    x_max_values.append(x_max)
    
    K = find_K(df_dz, nigCF, x_min, x_max, epsilon, t, theta, sigma, kappa)
    K_values.append(K)
    
    print(f"x_min: {x_min}, x_max: {x_max}")
    print(f"K: {K}")
    

    # call the function to find M and h
    #M, h = find_M_h(f, df_dz, compute_expression, x_min, x_max, t, theta, sigma, kappa, 
                    #d_minus, d_plus, norm_minus, norm_plus, K, n_f, epsilon, h_vals, M_vals, nu, c, decay_kappa)
    M, h = find_M_h_diff(compute_expression, x_min, x_max, d_minus, d_plus, norm_minus, norm_plus, K, epsilon, h_vals, M_vals, nu, c, decay_kappa)

    M_values.append(M)
    h_values.append(h)
    # Display M and h for this epsilon
    print(f"Found M: {M}, h: {h}")
    x_values, cdf_values = tabulate_cdf_using_hilbert(t, theta, sigma, kappa, x_min, x_max, K, h, M)
    samples = [inverse_transform_sampling(cdf_values, x_values, U) for U in uniform_samples]
    # Initialize arrays for simulated characteristic function real part and std deviation
    cf_real_sim = np.zeros_like(zValues)
    sd_real = np.zeros_like(zValues)
    for i, z in enumerate(zValues):
        cos_values = np.cos(z * np.array(samples))
        
        # Monte Carlo estimate for the real part of the characteristic function
        cf_real_sim[i] = np.mean(cos_values)
        # Standard deviation ofi the estimat
        sd_real[i] = np.sqrt((np.mean(cos_values**2) - cf_real_sim[i]**2) / num_samples)
    # Print the values
    print(f"Optimal parameters used: M = {M}, h = {h}, x_min = {x_min}, x_max = {x_max}")
    print("\nComparison between simulated and theoretical characteristic function:")
    for i, z in enumerate(zValues):
        print(f"z = {z:.2f}: Simulated = {cf_real_sim[i]:.6f}, Theoretical = {cf_real_theoretical[i].real:.6f}, "
              f"Abs Error = {np.abs(cf_real_sim[i] - cf_real_theoretical[i].real):.6f}")
        errors.append(np.abs(cf_real_sim[i] - cf_real_theoretical[i].real))
    print(f"max error = {max(errors)}")
    # Plot real component comparison
    plt.figure()
    plt.plot(zValues, cf_real_sim, label="Simulated Real Part")
    plt.fill_between(
    zValues,
    cf_real_sim - 3 * sd_real,
    cf_real_sim + 3 * sd_real,
    color="red",
    alpha=0.2,  # Adjust alpha to control transparency
    label="Simulated Real Part (with error)")
    plt.plot(zValues, real_parts_theoretical, label="Theoretical Real Part", linestyle="--")
    plt.title(f"Real Component of Characteristic Function of NIG\n: x_min = {x_min:.4f}, x_max = {x_max:.4f}, K = {K}, h = {h:.4f}, M = {M}")
    plt.legend(['Simulated Real Part', 'Simulated Real Part (with error)', 'Theoretical Real Part'])   
    # Plot absolute error for real part with 3 std deviation bounds
    plt.figure()
    plt.plot(zValues, np.abs(cf_real_sim - real_parts_theoretical), label="Absolute Error (Real)")
    plt.plot(zValues, 3 * sd_real, linestyle="--", label="3 Std Dev (Real)")
    plt.title(f"Error in Real Component of Characteristic Function of VG\n: x_min = {x_min:.4f}, x_max = {x_max:.4f}, K = {K}, h = {h:.4f}, M = {M}")
    plt.legend(['absolute error', '3 std dev'])
    plt.show()
