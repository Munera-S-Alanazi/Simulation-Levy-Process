import numpy as np
import matplotlib.pyplot as plt
from hilbert_transform import hilbert_transform_cdf

# Step 2: tabulate the CDF using Hilbert Transform
def tabulate_cdf_using_hilbert(mu, sigma, x_min, x_max, K, h, M):
    x_values = np.linspace(x_min, x_max, K + 1)
    cdf_values = [hilbert_transform_cdf(x, mu, sigma, h, M) for x in x_values]
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

# Run the test
if __name__ == "__main__":
 mu = 0
 sigma = 1 
 num_samples = 1000000 #number of uniform random samples for inverse transform sampling
 M = 50 # number of terms in Hilbert Transform
 h = 0.1 # step size for Hilbert Transform
 x_min = -4 # minimum x-value for the inverse CDF
 x_max = 4 # maximum x-value for the inverse CDF
 K = 100 # number of intervals in the x range
 #test_inverse_transform_with_hilbert(mu, sigma, x_min, x_max, K, num_samples, h, M)

 # Tabulate the CDF over the specified interval
 x_values, cdf_values = tabulate_cdf_using_hilbert(mu, sigma, x_min, x_max, K, h, M)
    
# Generate uniform random samples and transform them
 uniform_samples = np.random.rand(num_samples)
 inverse_samples = [inverse_transform_sampling(cdf_values, x_values, U) for U in uniform_samples]
    
 # Plot the results
 plt.hist(inverse_samples, bins=30, density=True, alpha=0.6, color='skyblue', label="Simulated (Inverse Transform)")
 x = np.linspace(x_min, x_max, num_samples)
 plt.plot(x, (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2), 'r-', lw=2, label="Theoretical PDF")
 plt.xlabel("x")
 plt.ylabel("Density")
 plt.legend()
 plt.title(f"Inverse Transform Sampling from Normal Distribution (mu={mu}, sigma={sigma})")
 plt.grid(True)
 plt.show()

