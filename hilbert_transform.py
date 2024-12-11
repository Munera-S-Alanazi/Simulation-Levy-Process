import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# the characteristic function of the normal distribution
def characteristic_function_normal(z, mu, sigma):
    return np.exp(1j * mu * z - 0.5 * (sigma ** 2) * (z ** 2))

# Step 1: Implement the Hilbert Transform to approximate the CDF
def hilbert_transform_cdf(x, mu, sigma, h, M):
    # Discretized Hilbert Transform summation
    summation = 0
    for m in range(-M, M + 1):
        z = (m - 0.5) * h
        term = np.exp(-1j * z * x) * characteristic_function_normal(z, mu, sigma) / (((m - 0.5) * np.pi))
        summation += term

    # Final calculation for the CDF approximation with corrected sign
    F_approx = 0.5 + summation * (1j/2.0)
    return F_approx.real


# the theoretical CDF of the normal distribution for comparison
def theoretical_cdf_normal(x, mu, sigma):
    return 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))



if __name__ == "__main__":
  # a list of different mu and sigma values to test
  mu_values = [0] #Mean
  sigma_values = [1] #Standard deviation
  M = 37 # terms in Hilbart transorm
  h = 0.17 #Discretization step size
  x_values = np.linspace(-4, 4, 100)  #  Value at which to evaluate the CDF

  for mu in mu_values:
    for sigma in sigma_values:
        # Calculate CDF values using the Hilbert Transform approximation and theoretical CDF
        approx_cdf_values = [hilbert_transform_cdf(x, mu, sigma, h, M) for x in x_values]
        theoretical_cdf_values = [theoretical_cdf_normal(x, mu, sigma) for x in x_values]

        # Calculate the error between simulated and theoretical values
        errors = [abs(approx - theoretical) for approx, theoretical in zip(approx_cdf_values, theoretical_cdf_values)]

        # Print simulated, theoretical values, and errors
        print(f"\nResults for mu={mu}, sigma={sigma}")
        print(f"Optimal parameters used: M = {M}, h = {h}, x_min = {-6.37}, x_max = {6.37}")
        print("x\tSimulated CDF (HT)\tTheoretical CDF (Normal)\tError")
        for x, approx, theoretical, error in zip(x_values, approx_cdf_values, theoretical_cdf_values, errors):
            print(f"{x:.2f}\t{approx:.6f}\t\t{theoretical:.6f}\t\t{error:.4e}")

        # Plot the results for comparison
        plt.plot(x_values, approx_cdf_values, label=f"Approximate CDF (HT), mu={mu}, sigma={sigma}", linestyle="--")
        plt.plot(x_values, theoretical_cdf_values, label=f"Theoretical CDF (Normal), mu={mu}, sigma={sigma}", linestyle="-")

  # Plot settings
  plt.xlabel("x")
  plt.ylabel("CDF")
  plt.title("Comparison of Approximate and Theoretical CDFs for Different mu and sigma")
  plt.legend(loc="best")
  plt.grid(True)
  plt.show()