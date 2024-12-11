import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Characteristic function for normal distribution
def characteristic_function_normal(z, mu, sigma):
    return np.exp(1j * mu * z - 0.5 * (sigma * z)**2)

def hilbert_transform_cdf_derivative(x, mu, sigma, h, M):
    summation = 0
    for m in range(1, M + 1):
        z = (m - 0.5) * h
        term = z * np.exp(-1j * z * x) * characteristic_function_normal(z, mu, sigma) / (((m - 0.5) * np.pi))
        summation += term

    # Final calculation for the PDF approximation
    f_approx = summation.real
    return f_approx

def normal_pdf(x, mu, sigma):
    coeff = 1 / (np.sqrt(2 * np.pi) * sigma)
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    return coeff * np.exp(exponent)

if __name__ == "__main__":

 # Parameters for the Hilbert Transform
 mu, sigma = 0, 1  # Standard normal distribution
 h, M = 0.1, 50    # Step size and number of summation terms

 # Generate x values for comparison
 x_values = np.linspace(-4, 4, 100)

 # Compute derivatives using the Hilbert Transform
 hilbert_derivatives = [hilbert_transform_cdf_derivative(x, mu, sigma, h, M) for x in x_values]
 print(hilbert_derivatives)

 # Compute exact PDF using scipy
 exact_pdf = normal_pdf(x_values, mu, sigma)

 # Plot the comparison
 plt.figure(figsize=(10, 6))
 plt.plot(x_values, exact_pdf, label="Exact PDF (scipy)", color="blue", linewidth=2)
 plt.plot(x_values, hilbert_derivatives, label="Hilbert Approximation", color="red", linestyle="--")
 plt.title("Comparison of Exact PDF and Hilbert Transform Approximation")
 plt.xlabel("x")
 plt.ylabel("PDF / CDF Derivative")
 plt.legend()
 plt.grid()
 plt.show()
