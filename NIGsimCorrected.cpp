#include <iostream>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>
#include <fstream>
#include <complex>

// the CF of NIG process as in Cont & Tankov book 
std::complex<double> nigCF(double t, double z, double theta, double sigma, double kappa) {
    std::complex<double> i(0.0, 1.0);
    std::complex<double> term1 = 1.0/ kappa;
    std::complex<double> term2 = 1.0 - 2.0*i*z*theta*kappa + kappa*sigma*sigma*z*z;
    return std::exp(t*(term1 - term1*std::sqrt(term2)));
} 

// Function to generate Inverse Gaussian distribution sample using Tankov and Cont
double inverseGaussianSample(double lambda, double mu) {
    //random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> normal_dist(0.0, 1.0); 
    // They said in the book " normal random variable N" not standard Normal varible!!
    double N = normal_dist(gen);
    double y = N * N;
    double x = mu + ((mu * mu * y) / (2 * lambda)) - mu / (2 * lambda) * sqrt(4 * mu * lambda * y + mu * mu * y * y);
    //std::cout << "(2 * lambda) = " << 2 * lambda << std::endl;
    //std::cout << "4 * mu * lambda * y + mu * mu * y * y = " << 4 * mu * lambda * y + mu * mu * y * y << std::endl;
    double u = std::uniform_real_distribution<>(0.0, 1.0)(gen); 
    if (u <= mu / (mu + x)) {
        //std::cout << "x= " << x << std::endl;
        return x;
    } else {
        //std::cout << "mu^2/x= " << (mu * mu) / x << std::endl;
        //std::cout << " x= " << x << std::endl;

        return (mu * mu) / x;
    }
}

// Now this function to simulate the variance gamma process as in Algorithem in Cont & Tankov book 
std::vector<double> simulateNIG(double t, int n, double theta, double sigma, double kappa) {
    // here I want to generate the time grid (n evenly spaced points between 0 and t)
    //std::vector<double> times(n);
    double delta_t;  // Step size
    if (n == 1) {
            delta_t = t; // First time step is just 0.0 according to the book
        } else {
            delta_t = t / (n - 1); // subsequent time steps are the differences
        }

    // I want now to initialize random number generators for normal distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> normal_dis(0.0, 1.0); // N(0, 1) distribution

    // This is a vector to store the discretized trajectory X(t_i)
    std::vector<double> X;
    double X_t = 0.0; // to store the sum incrementally starting from 0

    // Step 1: Simulate n independent inverse Gaussian variables
    std::vector<double> deltaS(n, 0.0);

    for (int i = 0; i < n; ++i) {
        deltaS[i] = inverseGaussianSample((delta_t*delta_t)/kappa, delta_t); // Now simulate IG with parameter delta_t^2 / kappa and mu = delta_t
        //std::cout << "deltaS[i] = " << deltaS[i] << std::endl;
    }

    // Step 2: Simulate n i.i.d N(0,1) random variables and compute Delta X_i
    for (int i = 0; i < n; ++i) {
        double N_i = normal_dis(gen); // simulate N(0,1)
        double deltaX_i = sigma * N_i * std::sqrt(deltaS[i]) + theta * deltaS[i]; // Compute Delta X_i
        X_t += deltaX_i; // incrementally sum Delta X_i to get X(t_i)
        X.push_back(X_t); // store the current X(t_i)
    }

    return X; // Return the entire discretized trajectory
}

// in MC do not forget you have theta sigma kappa!!
// MC estimation of E[exp(i * z * X_t)] with MC error
void monteCarloEstimate(double t, const std::vector<double>& zValues, double theta, double sigma, double kappa, int M, int n, 
                        std::vector<double>& real_CFsim, std::vector<double>& std_dev_real) {
    for (size_t i = 0; i < zValues.size(); ++i) {
        double z = zValues[i];
        std::vector<double> estimates(M, 0.0);  // initialize 'M' elements, all set to 0.0

        // run the MC simulation
        for (int j = 0; j < M; ++j) {
            // simulate the variance gamma process
            std::vector<double> X = simulateNIG(t, n, theta, sigma, kappa);

            // Now I want to get the final value X_t
            double X_t = X.back();

            // compute Re(exp(i * z * X_t)) and store it
            double exp_izXt_real = std::real(std::exp(std::complex<double>(0.0, z * X_t)));  // only real part
            estimates.push_back(exp_izXt_real);
        }

        // Now compute the MC estimate of E[exp(i * z * X_t)]
        double sum = 0.0;
        for (double est : estimates) {
            sum += est;
        }
        double mcEstimate = sum / static_cast<double>(M);

        // Now I want to ompute the standard deviation of the estimates
        double sum_sq = 0.0;
        for (double est : estimates) {
            sum_sq += std::pow(est - mcEstimate, 2.0);
        }
        double variance = sum_sq / static_cast<double>(M - 1);
        double stdDev = std::sqrt(variance);

        // Standard error of the Monte Carlo estimate
        double standardError = stdDev / std::sqrt(static_cast<double>(M));

        // Store results
        real_CFsim[i] = mcEstimate;
        std_dev_real[i] = 3.0 * standardError;  // 3 standard deviations
    }
}
void saveResultsToFile(const std::vector<double>& zValues, const std::vector<double>& real_cf_sim, 
                       const std::vector<double>& real_cf_theo, const std::vector<double>& abs_error, 
                       const std::vector<double>& std_dev, const std::string& filename) {
    std::ofstream file(filename);
    file << "zValues,Simulated Real,Theoretical Real,Absolute Error,3 Std Dev\n";
    for (size_t i = 0; i < zValues.size(); ++i) {
        file << zValues[i] << "," << real_cf_sim[i] << "," << real_cf_theo[i] << "," << abs_error[i] << "," << 3 * std_dev[i] << "\n";
    }
    file.close();
}

int main() {
    double theta = 0.2;  // Drift
    double sigma = 0.2;  // Volatility
    double kappa = 0.1;  // Variance rate 
    double t = 4.0;
    std::vector<double> zValues = {0, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0}; // values of z for characteristic function
    int n = 1;     // number of time steps 
    //std::vector<double> X_t = simulateNIG( t,  n,  sigma, theta,  kappa);

    int M = 1000000; // number of simulation
    std::vector<double> real_cf_sim(zValues.size());
    std::vector<double> std_dev_real(zValues.size());
    std::vector<double> real_cf_theo(zValues.size());
    monteCarloEstimate(t, zValues, theta, sigma, kappa, M, n, real_cf_sim, std_dev_real);

    for (size_t i = 0; i < zValues.size(); ++i) {
        std::complex<double> CFs = nigCF(t, zValues[i], theta, sigma, kappa);
        //std::cout << "Theoretical CF(" << zValues[i] << "): " << CFs << std::endl;
        real_cf_theo[i] = CFs.real();
        //monteCarloEstimate( t,  z,  theta,  sigma,  kappa,  M,  n);

     }
     // Now Calculate absolute error where the 3 std dev (already done in monteCarloEstimation)
    std::vector<double> abs_error(zValues.size());
    
    for (size_t i = 0; i < zValues.size(); ++i) {
        abs_error[i] = std::abs(real_cf_sim[i] - real_cf_theo[i]);
        std::cout << "Estimated E[exp(izX_t)]: " << real_cf_sim[i] << std::endl;
        std::cout << "Theoretical CF(" << zValues[i] << "): " << real_cf_theo[i] << std::endl;
        std::cout << "The abs error: " << abs_error[i] << std::endl;
        std::cout << "The MC error (3 std devs): " << std_dev_real[i] << std::endl;
    }
    


    // Save results to a csv file
    saveResultsToFile(zValues, real_cf_sim, real_cf_theo, abs_error, std_dev_real, "NIG_results.csv");

  return 0;
}
