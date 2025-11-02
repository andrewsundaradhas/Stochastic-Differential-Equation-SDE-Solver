#include "models.hpp"
#include "solvers.hpp"
#include "rng_pool.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>

// Function to run Monte Carlo simulation for the Heston model
void run_heston_mc(
    const sde::HestonModel& model,
    const sde::SDESolver& solver,
    double T,          // Terminal time
    std::size_t steps, // Number of time steps
    std::size_t paths, // Number of paths
    sde::RNGPool& rng_pool) {
    
    const double dt = T / steps;
    const double sqrt_dt = std::sqrt(dt);
    
    // Storage for terminal values
    std::vector<double> terminal_prices(paths);
    std::vector<double> terminal_variances(paths);
    
    // Correlation parameters
    double rho = model.rho();
    double sqrt_1_rho2 = std::sqrt(1.0 - rho * rho);
    
    // Run simulation
    #pragma omp parallel for
    for (std::size_t p = 0; p < paths; ++p) {
        // Initialize state: [S, v]
        sde::Vector x = {model.s0(), model.v0()};
        
        // Time stepping
        for (std::size_t i = 0; i < steps; ++i) {
            double t = i * dt;
            
            // Generate correlated Wiener increments
            double Z1 = rng_pool.normal();
            double Z2 = rng_pool.normal();
            
            // Correlated increments
            double dW1 = Z1 * sqrt_dt;
            double dW2 = (rho * Z1 + sqrt_1_rho2 * Z2) * sqrt_dt;
            
            // Take a step
            solver.step(model, x, t, dt, {dW1, dW2});
            
            // Ensure variance stays non-negative (full truncation scheme)
            x[1] = std::max(0.0, x[1]);
        }
        
        // Store terminal values
        terminal_prices[p] = x[0];
        terminal_variances[p] = x[1];
    }
    
    // Compute statistics for prices
    double sum_S = 0.0;
    double sum_S2 = 0.0;
    
    for (auto S : terminal_prices) {
        sum_S += S;
        sum_S2 += S * S;
    }
    
    double mean_S = sum_S / paths;
    double var_S = (sum_S2 / paths) - (mean_S * mean_S);
    
    // Compute statistics for variance
    double sum_v = 0.0;
    double sum_v2 = 0.0;
    
    for (auto v : terminal_variances) {
        sum_v += v;
        sum_v2 += v * v;
    }
    
    double mean_v = sum_v / paths;
    
    // Print results
    std::cout << "\n=== Heston Model (" << solver.name() << ") ===\n"
              << "Parameters: mu=" << model.mu() 
              << ", kappa=" << model.kappa() 
              << ", theta=" << model.theta() 
              << ", xi=" << model.xi() 
              << ", rho=" << model.rho() << "\n"
              << "Initial: S0=" << model.s0() 
              << ", v0=" << model.v0() << "\n"
              << "T=" << T << ", steps=" << steps << ", paths=" << paths << "\n"
              << "\nMonte Carlo Results:\n"
              << "  E[S_T] = " << mean_S << "\n"
              << "  Std[S_T] = " << std::sqrt(var_S) << "\n"
              << "  E[v_T] = " << mean_v << "\n";
}

int main() {
    // Set up model parameters (typical Heston parameters)
    double mu = 0.05;      // Drift
    double kappa = 2.0;    // Mean reversion speed of variance
    double theta = 0.04;   // Long-term variance (20% vol)
    double xi = 0.3;       // Vol of vol
    double rho = -0.7;     // Correlation between asset and vol
    double s0 = 100.0;     // Initial asset price
    double v0 = 0.04;      // Initial variance (20% vol)
    
    double T = 1.0;        // Terminal time (1 year)
    std::size_t steps = 252;  // Daily steps
    std::size_t paths = 10000; // Number of paths
    
    // Create model and solvers
    sde::HestonModel model(mu, kappa, theta, xi, rho, s0, v0);
    sde::EulerMaruyamaSolver euler_solver;
    sde::MilsteinSolver milstein_solver;
    
    // Initialize RNG pool
    sde::RNGPool rng_pool(std::thread::hardware_concurrency(), 42);
    
    std::cout << "Heston Model Simulation\n"
              << "======================\n";
    
    // Run with Euler-Maruyama
    auto start = std::chrono::high_resolution_clock::now();
    run_heston_mc(model, euler_solver, T, steps, paths, rng_pool);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time: " << elapsed.count() << " seconds\n\n";
    
    // Run with Milstein
    start = std::chrono::high_resolution_clock::now();
    run_heston_mc(model, milstein_solver, T, steps, paths, rng_pool);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Time: " << elapsed.count() << " seconds\n";
    
    return 0;
}
