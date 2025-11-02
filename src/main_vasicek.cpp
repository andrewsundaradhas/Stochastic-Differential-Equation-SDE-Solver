#include "models.hpp"
#include "solvers.hpp"
#include "rng_pool.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>

// Function to run Monte Carlo simulation for the Vasicek model
double run_vasicek_mc(
    const sde::VasicekModel& model,
    const sde::SDESolver& solver,
    double T,          // Terminal time
    std::size_t steps, // Number of time steps
    std::size_t paths, // Number of paths
    sde::RNGPool& rng_pool) {
    
    const double dt = T / steps;
    const double sqrt_dt = std::sqrt(dt);
    
    // Storage for terminal values
    std::vector<double> terminal_values(paths);
    
    // Run simulation
    #pragma omp parallel for
    for (std::size_t p = 0; p < paths; ++p) {
        // Initialize state
        sde::Vector x = {model.x0()};
        
        // Time stepping
        for (std::size_t i = 0; i < steps; ++i) {
            double t = i * dt;
            
            // Generate Wiener increment
            double dW = rng_pool.normal() * sqrt_dt;
            
            // Take a step
            solver.step(model, x, t, dt, {dW});
        }
        
        // Store terminal value
        terminal_values[p] = x[0];
    }
    
    // Compute statistics
    double sum = 0.0;
    double sum_sq = 0.0;
    
    for (auto x : terminal_values) {
        sum += x;
        sum_sq += x * x;
    }
    
    double mean = sum / paths;
    double variance = (sum_sq / paths) - (mean * mean);
    
    // Compare with analytical solution
    double expected_mean = model.expected_value(T, model.x0());
    double expected_var = model.variance(T);
    
    // Print results
    std::cout << "\n=== Vasicek Model (" << solver.name() << ") ===\n"
              << "Parameters: kappa=" << model.kappa() 
              << ", theta=" << model.theta() 
              << ", sigma=" << model.sigma() 
              << ", x0=" << model.x0() << "\n"
              << "T=" << T << ", steps=" << steps << ", paths=" << paths << "\n"
              << "\nMonte Carlo Results:\n"
              << "  E[X_T] = " << mean << " (Analytic: " << expected_mean << ")"
              << "  Error: " << std::abs(mean - expected_mean) / expected_mean * 100.0 << "%\n"
              << "  Var[X_T] = " << variance << " (Analytic: " << expected_var << ")"
              << "  Error: " << std::abs(variance - expected_var) / expected_var * 100.0 << "%\n";
    
    return mean;
}

int main() {
    // Set up model parameters
    double kappa = 1.0;    // Mean reversion speed
    double theta = 0.1;    // Long-term mean
    double sigma = 0.2;    // Volatility
    double x0 = 0.05;      // Initial value
    
    double T = 1.0;        // Terminal time
    std::size_t steps = 252;  // Daily steps for one year
    std::size_t paths = 10000; // Number of paths
    
    // Create model and solvers
    sde::VasicekModel model(kappa, theta, sigma, x0);
    sde::EulerMaruyamaSolver euler_solver;
    sde::MilsteinSolver milstein_solver;
    
    // Initialize RNG pool
    sde::RNGPool rng_pool(std::thread::hardware_concurrency(), 42);
    
    std::cout << "Vasicek Model Simulation\n"
              << "========================\n";
    
    // Run with Euler-Maruyama
    auto start = std::chrono::high_resolution_clock::now();
    run_vasicek_mc(model, euler_solver, T, steps, paths, rng_pool);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time: " << elapsed.count() << " seconds\n\n";
    
    // Run with Milstein
    start = std::chrono::high_resolution_clock::now();
    run_vasicek_mc(model, milstein_solver, T, steps, paths, rng_pool);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Time: " << elapsed.count() << " seconds\n";
    
    return 0;
}
