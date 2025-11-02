#include "../include/models.hpp"
#include "../include/solvers.hpp"
#include <gtest/gtest.h>
#include <cmath>

class VasicekTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common parameters for tests
        kappa = 1.0;
        theta = 0.1;
        sigma = 0.2;
        x0 = 0.05;
        T = 1.0;
        
        // Create model and solvers
        model = std::make_unique<sde::VasicekModel>(kappa, theta, sigma, x0);
        euler_solver = std::make_unique<sde::EulerMaruyamaSolver>();
        milstein_solver = std::make_unique<sde::MilsteinSolver>();
    }
    
    // Test parameters
    double kappa, theta, sigma, x0, T;
    
    // Test objects
    std::unique_ptr<sde::VasicekModel> model;
    std::unique_ptr<sde::EulerMaruyamaSolver> euler_solver;
    std::unique_ptr<sde::MilsteinSolver> milstein_solver;
    
    // Test tolerance
    const double tolerance = 1e-2;  // 1% tolerance for Monte Carlo tests
};

// Test analytical solutions
TEST_F(VasicekTest, AnalyticalSolutions) {
    // Test expected value
    double expected_mean = model->expected_value(T, x0);
    double expected_var = model->variance(T);
    
    // Theoretical values
    double theoretical_mean = theta + (x0 - theta) * std::exp(-kappa * T);
    double theoretical_var = (sigma * sigma) / (2.0 * kappa) * (1.0 - std::exp(-2.0 * kappa * T));
    
    EXPECT_NEAR(expected_mean, theoretical_mean, 1e-10);
    EXPECT_NEAR(expected_var, theoretical_var, 1e-10);
}

// Test drift and diffusion calculations
TEST_F(VasicekTest, DriftAndDiffusion) {
    sde::Vector x = {0.05};
    sde::Vector drift(1);
    sde::Matrix diffusion(1, sde::Vector(1));
    
    // Test drift
    model->drift(x, drift, 0.0);
    double expected_drift = kappa * (theta - x[0]);
    EXPECT_NEAR(drift[0], expected_drift, 1e-10);
    
    // Test diffusion
    model->diffusion(x, diffusion, 0.0);
    EXPECT_NEAR(diffusion[0][0], sigma, 1e-10);
}

// Test Euler-Maruyama solver with a single step
TEST_F(VasicekTest, EulerMaruyamaSingleStep) {
    sde::Vector x = {x0};
    double dt = 0.01;
    double dW = 0.1;  // Fixed increment for testing
    
    // Take a single step
    euler_solver->step(*model, x, 0.0, dt, {dW});
    
    // Expected value after one step
    double expected = x0 + kappa * (theta - x0) * dt + sigma * dW;
    
    EXPECT_NEAR(x[0], expected, 1e-10);
}

// Test Milstein solver with a single step
TEST_F(VasicekTest, MilsteinSingleStep) {
    sde::Vector x = {x0};
    double dt = 0.01;
    double dW = 0.1;  // Fixed increment for testing
    
    // Take a single step
    milstein_solver->step(*model, x, 0.0, dt, {dW});
    
    // For Vasicek model, Milstein should be same as Euler since b' = 0
    double expected = x0 + kappa * (theta - x0) * dt + sigma * dW;
    
    EXPECT_NEAR(x[0], expected, 1e-10);
}

// Test dimension methods
TEST_F(VasicekTest, Dimensions) {
    EXPECT_EQ(model->dimension(), 1);
    EXPECT_EQ(model->wiener_dimension(), 1);
}

// Main function for Google Test
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
