#include "../include/models.hpp"
#include "../include/solvers.hpp"
#include <gtest/gtest.h>
#include <cmath>

class HestonTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common parameters for tests
        mu = 0.05;
        kappa = 2.0;
        theta = 0.04;
        xi = 0.3;
        rho = -0.7;
        s0 = 100.0;
        v0 = 0.04;
        
        // Create model and solvers
        model = std::make_unique<sde::HestonModel>(mu, kappa, theta, xi, rho, s0, v0);
        euler_solver = std::make_unique<sde::EulerMaruyamaSolver>();
        milstein_solver = std::make_unique<sde::MilsteinSolver>();
    }
    
    // Test parameters
    double mu, kappa, theta, xi, rho, s0, v0;
    
    // Test objects
    std::unique_ptr<sde::HestonModel> model;
    std::unique_ptr<sde::EulerMaruyamaSolver> euler_solver;
    std::unique_ptr<sde::MilsteinSolver> milstein_solver;
    
    // Test tolerance
    const double tolerance = 1e-2;
};

// Test drift and diffusion calculations
TEST_F(HestonTest, DriftAndDiffusion) {
    sde::Vector x = {s0, v0};  // [S, v]
    sde::Vector drift(2);
    sde::Matrix diffusion(2, sde::Vector(2));
    
    // Test drift
    model->drift(x, drift, 0.0);
    EXPECT_NEAR(drift[0], mu * x[0], 1e-10);  // μS
    EXPECT_NEAR(drift[1], kappa * (theta - x[1]), 1e-10);  // κ(θ - v)
    
    // Test diffusion
    model->diffusion(x, diffusion, 0.0);
    double sqrt_v = std::sqrt(x[1]);
    EXPECT_NEAR(diffusion[0][0], sqrt_v * x[0], 1e-10);  // √v*S
    EXPECT_NEAR(diffusion[1][0], rho * xi * sqrt_v, 1e-10);  // ρξ√v
    EXPECT_NEAR(diffusion[1][1], xi * sqrt_v * std::sqrt(1 - rho*rho), 1e-10);  // ξ√v√(1-ρ²)
}

// Test single step of Euler-Maruyama
TEST_F(HestonTest, EulerMaruyamaSingleStep) {
    sde::Vector x = {s0, v0};
    double dt = 0.01;
    double dW1 = 0.1, dW2 = -0.1;  // Fixed increments for testing
    
    // Take a single step
    euler_solver->step(*model, x, 0.0, dt, {dW1, dW2});
    
    // Expected values after one step
    double expected_S = s0 + mu * s0 * dt + std::sqrt(v0) * s0 * dW1;
    double expected_v = v0 + kappa * (theta - v0) * dt + xi * std::sqrt(v0) * dW2;
    
    EXPECT_NEAR(x[0], expected_S, 1e-10);
    EXPECT_GE(x[1], 0.0);  // Variance should stay non-negative
}

// Test dimensions
TEST_F(HestonTest, Dimensions) {
    EXPECT_EQ(model->dimension(), 2);
    EXPECT_EQ(model->wiener_dimension(), 2);
}

// Test parameter getters
TEST_F(HestonTest, ParameterGetters) {
    EXPECT_DOUBLE_EQ(model->mu(), mu);
    EXPECT_DOUBLE_EQ(model->kappa(), kappa);
    EXPECT_DOUBLE_EQ(model->theta(), theta);
    EXPECT_DOUBLE_EQ(model->xi(), xi);
    EXPECT_DOUBLE_EQ(model->rho(), rho);
    EXPECT_DOUBLE_EQ(model->s0(), s0);
    EXPECT_DOUBLE_EQ(model->v0(), v0);
}

// Test non-negative variance
TEST_F(HestonTest, NonNegativeVariance) {
    sde::Vector x = {s0, 0.0};  // v = 0
    double dt = 0.01;
    
    // Take a step that would make variance negative
    euler_solver->step(*model, x, 0.0, dt, {-0.2, -0.2});
    
    // Variance should be non-negative
    EXPECT_GE(x[1], 0.0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
