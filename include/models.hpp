#pragma once

#include "sde.hpp"
#include <cmath>

namespace sde {

/**
 * @brief Vasicek (Ornstein-Uhlenbeck) model
 * 
 * dX_t = κ(θ - X_t)dt + σdW_t
 */
class VasicekModel : public SDE {
public:
    /**
     * @brief Construct a new Vasicek model
     * @param kappa Mean reversion speed
     * @param theta Long-term mean
     * @param sigma Volatility
     * @param x0 Initial value
     */
    VasicekModel(double kappa, double theta, double sigma, double x0 = 0.0)
        : kappa_(kappa), theta_(theta), sigma_(sigma), x0_(x0) {}
    
    void drift(const Vector& x, Vector& output, double t) const override {
        output[0] = kappa_ * (theta_ - x[0]);
    }
    
    void diffusion(const Vector& x, Matrix& output, double t) const override {
        output[0][0] = sigma_;
    }
    
    std::size_t dimension() const override { return 1; }
    std::size_t wiener_dimension() const override { return 1; }
    
    // Analytical solutions
    double expected_value(double t, double x0) const {
        return theta_ + (x0 - theta_) * std::exp(-kappa_ * t);
    }
    
    double variance(double t) const {
        return (sigma_ * sigma_) / (2.0 * kappa_) * (1.0 - std::exp(-2.0 * kappa_ * t));
    }
    
    // Getters
    double kappa() const { return kappa_; }
    double theta() const { return theta_; }
    double sigma() const { return sigma_; }
    double x0() const { return x0_; }
    
private:
    double kappa_;  // Mean reversion speed
    double theta_;  // Long-term mean
    double sigma_;  // Volatility
    double x0_;     // Initial value
};

/**
 * @brief Heston stochastic volatility model
 * 
 * dS_t = μS_t dt + √v_t S_t dW_t^1
 * dv_t = κ(θ - v_t)dt + ξ√v_t dW_t^2
 * 
 * with dW_t^1 dW_t^2 = ρ dt
 */
class HestonModel : public SDE {
public:
    /**
     * @brief Construct a new Heston model
     * @param mu Drift of the asset
     * @param kappa Mean reversion speed of volatility
     * @param theta Long-term variance
     * @param xi Volatility of volatility
     * @param rho Correlation between asset and volatility
     * @param s0 Initial asset price
     * @param v0 Initial variance
     */
    HestonModel(double mu, double kappa, double theta, double xi, double rho,
               double s0 = 100.0, double v0 = 0.04)
        : mu_(mu), kappa_(kappa), theta_(theta), xi_(xi), rho_(rho),
          s0_(s0), v0_(v0) {
        // Ensure correlation is in [-1, 1]
        rho_ = std::max(-1.0, std::min(1.0, rho));
    }
    
    void drift(const Vector& x, Vector& output, double t) const override {
        // x[0] = S (asset price), x[1] = v (variance)
        output[0] = mu_ * x[0];                    // μS
        output[1] = kappa_ * (theta_ - x[1]);      // κ(θ - v)
    }
    
    void diffusion(const Vector& x, Matrix& output, double t) const override {
        // Diffusion matrix:
        // [ σ₁S   0   ]
        // [ ρξ√v  ξ√v√(1-ρ²) ]
        
        double sqrt_v = std::sqrt(std::max(x[1], 0.0));  // Ensure non-negative variance
        double sqrt_1_rho2 = std::sqrt(1.0 - rho_ * rho_);
        
        // First row: dS_t
        output[0][0] = sqrt_v * x[0];  // σ₁S = √v * S
        output[0][1] = 0.0;            // No direct effect of second Wiener on S
        
        // Second row: dv_t
        output[1][0] = rho_ * xi_ * sqrt_v;          // ρξ√v
        output[1][1] = xi_ * sqrt_v * sqrt_1_rho2;   // ξ√v√(1-ρ²)
    }
    
    std::size_t dimension() const override { return 2; }  // S and v
    std::size_t wiener_dimension() const override { return 2; }  // Two correlated Wiener processes
    
    // Getters
    double mu() const { return mu_; }
    double kappa() const { return kappa_; }
    double theta() const { return theta_; }
    double xi() const { return xi_; }
    double rho() const { return rho_; }
    double s0() const { return s0_; }
    double v0() const { return v0_; }
    
private:
    double mu_;     // Drift of the asset
    double kappa_;  // Mean reversion speed of volatility
    double theta_;  // Long-term variance
    double xi_;     // Volatility of volatility
    double rho_;    // Correlation between asset and volatility
    double s0_;     // Initial asset price
    double v0_;     // Initial variance
};

} // namespace sde
