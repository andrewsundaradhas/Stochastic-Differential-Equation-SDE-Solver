#pragma once

#include "sde.hpp"
#include <cmath>

namespace sde {

/**
 * @brief Euler-Maruyama method for SDEs
 */
class EulerMaruyamaSolver : public SDESolver {
public:
    void step(const SDE& sde, Vector& x, double t, double dt, const Vector& dW) const override {
        const std::size_t dim = sde.dimension();
        const std::size_t w_dim = sde.wiener_dimension();
        
        // Compute drift term
        Vector a(dim);
        sde.drift(x, a, t);
        
        // Compute diffusion term
        Matrix b(dim, Vector(w_dim));
        sde.diffusion(x, b, t);
        
        // Update state
        for (std::size_t i = 0; i < dim; ++i) {
            x[i] += a[i] * dt;
            for (std::size_t j = 0; j < w_dim; ++j) {
                x[i] += b[i][j] * dW[j];
            }
        }
    }
    
    const char* name() const override { return "Euler-Maruyama"; }
};

/**
 * @brief Milstein method for SDEs (for 1D SDEs or diagonal noise)
 */
class MilsteinSolver : public SDESolver {
public:
    void step(const SDE& sde, Vector& x, double t, double dt, const Vector& dW) const override {
        // For simplicity, this implementation assumes 1D SDEs or diagonal noise
        // A full implementation would need to handle the full derivative of the diffusion term
        
        const std::size_t dim = sde.dimension();
        const std::size_t w_dim = sde.wiener_dimension();
        
        // Compute drift term
        Vector a(dim);
        sde.drift(x, a, t);
        
        // Compute diffusion term
        Matrix b(dim, Vector(w_dim));
        sde.diffusion(x, b, t);
        
        // For 1D case or diagonal noise
        for (std::size_t i = 0; i < dim; ++i) {
            // Euler-Maruyama part
            x[i] += a[i] * dt;
            
            // Milstein correction (simplified for diagonal noise)
            for (std::size_t j = 0; j < w_dim; ++j) {
                x[i] += b[i][j] * dW[j];
                
                // Add Milstein correction term if i == j (diagonal noise)
                if (i == j) {
                    // Approximate derivative of b[i][j] with respect to x[i]
                    const double h = 1e-6;
                    Vector x_plus_h = x;
                    x_plus_h[i] += h;
                    
                    Matrix b_plus_h(dim, Vector(w_dim));
                    sde.diffusion(x_plus_h, b_plus_h, t);
                    
                    double db_dx = (b_plus_h[i][j] - b[i][j]) / h;
                    x[i] += 0.5 * b[i][j] * db_dx * (dW[j] * dW[j] - dt);
                }
            }
        }
    }
    
    const char* name() const override { return "Milstein"; }
};

} // namespace sde
