#pragma once

#include <vector>
#include <memory>
#include <random>

namespace sde {

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

/**
 * @brief Abstract base class for Stochastic Differential Equations (SDEs)
 */
class SDE {
public:
    virtual ~SDE() = default;
    
    /**
     * @brief Compute the drift term a(X_t, t)
     * @param x Current state vector
     * @param output Output vector for drift terms
     * @param t Current time
     */
    virtual void drift(const Vector& x, Vector& output, double t) const = 0;
    
    /**
     * @brief Compute the diffusion term b(X_t, t)
     * @param x Current state vector
     * @param output Output matrix for diffusion terms
     * @param t Current time
     */
    virtual void diffusion(const Vector& x, Matrix& output, double t) const = 0;
    
    /**
     * @brief Get the dimension of the state vector
     */
    virtual std::size_t dimension() const = 0;
    
    /**
     * @brief Get the number of Wiener processes
     */
    virtual std::size_t wiener_dimension() const = 0;
};

/**
 * @brief Base class for SDE solvers
 */
class SDESolver {
public:
    virtual ~SDESolver() = default;
    
    /**
     * @brief Perform one step of the SDE solver
     * @param sde The SDE to solve
     * @param x Current state vector (input/output)
     * @param t Current time
     * @param dt Time step
     * @param dW Vector of Wiener increments
     */
    virtual void step(const SDE& sde, Vector& x, double t, double dt, const Vector& dW) const = 0;
    
    /**
     * @brief Get the name of the solver
     */
    virtual const char* name() const = 0;
};

} // namespace sde
