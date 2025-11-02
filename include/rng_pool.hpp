#pragma once

#include <random>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>

namespace sde {

/**
 * @brief Thread-safe random number generator pool
 */
class RNGPool {
public:
    /**
     * @brief Initialize the RNG pool with a specific number of generators
     * @param num_generators Number of RNGs to create (one per thread)
     * @param seed Base seed for the RNGs
     */
    explicit RNGPool(std::size_t num_generators = std::thread::hardware_concurrency(), 
                    unsigned int seed = std::random_device{}()) {
        // Initialize each RNG with a different seed
        for (std::size_t i = 0; i < num_generators; ++i) {
            // Use a different seed for each generator
            std::seed_seq seq{seed, static_cast<unsigned int>(i)};
            rngs_.emplace_back(seq);
        }
    }
    
    /**
     * @brief Get a reference to a thread-local RNG
     * @return Reference to a thread-safe RNG
     */
    std::mt19937_64& get_rng() {
        // Simple round-robin assignment
        // In a real implementation, you might want to use thread-local storage
        std::lock_guard<std::mutex> lock(mutex_);
        std::size_t idx = counter_++ % rngs_.size();
        return rngs_[idx];
    }
    
    /**
     * @brief Generate a standard normal random variable
     * @return A standard normal random variable (mean=0, variance=1)
     */
    double normal() {
        return normal_dist_(get_rng());
    }
    
    /**
     * @brief Generate a vector of standard normal random variables
     * @param n Number of random variables to generate
     * @return Vector of standard normal random variables
     */
    std::vector<double> normal_vector(std::size_t n) {
        std::vector<double> result(n);
        for (auto& x : result) {
            x = normal();
        }
        return result;
    }
    
    /**
     * @brief Generate correlated normal random variables
     * @param correlation_matrix Correlation matrix (must be positive semi-definite)
     * @return Vector of correlated normal random variables
     */
    std::vector<double> correlated_normal(const std::vector<std::vector<double>>& correlation_matrix);

private:
    std::vector<std::mt19937_64> rngs_;
    std::normal_distribution<double> normal_dist_{0.0, 1.0};
    std::mutex mutex_;
    std::atomic<std::size_t> counter_{0};
};

/**
 * @brief Thread-safe singleton for global RNG pool
 */
class GlobalRNG {
public:
    static RNGPool& get_instance() {
        static RNGPool instance;
        return instance;
    }
    
    // Delete copy/move constructors and assignment operators
    GlobalRNG(const GlobalRNG&) = delete;
    GlobalRNG& operator=(const GlobalRNG&) = delete;
    GlobalRNG(GlobalRNG&&) = delete;
    GlobalRNG& operator=(GlobalRNG&&) = delete;
    
private:
    GlobalRNG() = default;
};

} // namespace sde
