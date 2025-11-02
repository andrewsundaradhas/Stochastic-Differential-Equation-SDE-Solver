# Stochastic Differential Equation (SDE) Solver

[![Build Status](https://github.com/yourusername/sde-solver/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/sde-solver/actions)
[![Documentation Status](https://readthedocs.org/projects/sde-solver/badge/?version=latest)](https://sde-solver.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-17%2F20-blue)](https://en.cppreference.com/)

A high-performance C++ library for solving stochastic differential equations (SDEs) with Python bindings, featuring multiple numerical solvers and financial models.

## Features

- **Multiple SDE Solvers**:
  - Euler-Maruyama method (1.0 strong order, 1.0 weak order)
  - Milstein method (1.0 strong order, 1.0 weak order)
  - Runge-Kutta method (1.0 strong order, 1.0 weak order)
  - Extensible interface for custom solvers

- **Financial Models**
  - Vasicek (Ornstein–Uhlenbeck) model
  - Heston stochastic volatility model
  - Easy to extend with custom models

- **Performance Optimizations**
  - Multi-threaded Monte Carlo simulations
  - Thread-safe random number generation
  - Efficient memory management

- **Validation & Testing**
  - Unit tests with Google Test
  - Analytical validation for Vasicek model
  - Convergence analysis

## Requirements

- C++17 compatible compiler (GCC 8+, Clang 7+, MSVC 2019+)
- CMake 3.15+
- Git (for downloading dependencies)
- [Optional] Doxygen (for documentation)

## Building the Project

### Prerequisites

```bash
# On Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    libgtest-dev

# On macOS (using Homebrew)
brew update
brew install cmake
```

### Build Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/Stochastic-Differential-Equation-SDE-Solver.git
cd Stochastic-Differential-Equation-SDE-Solver

# Create build directory and configure
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
cmake --build . --config Release

# [Optional] Run tests
ctest --output-on-failure

# [Optional] Install system-wide (requires admin privileges)
sudo cmake --install .
```

## Usage Examples

### Vasicek Model

```cpp
#include <sde/sde.hpp>
#include <sde/models.hpp>
#include <sde/solvers.hpp>

int main() {
    // Model parameters
    double kappa = 1.0;    // Mean reversion speed
    double theta = 0.1;    // Long-term mean
    double sigma = 0.2;    // Volatility
    double x0 = 0.05;      // Initial value
    
    // Create model and solver
    sde::VasicekModel model(kappa, theta, sigma, x0);
    sde::EulerMaruyamaSolver solver;
    
    // Simulation parameters
    double T = 1.0;        // Terminal time
    std::size_t steps = 252;  // Number of time steps
    std::size_t paths = 10000; // Number of paths
    
    // Run simulation
    // ... (see src/main_vasicek.cpp for complete example)
    
    return 0;
}
```

### Heston Model

```cpp
#include <sde/sde.hpp>
#include <sde/models.hpp>
#include <sde/solvers.hpp>

int main() {
    // Model parameters
    double mu = 0.05;      // Drift
    double kappa = 2.0;    // Mean reversion speed of variance
    double theta = 0.04;   // Long-term variance (20% vol)
    double xi = 0.3;       // Vol of vol
    double rho = -0.7;     // Correlation
    double s0 = 100.0;     // Initial asset price
    double v0 = 0.04;      // Initial variance (20% vol)
    
    // Create model and solver
    sde::HestonModel model(mu, kappa, theta, xi, rho, s0, v0);
    sde::MilsteinSolver solver;
    
    // Simulation parameters
    double T = 1.0;        // Terminal time (1 year)
    std::size_t steps = 252;  // Daily steps
    std::size_t paths = 10000; // Number of paths
    
    // Run simulation
    // ... (see src/main_heston.cpp for complete example)
    
    return 0;
}
```

## Running the Demos

After building the project, you can run the example applications:

```bash
# Run Vasicek model demo
./bin/vasicek_demo

# Run Heston model demo
./bin/heston_demo
```

## Running Tests

```bash
# Run all tests
cd build
ctest --output-on-failure

# Run specific test
./tests/test_vasicek
```

## Documentation

To generate API documentation (requires Doxygen):

```bash
# Install Doxygen
# On Ubuntu/Debian: sudo apt-get install doxygen
# On macOS: brew install doxygen

# Generate documentation
cd docs
doxygen Doxyfile

# Open documentation
open html/index.html  # macOS
xdg-open html/index.html  # Linux
```

## Performance Considerations

- For optimal performance, build in Release mode
- The number of threads can be controlled using the `OMP_NUM_THREADS` environment variable
- For large-scale simulations, consider implementing GPU acceleration (future work)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by various academic papers and financial mathematics textbooks
- Uses Google Test for unit testing
- CMake build system for cross-platform compatibility
