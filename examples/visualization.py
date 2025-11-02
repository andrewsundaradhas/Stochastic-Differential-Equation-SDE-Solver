"""
SDE Solver Visualization Examples
--------------------------------

This script demonstrates how to use the SDE Solver library to simulate
and visualize stochastic differential equations used in quantitative finance.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import os
import sys

# Add the build directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build'))
try:
    import sde_solver
except ImportError:
    print("Error: Could not import sde_solver. Make sure to build the Python bindings first.")
    sys.exit(1)

def plot_vasicek_paths():
    """Simulate and plot paths from the Vasicek model."""
    print("Running Vasicek model simulation...")
    
    # Model parameters
    kappa = 1.0    # Mean reversion speed
    theta = 0.1    # Long-term mean
    sigma = 0.2    # Volatility
    x0 = 0.05      # Initial value
    
    # Simulation parameters
    T = 1.0          # Time horizon
    steps = 252      # Number of time steps (daily for 1 year)
    num_paths = 20   # Number of paths to simulate
    
    # Create model and solver
    model = sde_solver.VasicekModel(kappa, theta, sigma, x0)
    solver = sde_solver.EulerMaruyamaSolver()
    
    # Run simulation
    result = sde_solver.simulate(model, solver, T, steps, num_paths)
    
    # Create time array
    time = np.array(result.time)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Plot individual paths
    for path in result.paths:
        plt.plot(time, path, 'b-', alpha=0.3, linewidth=0.8)
    
    # Plot mean and confidence intervals
    plt.plot(time, result.mean, 'r-', linewidth=2, label='Sample Mean')
    plt.fill_between(time, 
                    [m - 2*s for m, s in zip(result.mean, result.std_dev)],
                    [m + 2*s for m, s in zip(result.mean, result.std_dev)],
                    color='r', alpha=0.2, label='95% Confidence')
    
    # Plot analytical mean and confidence intervals
    analytical_mean = [model.expected_value(t, x0) for t in time]
    analytical_std = [np.sqrt(model.variance(t)) for t in time]
    plt.plot(time, analytical_mean, 'k--', linewidth=2, label='Theoretical Mean')
    plt.fill_between(time, 
                    [m - 2*s for m, s in zip(analytical_mean, analytical_std)],
                    [m + 2*s for m, s in zip(analytical_mean, analytical_std)],
                    color='gray', alpha=0.2, label='Theoretical 95% Confidence')
    
    plt.title(f'Vasicek Model Simulation (κ={kappa}, θ={theta}, σ={sigma}, x0={x0})')
    plt.xlabel('Time (years)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save figure
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/vasicek_paths.png', dpi=300, bbox_inches='tight')
    print("Saved figure to 'figures/vasicek_paths.png'")

def plot_heston_paths():
    """Simulate and plot paths from the Heston model."""
    print("Running Heston model simulation...")
    
    # Model parameters
    mu = 0.05      # Drift
    kappa = 2.0    # Mean reversion speed of variance
    theta = 0.04   # Long-term variance (20% vol)
    xi = 0.3       # Vol of vol
    rho = -0.7     # Correlation
    s0 = 100.0     # Initial asset price
    v0 = 0.04      # Initial variance (20% vol)
    
    # Simulation parameters
    T = 1.0          # Time horizon
    steps = 252      # Number of time steps (daily for 1 year)
    num_paths = 10   # Number of paths to simulate
    
    # Create model and solver
    model = sde_solver.HestonModel(mu, kappa, theta, xi, rho, s0, v0)
    solver = sde_solver.MilsteinSolver()
    
    # Run simulation
    result = sde_solver.simulate(model, solver, T, steps, num_paths)
    
    # Create time array
    time = np.array(result.time)
    
    # Reshape results (since Heston model has 2 dimensions: [S, v])
    price_paths = np.array([path[::2] for path in result.paths])  # Even indices are prices
    vol_paths = np.array([path[1::2] for path in result.paths])   # Odd indices are variances
    vol_paths = np.sqrt(vol_paths)  # Convert variance to volatility
    
    # Plot price paths
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 1, 1)
    for path in price_paths:
        plt.plot(time, path, alpha=0.6, linewidth=0.8)
    
    plt.title(f'Heston Model - Price Paths (ρ={rho:.1f})')
    plt.xlabel('Time (years)')
    plt.ylabel('Asset Price')
    plt.grid(True)
    
    # Plot volatility paths
    plt.subplot(2, 1, 2)
    for path in vol_paths:
        plt.plot(time, path, alpha=0.6, linewidth=0.8)
    
    plt.title('Stochastic Volatility Paths')
    plt.xlabel('Time (years)')
    plt.ylabel('Volatility')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/heston_paths.png', dpi=300, bbox_inches='tight')
    print("Saved figure to 'figures/heston_paths.png'")

def plot_solver_comparison():
    """Compare different SDE solvers on the Vasicek model."""
    print("Running solver comparison...")
    
    # Model parameters
    kappa = 1.0    # Mean reversion speed
    theta = 0.1    # Long-term mean
    sigma = 0.2    # Volatility
    x0 = 0.05      # Initial value
    
    # Simulation parameters
    T = 1.0          # Time horizon
    steps = 252      # Number of time steps
    num_paths = 1000  # Number of paths for statistics
    
    # Create model and solvers
    model = sde_solver.VasicekModel(kappa, theta, sigma, x0)
    euler = sde_solver.EulerMaruyamaSolver()
    milstein = sde_solver.MilsteinSolver()
    rk = sde_solver.RungeKuttaSolver()
    
    # Run simulations
    print("Running Euler-Maruyama...")
    result_euler = sde_solver.simulate(model, euler, T, steps, num_paths)
    
    print("Running Milstein...")
    result_milstein = sde_solver.simulate(model, milstein, T, steps, num_paths)
    
    print("Running Runge-Kutta...")
    result_rk = sde_solver.simulate(model, rk, T, steps, num_paths)
    
    # Create time array
    time = np.array(result_euler.time)
    
    # Compute errors from analytical solution
    analytical_mean = np.array([model.expected_value(t, x0) for t in time])
    
    euler_error = np.abs(np.array(result_euler.mean) - analytical_mean)
    milstein_error = np.abs(np.array(result_milstein.mean) - analytical_mean)
    rk_error = np.abs(np.array(result_rk.mean) - analytical_mean)
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    plt.semilogy(time, euler_error, 'b-', label='Euler-Maruyama')
    plt.semilogy(time, milstein_error, 'r-', label='Milstein')
    plt.semilogy(time, rk_error, 'g-', label='Runge-Kutta')
    
    plt.title('Absolute Error from Analytical Solution')
    plt.xlabel('Time (years)')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Save figure
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/solver_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved figure to 'figures/solver_comparison.png'")

def plot_volatility_surface():
    """Generate a volatility surface using the Heston model."""
    print("Generating volatility surface...")
    
    # Model parameters
    mu = 0.05      # Drift
    kappa = 2.0    # Mean reversion speed of variance
    theta = 0.04   # Long-term variance (20% vol)
    xi = 0.3       # Vol of vol
    rho = -0.7     # Correlation
    s0 = 100.0     # Initial asset price
    v0 = 0.04      # Initial variance (20% vol)
    
    # Simulation parameters
    T = 1.0          # Time horizon
    steps = 252      # Number of time steps
    num_paths = 5000  # Number of paths for accurate implied vol
    
    # Create model and solver
    model = sde_solver.HestonModel(mu, kappa, theta, xi, rho, s0, v0)
    solver = sde_solver.MilsteinSolver()
    
    # Run simulation
    result = sde_solver.simulate(model, solver, T, steps, num_paths)
    
    # Extract terminal prices (last time step, price dimension)
    terminal_prices = np.array([path[-2] for path in result.paths])
    
    # Define moneyness and maturity grid
    moneyness = np.linspace(0.8, 1.2, 20)  # S/K ratio
    maturities = np.linspace(0.1, 1.0, 10)  # Years
    
    # Calculate implied volatilities (simplified for demonstration)
    # In practice, you would use a proper implied vol calculator
    implied_vols = np.zeros((len(moneyness), len(maturities)))
    
    for i, k in enumerate(moneyness):
        strike = s0 * k
        for j, t in enumerate(maturities):
            # Simplified implied vol calculation (for demonstration)
            # In practice, use a proper BS formula
            itm = terminal_prices > strike
            call_prices = np.maximum(0, terminal_prices - strike)
            avg_call = np.mean(call_prices)
            
            # Simplified BS implied vol (for demonstration)
            if avg_call > 0:
                # This is a very rough approximation
                implied_vols[i, j] = np.sqrt(2 * np.pi / t) * avg_call / s0
            else:
                implied_vols[i, j] = 0.0
    
    # Create 3D plot
    X, Y = np.meshgrid(maturities, moneyness)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, implied_vols, cmap=cm.viridis,
                          linewidth=0, antialiased=True)
    
    ax.set_xlabel('Time to Maturity (years)')
    ax.set_ylabel('Moneyness (S/K)')
    ax.set_zlabel('Implied Volatility')
    ax.set_title('Heston Model Implied Volatility Surface')
    
    # Add color bar
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    # Save figure
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/volatility_surface.png', dpi=300, bbox_inches='tight')
    print("Saved figure to 'figures/volatility_surface.png'")

if __name__ == "__main__":
    # Create figures directory
    os.makedirs('figures', exist_ok=True)
    
    # Run examples
    plot_vasicek_paths()
    plot_heston_paths()
    plot_solver_comparison()
    plot_volatility_surface()
    
    print("\nAll examples completed. Check the 'figures' directory for output plots.")
    plt.show()
