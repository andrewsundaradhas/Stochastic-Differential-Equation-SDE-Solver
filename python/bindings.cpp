#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include "../include/models.hpp"
#include "../include/solvers.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

// Convert between std::vector and numpy array
py::array_t<double> vector_to_np(const std::vector<double>& vec) {
    return py::array_t<double>(vec.size(), vec.data());
}

std::vector<double> np_to_vector(const py::array_t<double>& arr) {
    auto buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Number of dimensions must be 1");
    }
    double* ptr = static_cast<double*>(buf.ptr);
    return std::vector<double>(ptr, ptr + buf.size);
}

// Simulation result structure
struct SimulationResult {
    std::vector<double> time;
    std::vector<std::vector<double>> paths;
    std::vector<double> mean;
    std::vector<double> std_dev;
};

// Function to run simulation
SimulationResult run_simulation(
    const sde::SDE& model,
    const sde::SDESolver& solver,
    double T,
    size_t steps,
    size_t num_paths,
    unsigned int seed = 42) {
    
    const double dt = T / steps;
    const size_t dim = model.dimension();
    const size_t w_dim = model.wiener_dimension();
    
    // Initialize RNG
    sde::RNGPool rng_pool(std::thread::hardware_concurrency(), seed);
    
    // Initialize result storage
    SimulationResult result;
    result.time.resize(steps + 1);
    result.paths.resize(num_paths, std::vector<double>((steps + 1) * dim));
    result.mean.resize((steps + 1) * dim, 0.0);
    result.std_dev.resize((steps + 1) * dim, 0.0);
    
    // Initialize time points
    for (size_t i = 0; i <= steps; ++i) {
        result.time[i] = i * dt;
    }
    
    // Run simulation
    #pragma omp parallel for
    for (size_t p = 0; p < num_paths; ++p) {
        // Initialize state
        sde::Vector x;
        if (auto vasicek = dynamic_cast<const sde::VasicekModel*>(&model)) {
            x = {vasicek->x0()};
        } else if (auto heston = dynamic_cast<const sde::HestonModel*>(&model)) {
            x = {heston->s0(), heston->v0()};
        } else {
            x = sde::Vector(dim, 0.0);
        }
        
        // Store initial state
        for (size_t d = 0; d < dim; ++d) {
            result.paths[p][d] = x[d];
        }
        
        // Time stepping
        for (size_t i = 1; i <= steps; ++i) {
            // Generate Wiener increments
            std::vector<double> dW(w_dim);
            for (size_t j = 0; j < w_dim; ++j) {
                dW[j] = rng_pool.normal() * std::sqrt(dt);
            }
            
            // Take a step
            solver.step(model, x, (i-1)*dt, dt, dW);
            
            // Ensure non-negative variance for Heston model
            if (dynamic_cast<const sde::HestonModel*>(&model)) {
                x[1] = std::max(0.0, x[1]);
            }
            
            // Store state
            for (size_t d = 0; d < dim; ++d) {
                result.paths[p][i * dim + d] = x[d];
            }
        }
    }
    
    // Compute statistics
    for (size_t i = 0; i <= steps; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            // Compute mean
            double sum = 0.0;
            for (size_t p = 0; p < num_paths; ++p) {
                sum += result.paths[p][i * dim + d];
            }
            result.mean[i * dim + d] = sum / num_paths;
            
            // Compute standard deviation
            double sum_sq = 0.0;
            for (size_t p = 0; p < num_paths; ++p) {
                double diff = result.paths[p][i * dim + d] - result.mean[i * dim + d];
                sum_sq += diff * diff;
            }
            result.std_dev[i * dim + d] = std::sqrt(sum_sq / (num_paths - 1));
        }
    }
    
    return result;
}

PYBIND11_MODULE(sde_solver, m) {
    m.doc() = "Python bindings for SDE Solver";
    
    // Vector type
    py::class_<std::vector<double>>(m, "Vector")
        .def(py::init<>())
        .def("__getitem__", [](const std::vector<double> &v, int i) {
            if (i < 0) i += v.size();
            if (i < 0 || i >= v.size()) throw py::index_error();
            return v[i];
        })
        .def("__setitem__", [](std::vector<double> &v, int i, double value) {
            if (i < 0) i += v.size();
            if (i < 0 || i >= v.size()) throw py::index_error();
            v[i] = value;
        })
        .def("__len__", &std::vector<double>::size)
        .def("__iter__", [](const std::vector<double> &v) {
            return py::make_iterator(v.begin(), v.end());
        }, py::keep_alive<0, 1>());
    
    // Simulation result
    py::class_<SimulationResult>(m, "SimulationResult")
        .def_readonly("time", &SimulationResult::time)
        .def_readonly("paths", &SimulationResult::paths)
        .def_readonly("mean", &SimulationResult::mean)
        .def_readonly("std_dev", &SimulationResult::std_dev)
        .def("__str__", [](const SimulationResult &r) {
            return "SimulationResult(time=" + std::to_string(r.time.size()) + 
                   " points, paths=" + std::to_string(r.paths.size()) + ")";
        });
    
    // SDE base class
    py::class_<sde::SDE, std::shared_ptr<sde::SDE>>(m, "SDE")
        .def("dimension", &sde::SDE::dimension)
        .def("wiener_dimension", &sde::SDE::wiener_dimension);
    
    // Vasicek Model
    py::class_<sde::VasicekModel, sde::SDE, std::shared_ptr<sde::VasicekModel>>(m, "VasicekModel")
        .def(py::init<double, double, double, double>(),
             "kappa"_a, "theta"_a, "sigma"_a, "x0"_a = 0.0)
        .def("expected_value", &sde::VasicekModel::expected_value)
        .def("variance", &sde::VasicekModel::variance)
        .def_property_readonly("kappa", &sde::VasicekModel::kappa)
        .def_property_readonly("theta", &sde::VasicekModel::theta)
        .def_property_readonly("sigma", &sde::VasicekModel::sigma)
        .def_property_readonly("x0", &sde::VasicekModel::x0);
    
    // Heston Model
    py::class_<sde::HestonModel, sde::SDE, std::shared_ptr<sde::HestonModel>>(m, "HestonModel")
        .def(py::init<double, double, double, double, double, double, double>(),
             "mu"_a, "kappa"_a, "theta"_a, "xi"_a, "rho"_a,
             "s0"_a = 100.0, "v0"_a = 0.04)
        .def_property_readonly("mu", &sde::HestonModel::mu)
        .def_property_readonly("kappa", &sde::HestonModel::kappa)
        .def_property_readonly("theta", &sde::HestonModel::theta)
        .def_property_readonly("xi", &sde::HestonModel::xi)
        .def_property_readonly("rho", &sde::HestonModel::rho)
        .def_property_readonly("s0", &sde::HestonModel::s0)
        .def_property_readonly("v0", &sde::HestonModel::v0);
    
    // SDESolver base class
    py::class_<sde::SDESolver, std::shared_ptr<sde::SDESolver>>(m, "SDESolver");
    
    // Euler-Maruyama Solver
    py::class_<sde::EulerMaruyamaSolver, sde::SDESolver, 
               std::shared_ptr<sde::EulerMaruyamaSolver>>(m, "EulerMaruyamaSolver")
        .def(py::init<>())
        .def("__str__", [](const sde::EulerMaruyamaSolver &s) {
            return "EulerMaruyamaSolver()";
        });
    
    // Milstein Solver
    py::class_<sde::MilsteinSolver, sde::SDESolver, 
               std::shared_ptr<sde::MilsteinSolver>>(m, "MilsteinSolver")
        .def(py::init<>())
        .def("__str__", [](const sde::MilsteinSolver &s) {
            return "MilsteinSolver()";
        });
    
    // Runge-Kutta Solver
    class RungeKuttaSolver : public sde::SDESolver {
    public:
        void step(const sde::SDE& sde, sde::Vector& x, double t, double dt, 
                 const sde::Vector& dW) const override {
            const size_t dim = sde.dimension();
            const size_t w_dim = sde.wiener_dimension();
            
            sde::Vector k1(dim), k2(dim), k3(dim), k4(dim);
            sde::Vector x_temp = x;
            sde::Vector a(dim);
            sde::Matrix b(dim, sde::Vector(w_dim));
            
            // Stage 1
            sde.drift(x, a, t);
            sde.diffusion(x, b, t);
            for (size_t i = 0; i < dim; ++i) {
                k1[i] = a[i] * dt;
                for (size_t j = 0; j < w_dim; ++j) {
                    k1[i] += b[i][j] * dW[j];
                }
            }
            
            // Stage 2
            for (size_t i = 0; i < dim; ++i) {
                x_temp[i] = x[i] + 0.5 * k1[i];
            }
            sde.drift(x_temp, a, t + 0.5 * dt);
            sde.diffusion(x_temp, b, t + 0.5 * dt);
            for (size_t i = 0; i < dim; ++i) {
                k2[i] = a[i] * dt;
                for (size_t j = 0; j < w_dim; ++j) {
                    k2[i] += b[i][j] * dW[j];
                }
            }
            
            // Stage 3
            for (size_t i = 0; i < dim; ++i) {
                x_temp[i] = x[i] + 0.5 * k2[i];
            }
            sde.drift(x_temp, a, t + 0.5 * dt);
            sde.diffusion(x_temp, b, t + 0.5 * dt);
            for (size_t i = 0; i < dim; ++i) {
                k3[i] = a[i] * dt;
                for (size_t j = 0; j < w_dim; ++j) {
                    k3[i] += b[i][j] * dW[j];
                }
            }
            
            // Stage 4
            for (size_t i = 0; i < dim; ++i) {
                x_temp[i] = x[i] + k3[i];
            }
            sde.drift(x_temp, a, t + dt);
            sde.diffusion(x_temp, b, t + dt);
            for (size_t i = 0; i < dim; ++i) {
                k4[i] = a[i] * dt;
                for (size_t j = 0; j < w_dim; ++j) {
                    k4[i] += b[i][j] * dW[j];
                }
            }
            
            // Update state
            for (size_t i = 0; i < dim; ++i) {
                x[i] += (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
            }
        }
        
        const char* name() const override { return "Runge-Kutta"; }
    };
    
    py::class_<RungeKuttaSolver, sde::SDESolver, 
               std::shared_ptr<RungeKuttaSolver>>(m, "RungeKuttaSolver")
        .def(py::init<>())
        .def("__str__", [](const RungeKuttaSolver &s) {
            return "RungeKuttaSolver()";
        });
    
    // Simulation function
    m.def("simulate", &run_simulation,
          "model"_a, "solver"_a, "T"_a, "steps"_a, "num_paths"_a, "seed"_a = 42,
          "Run a Monte Carlo simulation of an SDE");
    
    // Version info
    m.attr("__version__") = "1.0.0";
    
    // Add docstrings
    #ifdef VERSION_INFO
        m.attr("__version__") = VERSION_INFO;
    #else
        m.attr("__version__") = "dev";
    #endif
}
