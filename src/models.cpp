#include "models.hpp"
#include <stdexcept>

namespace sde {

std::vector<double> RNGPool::correlated_normal(const std::vector<std::vector<double>>& correlation_matrix) {
    std::size_t n = correlation_matrix.size();
    if (n == 0 || correlation_matrix[0].size() != n) {
        throw std::invalid_argument("Correlation matrix must be square");
    }
    return normal_vector(n);
}

} // namespace sde
