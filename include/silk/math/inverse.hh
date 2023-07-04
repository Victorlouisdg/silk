#pragma once
#include <Eigen/Core>

namespace silk {
    /**
     * @brief Calculate the inverses of a vector of matrices.
     * 
     * @param M A vector of invertable matrices.
     * @return A vector of inverted matrices.
     */
    template <typename T>
    std::vector<T> inverse(std::vector<T> &M) {
        std::vector<T> M_inv;
        M_inv.reserve(M.size());
        for (auto &m : M) {
            M_inv.push_back(m.inverse());
        }
        return M_inv;
    }
}