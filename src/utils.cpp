#include <functional>
#include <numeric>

#include "utils.hpp"

namespace qtnh {
  namespace utils {
    void throw_unimplemented() { 
      throw std::runtime_error("Unimplemented funciton!"); 
    }

    std::size_t dims_to_size(qtnh::tidx_tup dims) { 
      return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<qtnh::tidx>()); 
    }

    std::size_t idxs_to_i(qtnh::tidx_tup idxs, qtnh::tidx_tup dims) { 
      std::size_t i = 0;
      std::size_t base = 1;
      for (int j = idxs.size() - 1; j >= 0; --j) {
          i += idxs.at(j) * base;
          base *= dims.at(j);
      }

      return i;
    }

    qtnh::tidx_tup i_to_idxs(std::size_t i, qtnh::tidx_tup dims) {
      auto idxs = dims;
      for (int j = dims.size() - 1; j >= 0; --j) {
        idxs.at(j) = i % dims.at(j);
        i /= dims.at(j);
      }

      return idxs;
    }
  }
}