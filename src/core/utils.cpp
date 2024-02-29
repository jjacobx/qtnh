#include <functional>
#include <numeric>

#include "core/utils.hpp"

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

    qtnh::tidx_tup concat_dims(qtnh::tidx_tup dims1, qtnh::tidx_tup dims2) {
      qtnh::tidx_tup dims3;
      dims3.reserve(dims1.size() + dims2.size());
      dims3.insert(dims3.end(), dims1.begin(), dims1.end());
      dims3.insert(dims3.end(), dims2.begin(), dims2.end());

      return dims3;
    }

    std::pair<qtnh::tidx_tup, qtnh::tidx_tup> split_dims(qtnh::tidx_tup dims, qtnh::tidx_tup_st n) {
      return { qtnh::tidx_tup(dims.begin(), dims.begin() + n), qtnh::tidx_tup(dims.begin() + n, dims.end()) };
    }

    std::vector<qtnh::wire> invert_wires(std::vector<qtnh::wire> wires) {
      for (auto& w : wires) {
        std::swap(w.first, w.second);
      }

      return wires;
    }
  }
}