#include <functional>
#include <iostream>
#include <mpi.h>
#include <numeric>

#include "core/utils.hpp"

namespace qtnh {
  namespace utils {
    void throw_unimplemented() { 
      throw std::runtime_error("Unimplemented function!"); 
    }

    bool is_root() {
      int proc_id;
      MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
      return (proc_id == 0);
    }

    void barrier() {
      MPI_Barrier(MPI_COMM_WORLD);
    }

    void report(std::string s) {
      static auto global_counter = 0UL;
      if (is_root()) std::cout << "REPORTING " << s << " AT " << global_counter++ << "\n";
    }

    std::size_t dims_to_size(qtnh::tidx_tup dims) { 
      auto size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<qtnh::tidx>()); 
      return std::size_t(size);
    }

    std::size_t idxs_to_i(qtnh::tidx_tup idxs, qtnh::tidx_tup dims) { 
      std::size_t i = 0;
      std::size_t base = 1;
      for (auto j = idxs.size(); j > 0; --j) {
          i += idxs.at(j - 1) * base;
          base *= dims.at(j - 1);
      }

      return i;
    }

    qtnh::tidx_tup i_to_idxs(std::size_t i, qtnh::tidx_tup dims) {
      auto idxs = dims;
      for (auto j = dims.size(); j > 0; --j) {
        idxs.at(j - 1) = i % dims.at(j - 1);
        i /= dims.at(j - 1);
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

    qtnh::tidx_tup halve_dims(qtnh::tidx_tup dims) {
      return qtnh::tidx_tup(dims.begin(), dims.begin() + dims.size() / 2);
    }

    std::vector<qtnh::wire> invert_wires(std::vector<qtnh::wire> wires) {
      for (auto& w : wires) {
        std::swap(w.first, w.second);
      }

      return wires;
    }

    bool equal(qtnh::tel a, qtnh::tel b, double tol) {
      auto diff_mod_sq = std::pow(a.real() - b.real(), 2.0) + std::pow(a.imag() - b.imag(), 2.0);
      return std::sqrt(diff_mod_sq) < tol;
    }

    bool compatible(qtnh::tidx_tup dims1, qtnh::tidx_tup dims2) {
      return (dims_to_size(dims1) == dims_to_size(dims2));
    }
  }
}