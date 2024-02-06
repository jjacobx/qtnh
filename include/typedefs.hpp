#ifndef TYPEDEFS_HPP
#define TYPEDEFS_HPP

#include <complex>
#include <functional>
#include <numeric>
#include <vector>

namespace qtnh {
  typedef std::size_t tidx;
  typedef std::vector<qtnh::tidx> tidx_tup;
  typedef unsigned short int tidx_tup_st;

  enum class TIdxFlag { open, closed, self, oob = 99 };

  typedef std::vector<TIdxFlag> tidx_flags;
  typedef std::complex<double> tel;

  typedef std::pair<tidx, tidx> wire;

  inline void throw_unimplemented() { 
    throw std::runtime_error("Unimplemented funciton!"); 
  }
  inline std::size_t dims_to_size(qtnh::tidx_tup dims) { 
    return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<qtnh::tidx>()); 
  }
  inline std::size_t idxs_to_i(qtnh::tidx_tup idxs, qtnh::tidx_tup dims) { 
    std::size_t i = 0;
    std::size_t base = 1;
    for (int j = idxs.size() - 1; j >= 0; --j) {
        i += idxs.at(j) * base;
        base *= dims.at(j);
    }

    return i;
  }
}

#endif