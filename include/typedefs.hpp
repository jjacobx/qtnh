#ifndef TYPEDEFS_HPP
#define TYPEDEFS_HPP

#include <complex>
#include <vector>

namespace qtnh {
  typedef std::size_t tidx;
  typedef std::vector<qtnh::tidx> tidx_tup;
  typedef unsigned short int tidx_tup_st;

  enum class TIdxFlag { open, closed, self, oob = 99 };

  typedef std::vector<TIdxFlag> tidx_flags;
  typedef std::complex<double> tel;

  typedef std::pair<tidx, tidx> wire;

  inline void throw_unimplemented() { throw std::runtime_error("Unimplemented funciton!"); }
}

#endif