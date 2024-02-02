#ifndef TYPEDEFS_HPP
#define TYPEDEFS_HPP

#include <complex>
#include <vector>

namespace qtnh {
  typedef std::size_t tidx;
  typedef std::vector<qtnh::tidx> tidx_tup;
  typedef unsigned short int tidx_tup_st;

  enum class TIdxFlag { open, closed, oob };

  typedef std::vector<TIdxFlag> tidx_flags;
  typedef std::complex<double> tel;

  typedef std::pair<tidx, tidx> wire;
}

#endif