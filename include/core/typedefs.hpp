#ifndef _CORE__TYPEDEFS_HPP
#define _CORE__TYPEDEFS_HPP

#include <complex>
#include <functional>
#include <numeric>
#include <vector>

#define ROOT_COUT \
int _r; MPI_Comm_rank(MPI_COMM_WORLD, &_r); \
if (!_r) std::cout

// By default broadcast shared tensors
// Can be switched off for performance reasons
#ifndef DEF_STENSOR_BCAST
#define DEF_STENSOR_BCAST 1
#endif

namespace qtnh {
  typedef std::size_t tidx;
  typedef std::vector<qtnh::tidx> tidx_tup;
  typedef unsigned short int tidx_tup_st;
  typedef unsigned int uint;

  enum class TIdxFlag { open, closed, self, oob = 99 };

  typedef std::vector<TIdxFlag> tidx_flags;
  typedef std::complex<double> tel;

  typedef std::pair<qtnh::tidx_tup_st, qtnh::tidx_tup_st> wire;
}

#endif