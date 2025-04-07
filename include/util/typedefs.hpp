#ifndef __UTIL_TYPEDEFS__
#define __UTIL_TYPEDEFS__

#include <complex>
#include <limits>
#include <functional>
#include <numeric>
#include <vector>

// Helper macro to print from root
#define ROOT_COUT \
int _r; MPI_Comm_rank(MPI_COMM_WORLD, &_r); \
if (!_r) std::cout

namespace qtnh {
  using tidx = std::size_t;          ///< Tensor index dimensions. 
  using uint = unsigned int;         ///< Unsigned int for IDs. 

  using tidx_tup = std::vector<qtnh::tidx>;  ///< Tuple of tensor indices – used for accessing tensor elements. 
  // * This might be useful in the future if tuples are reimplemented with a smaller size type. 
  using tidx_tup_st = std::size_t;           ///< Tensor index tuple dimensions. 

  using tel = std::complex<double>;  ///< Tensor element type. 
  
  using wire = std::pair<qtnh::tidx_tup_st, qtnh::tidx_tup_st>; ///< A pair of contracted indices. 

  constexpr std::size_t X = std::numeric_limits<std::size_t>::max();
  constexpr double ZERO_TOL = 1E-12;
}

#endif