#ifndef __UTIL_TYPEDEFS__
#define __UTIL_TYPEDEFS__

#include <complex>
#include <limits>
#include <functional>
#include <mpi.h>
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
  using tidx_tup_st = std::size_t;                      ///< tidx_tup position. 
  using tidx_tup_ids = std::vector<qtnh::tidx_tup_st>;  ///< Group of tidx_tup positions. 

  using tel = std::complex<double>;  ///< Tensor element type. 
  
  using wire = std::pair<qtnh::tidx_tup_st, qtnh::tidx_tup_st>; ///< A pair of contracted indices. 

  using tel_fun = qtnh::tel(*)(qtnh::tel, qtnh::tel);
  using mpi_fun = void(*)(void*, void*, int*, MPI_Datatype*);

  constexpr std::size_t X = std::numeric_limits<qtnh::tidx_tup_st>::max();
  constexpr double ZERO_TOL = 1E-10;
}

#endif