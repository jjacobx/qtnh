#ifndef _CORE__TYPEDEFS_HPP
#define _CORE__TYPEDEFS_HPP

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

  /// Broadcaster parameters container for sharing tensors across processes. 
  struct BcParams {
    qtnh::uint str;  ///< Number of times each local tensor chunk is repeated across contiguous processes. 
    qtnh::uint cyc;  ///< Number of times the entire tensor structure is repeated. 
    qtnh::uint off;  ///< Number of empty processes before the tensor begins. 
  };

  /// Tensor type labels for determining contraction function to use. 
  enum class TT {
    tensor, 
    denseTensorBase, 
    denseTensor, 
    rescTensor, 
    symmTensorBase, 
    symmTensor, 
    swapTensor, 
    diagTensorBase, 
    diagTensor, 
    idenTensor
  };

  /// Tensor index input/output labels for symmetric tensors. 
  enum class TIdxIO {
    in,   ///< Input indices are the first parts of distributed/local dimension arrays. 
    out   ///< Output indices are the second parts of distributed/local dimension arrays. 
  };

  constexpr std::size_t X = std::numeric_limits<std::size_t>::max();
}

#endif