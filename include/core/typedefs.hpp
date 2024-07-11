#ifndef _CORE__TYPEDEFS_HPP
#define _CORE__TYPEDEFS_HPP

#include <complex>
#include <functional>
#include <numeric>
#include <vector>

// Helper macro to print from root
#define ROOT_COUT \
int _r; MPI_Comm_rank(MPI_COMM_WORLD, &_r); \
if (!_r) std::cout

// By default broadcast shared tensors
// Can be switched off for performance reasons
#ifndef DEF_STENSOR_BCAST
#define DEF_STENSOR_BCAST 1
#endif

namespace qtnh {
  typedef std::size_t tidx;                ///< Tensor index dimensions. 
  typedef unsigned short int tidx_tup_st;  ///< Tensor index tuple dimensions. 
  typedef unsigned int uint;               ///< Unsigned int for IDs. 

  /// Tensor index type labels for contraction. 
  enum class TIdxT { 
    open,     ///< Open indices – increase independenly on contracted tensors. 
    closed,   ///< Closed indices – increase in pairs on contracted tensors. 
    oob = 99  ///< Out-of-bounds label – indicates that index went out of scope. 
  };
  typedef std::pair<TIdxT, qtnh::tidx_tup_st> tifl;  ///< Tensor index flag – consists of a type and a tag. 

  typedef std::vector<qtnh::tidx> tidx_tup;  ///< Tuple of tensor indices – used for accessing tensor elements. 
  typedef std::vector<qtnh::tifl> tifl_tup;  ///< Tuple of tensor flags – stores labels of all indices under contracion. 

  typedef std::complex<double> tel;  ///< Tensor element type. 

  typedef std::pair<qtnh::tidx_tup_st, qtnh::tidx_tup_st> wire; ///< A pair of contracted indices. 

  /// Broadcaster parameters container for sharing tensors across processes. 
  struct BcParams {
    qtnh::uint str;  ///< Number of times each local tensor chunk is repeated across contiguous processes. 
    qtnh::uint cyc;  ///< Number of times the entire tensor structure is repeated. 
    qtnh::uint off;  ///< Number of empty processes before the tensor begins. 

    /// Simple constructor of all tensor parameters. 
    BcParams(qtnh::uint str, qtnh::uint cyc, qtnh::uint off) 
      : str(str), cyc(cyc), off(off) {}
  };

  /// Tensor type labels for determining contraction function to use. 
  enum class TT {
    tensor, 
    denseTensorBase, 
    denseTensor, 
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
}

#endif