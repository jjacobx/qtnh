#ifndef _CORE__UTILS_HPP
#define _CORE__UTILS_HPP

#include "typedefs.hpp"

namespace qtnh {
  /// Namespace for helper functions. 
  namespace utils {
    /// Indicate a method is unimplemented. 
    /// Throws error if invoked. 
    void throw_unimplemented();

    /// @brief Convert tensor index dimensions tuple to tensor size. 
    /// @param dims tensor index dimensions. 
    /// @return Number of tensor elements. 
    std::size_t dims_to_size(qtnh::tidx_tup);

    /// @brief Convert tensor indices tuple to array index. 
    /// @param idxs tuple of indices to convert. 
    /// @param dims tensor index dimensions. 
    /// @return Array index, assuming all elements are stored sequentially. 
    std::size_t idxs_to_i(qtnh::tidx_tup, qtnh::tidx_tup);

    /// @brief Convert array index back to tensor indices tuple. 
    /// @param i array index, assuming all elements are stored sequentially. 
    /// @param dims tensor index dimensions. 
    /// @return Tuple of tensor indices pointing at the element. 
    qtnh::tidx_tup i_to_idxs(std::size_t, qtnh::tidx_tup);

    /// @brief Concatenate two arrays of tensor index dimensions tuples. 
    /// @param dims1 first index dimensions tuple. 
    /// @param dims2 second index dimensions tuple. 
    /// @return Combined tensor index dimensions tuple. 
    qtnh::tidx_tup concat_dims(qtnh::tidx_tup, qtnh::tidx_tup);

    /// @brief Split tensor index dimensions tuple at given position. 
    /// @param dims tensor index dimensions to split. 
    /// @param n position of index dimension before which to insert the split. 
    /// @return A pair of tensor index dimensions tuples. 
    std::pair<qtnh::tidx_tup, qtnh::tidx_tup> split_dims(qtnh::tidx_tup, qtnh::tidx_tup_st);
    
    /// @brief Invert the direction of tensor cotraction wires. 
    /// @param wires a vector of cotraction wires to invert. 
    /// @return A vector of cotraction wires, where each wire has a reversed direction. 
    std::vector<qtnh::wire> invert_wires(std::vector<qtnh::wire>);
  }
}

#endif