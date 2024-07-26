#ifndef _CORE__UTILS_HPP
#define _CORE__UTILS_HPP

#include <memory>

#include "typedefs.hpp"

namespace qtnh {
  namespace utils {
    /// Indicate a method is unimplemented. 
    /// Throws error if invoked. 
    void throw_unimplemented();

    /// @brief Convert tensor index dimensions tuple to tensor size. 
    /// @param dims Tensor index dimensions. 
    /// @return Number of tensor elements. 
    std::size_t dims_to_size(qtnh::tidx_tup dims);

    /// @brief Convert tensor indices tuple to array index. 
    /// @param idxs Tuple of indices to convert. 
    /// @param dims Tensor index dimensions. 
    /// @return Array index, assuming all elements are stored sequentially. 
    std::size_t idxs_to_i(qtnh::tidx_tup idxs, qtnh::tidx_tup dims);

    /// @brief Convert array index back to tensor indices tuple. 
    /// @param i Array index, assuming all elements are stored sequentially. 
    /// @param dims Tensor index dimensions. 
    /// @return Tuple of tensor indices pointing at the element. 
    qtnh::tidx_tup i_to_idxs(std::size_t i, qtnh::tidx_tup dims);

    /// @brief Concatenate two arrays of tensor index dimensions tuples. 
    /// @param dims1 First index dimensions tuple. 
    /// @param dims2 Second index dimensions tuple. 
    /// @return Combined tensor index dimensions tuple. 
    qtnh::tidx_tup concat_dims(qtnh::tidx_tup dims1, qtnh::tidx_tup dims2);

    /// @brief Split tensor index dimensions tuple at given position. 
    /// @param dims Tensor index dimensions to split. 
    /// @param n Position of index dimension before which to insert the split. 
    /// @return A pair of tensor index dimensions tuples. 
    std::pair<qtnh::tidx_tup, qtnh::tidx_tup> split_dims(qtnh::tidx_tup dims, qtnh::tidx_tup_st n);

    
    
    /// @brief Invert the direction of tensor cotraction wires. 
    /// @param ws A vector of cotraction wires to invert. 
    /// @return A vector of cotraction wires, where each wire has a reversed direction. 
    std::vector<qtnh::wire> invert_wires(std::vector<qtnh::wire> ws);

    /// @brief Compare two complex elements within given tolerance. 
    /// @param a First complex element. 
    /// @param b Second complex element. 
    /// @param tol Maximum allowed magnitude of the difference between the elements (default 1E-5). 
    /// @return True if elements are approximately equal and false otherwise. 
    bool equal(qtnh::tel a, qtnh::tel b, double tol = 1E-5);

    template<typename T>
    std::unique_ptr<T> one_unique(std::unique_ptr<T> u, T* t) {
      if (u.get() == t) return u;
      else return std::unique_ptr<T>(t);
    }

    template<typename T, typename U>
    std::unique_ptr<T> one_unique(std::unique_ptr<U> u, T* t) {
      auto p = u.release();
      if (dynamic_cast<T*>(p) != t) delete p;
      return std::unique_ptr<T>(t);
    }

    template <typename T>
    std::pair<std::vector<T>, std::vector<T>> split_vec(std::vector<T> vec, std::size_t n) {
      return { std::vector<T>(vec.begin(), vec.begin() + n), std::vector<T>(vec.begin() + n, vec.end()) };
    }
  }

  namespace ops {
    template<typename T>
    std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
      for (std::size_t i = 0; i < v.size(); ++i) {
        out << v.at(i);
        if (i + 1 < v.size()) out << ", ";
      }

      return out;
    }
  }
}

#endif