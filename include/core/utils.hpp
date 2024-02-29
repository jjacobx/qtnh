#ifndef _CORE__UTILS_HPP
#define _CORE__UTILS_HPP

#include "typedefs.hpp"

namespace qtnh {
  namespace utils {
    void throw_unimplemented();

    std::size_t dims_to_size(qtnh::tidx_tup);
    std::size_t idxs_to_i(qtnh::tidx_tup, qtnh::tidx_tup);
    qtnh::tidx_tup i_to_idxs(std::size_t, qtnh::tidx_tup);

    qtnh::tidx_tup concat_dims(qtnh::tidx_tup, qtnh::tidx_tup);
    std::pair<qtnh::tidx_tup, qtnh::tidx_tup> split_dims(qtnh::tidx_tup, qtnh::tidx_tup_st);
    
    std::vector<qtnh::wire> invert_wires(std::vector<qtnh::wire> wires);
  }
}

#endif