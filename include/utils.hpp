#include "typedefs.hpp"

namespace qtnh {
  namespace utils {
    void throw_unimplemented();

    std::size_t dims_to_size(qtnh::tidx_tup);
    std::size_t idxs_to_i(qtnh::tidx_tup, qtnh::tidx_tup);
    qtnh::tidx_tup i_to_idxs(std::size_t, qtnh::tidx_tup);
  }
}