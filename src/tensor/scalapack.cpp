#include "tensor/scalapack.hpp"

namespace qtnh {
  namespace decomp {
    ProcGrid tensor_to_grid(Tensor* t, qtnh::tidx_tup_st split) {
      int zero = 0;
      int context;
      blacs_get_(&zero, &zero, &context);

      int nprows, npcols;
      auto dis_len = t->disDims().size();
      if (split >= dis_len) {
        nprows = int(t->disSize());
        npcols = 1;
      } else {
        auto [dims1, dims2] = utils::split_dims(t->disDims(), split);
        nprows = int(utils::dims_to_size(dims1));
        npcols = int(t->disSize() / nprows);
        (void)dims2;
      }

      auto layout = 'R';
      blacs_gridinit_(&context, &layout, &nprows, &npcols);

      int myrow, mycol;
      blacs_gridinfo_(&context, &nprows, &npcols, &myrow, &mycol);

      return { context, myrow, mycol };
    }
  }
}
