#include "lalg/matrix.hpp"
#include "lalg/routines.hpp"

namespace qtnh {
  namespace lalg {
    BlockMatrix::BlockMatrix(int nrows, int ncols, int nprows, int npcols)
      : BlockMatrix(nrows, ncols, nprows, npcols, nullptr) {
      auto loc_size = (nrows * ncols) / (nprows * npcols);
      loc_els_p_ = std::make_unique<cvec>(loc_size);
    }
    
    BlockMatrix::BlockMatrix(int nrows, int ncols, int nprows, int npcols, cvec* loc_els_p)
      : nrows_(nrows), ncols_(ncols), nprows_(nprows), npcols_(npcols), loc_els_p_(loc_els_p) {
      auto context = -1;
      sl_init_(&context, &nprows_, &npcols_);

      auto row = -1, col = -1;
      blacs_gridinfo_(&context, &nprows_, &npcols_, &row, &col);

      grid_ = { context, row, col };
    }

    BlockMatrix::~BlockMatrix() {
      blacs_gridexit_(&grid_.context);
    }
  }
}
