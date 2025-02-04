#include "lalg/matrix.hpp"
#include "lalg/routines.hpp"

namespace qtnh {
  namespace lalg {
    ProcGrid::ProcGrid(int nprows, int npcols) 
      : nprows_(nprows), npcols_(npcols) {
      sl_init_(&context_, &nprows_, &npcols_);
      active_ = (context_ != -1);

      blacs_gridinfo_(&context_, &nprows_, &npcols_, &row_, &col_);
    }

    ProcGrid::~ProcGrid() {
      if (active_) blacs_gridexit_(&context_);
    }

    BlockMatrix::BlockMatrix(const ProcGrid& grid, int nrows, int ncols)
      : BlockMatrix(grid, nrows, ncols, nullptr) {
      auto proc_dims = grid.procDims();
      auto loc_size = (nrows * ncols) / (proc_dims.first * proc_dims.second);
      loc_els_p_ = std::make_unique<cvec>(loc_size);
    }
    
    BlockMatrix::BlockMatrix(const ProcGrid& grid, int nrows, int ncols, cvec* loc_els_p)
      : grid_(grid), nrows_(nrows), ncols_(ncols), loc_els_p_(loc_els_p) {}
  }
}
