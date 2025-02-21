#include "lalg/matrix.hpp"
#include "lalg/routines.hpp"

namespace qtnh {
  namespace lalg {
    ProcGrid::ProcGrid(int nprows, int npcols)
    : nprows_(nprows)
    , npcols_(npcols)
    {
      sl_init_(&context_, &nprows_, &npcols_);
      active_ = (context_ != -1);

      blacs_gridinfo_(&context_, &nprows_, &npcols_, &row_, &col_);
    }

    ProcGrid::~ProcGrid() {
      if (active_) blacs_gridexit_(&context_);
    }

    BlockCyclicMatrix::BlockCyclicMatrix(const ProcGrid& grid, int m, int n, int mb, int nb)
    : BlockCyclicMatrix(grid, m, n, mb, nb, {})
    {
      auto proc_dims = grid.procDims();
      auto loc_size = (m * n) / (proc_dims.first * proc_dims.second);
      loc_els_p_ = cvec(loc_size);
    }

    BlockCyclicMatrix::BlockCyclicMatrix(const ProcGrid& grid, int m, int n, int mb, int nb,
                                         cvec&& loc_els_p)
    : grid_(grid)
    , m_(m)
    , n_(n)
    , mb_(mb)
    , nb_(nb)
    , loc_els_p_(loc_els_p)
    {}
  }
}
