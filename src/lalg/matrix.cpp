#include "lalg/matrix.hpp"
#include "lalg/routines.hpp"

namespace qtnh {
  namespace lalg {
    ProcGrid::ProcGrid(int nprows, int npcols) 
    : ProcGrid(nprows, npcols, 0) 
    {}

    ProcGrid::ProcGrid(int nprows, int npcols, int offset)
    : nprows_(nprows)
    , npcols_(npcols)
    , offset_(offset)
    {
      sl_init_(&context_, &nprows_, &npcols_);
      active_ = (context_ != -1);

      // Grid info resets values for rows/cols. 
      blacs_gridinfo_(&context_, &nprows, &npcols, &row_, &col_);
    }

    ProcGrid::~ProcGrid() {
      if (active_) blacs_gridexit_(&context_);
    }

    int ProcGrid::getPNum(mtup pidxs) const {
      // if (active_) {
      //   return blacs_pnum_(const_cast<int*>(&context_), &pidxs.first, &pidxs.second);
      // } else {
      //   return -1;
      // }

      if (pidxs.first >= 0 && pidxs.first < nprows_ && 
          pidxs.second >= 0 && pidxs.second < npcols_) {
        return pidxs.first * npcols_ + pidxs.second + offset_;
      } else {
        return -1;
      }
    }

    mtup ProcGrid::getPIdxs(int pnum) const {
      // mtup pidxs { -1, -1 };
      // blacs_pcoord_(const_cast<int*>(&context_), &pnum, &pidxs.first, &pidxs.second);
      // return pidxs;

      auto idx_row = (pnum - offset_) / npcols_;
      auto idx_col = (pnum - offset_) % npcols_;

      if (pnum >= offset_ && idx_row < nprows_ && idx_col < npcols_) {
        return { idx_row, idx_col };
      } else {
        return { -1, -1 };
      }
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
