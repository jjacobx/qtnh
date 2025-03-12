#include <mpi.h>

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

    BlockCyclicMatrix::BlockCyclicMatrix(const ProcGrid& pg, mtup tot_dims, mtup blk_dims)
    : BlockCyclicMatrix(pg, tot_dims, blk_dims, cvec {})
    {
      auto loc_size = std::size_t(mc_ * nc_ * mb_ * nb_);
      loc_els_ = cvec(loc_size);
    }

    BlockCyclicMatrix::BlockCyclicMatrix(const ProcGrid& pg, mtup tot_dims, mtup blk_dims, 
                                         cvec&& loc_els)
    : pg_(pg)
    , mb_(blk_dims.first)
    , nb_(blk_dims.second)
    , md_(pg.procDims().first)
    , nd_(pg.procDims().second)
    , mc_(tot_dims.first / md_ / mb_)
    , nc_(tot_dims.second / nd_ / nb_)
    , m_(tot_dims.first)
    , n_(tot_dims.second)
    , loc_els_(loc_els)
    {}

    BlockCyclicMatrix::BlockCyclicMatrix(const ProcGrid& pg, mtup blk_dims, mtup dis_dims, mtup cyc_dims)
    : BlockCyclicMatrix(pg, blk_dims, dis_dims, cyc_dims, cvec {})
    {
      auto loc_size = std::size_t(mc_ * nc_ * mb_ * nb_);
      loc_els_ = cvec(loc_size);
    }

    BlockCyclicMatrix::BlockCyclicMatrix(const ProcGrid& pg, mtup blk_dims, mtup dis_dims, mtup cyc_dims, 
                                         cvec&& loc_els)
    : pg_(pg)
    , mb_(blk_dims.first)
    , nb_(blk_dims.second)
    , md_(dis_dims.first)
    , nd_(dis_dims.second)
    , mc_(cyc_dims.first)
    , nc_(cyc_dims.second)
    , m_(mc_ * md_ * mb_)
    , n_(nc_ * nd_ * nb_)
    , loc_els_(loc_els)
    {}
  }
}