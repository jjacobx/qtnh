#ifndef __LALG_MATRIX__
#define __LALG_MATRIX__

#include <memory>
#include "core/typedefs.hpp"

namespace qtnh {
  namespace lalg {
    using cvec = std::vector<qtnh::tel>;
    using mtup = std::pair<int, int>;

    class ProcGrid {
      public:
        ProcGrid() = delete;
        ProcGrid(int nprows, int npcols);
        ~ProcGrid();

        constexpr mtup procDims() const { return { nprows_, npcols_ }; }
        constexpr mtup procIdxs() const { return { row_, col_ }; }
        constexpr bool active() const { return active_; }
        constexpr bool context() const { return context_; }

      private:
        int nprows_;
        int npcols_;

        bool active_ = false;
        int context_ = -1;
        int row_ = -1;
        int col_ = -1;
    };

    class BlockCyclicMatrix {
      public:
        BlockCyclicMatrix() = delete;
        BlockCyclicMatrix(const ProcGrid& grid, int nrows, int ncols, int nblock);
        BlockCyclicMatrix(const ProcGrid& grid, int nrows, int ncols, int nblock, cvec&& loc_els_p);
        ~BlockCyclicMatrix() = default;

        const ProcGrid& grid() const { return grid_; }
        qtnh::tel* data() { return loc_els_p_.data(); }

        constexpr mtup totDims() const { return { nrows_, ncols_ }; }
        constexpr mtup cycDims() const { 
          return { 
            nrows_ / grid_.procDims().first / nblock_, 
            ncols_ / grid_.procDims().second / nblock_
          };
        }
        constexpr int nBlock() const { return nblock_; }

        constexpr std::array<int, 9> const descSVD() {
          return { 
            1,                          // DTYPE
            grid_.context(),            // CTXT
            nrows_,                     // M
            ncols_,                     // N
            nblock_,                    // MB
            nblock_,                    // NB
            0,                          // RSRC
            0,                          // CSRC
            nblock_ * cycDims().first   // LLD
          };
        }

        constexpr std::array<int, 11> const descMM() {
          return { 
            1,                          // DTYPE
            grid_.context(),            // CTXT
            nrows_,                     // M
            ncols_,                     // N
            nblock_,                    // IMB
            nblock_,                    // INB
            nblock_,                    // MB
            nblock_,                    // NB
            0,                          // RSRC
            0,                          // CSRC
            nblock_ * cycDims().first   // LLD
          };
        }

        cvec&& extractEls() { return std::move(loc_els_p_); }

      private:
        const ProcGrid& grid_;

        int nrows_;
        int ncols_;
        int nblock_;

        // Remember the elements need to be in column-major order. 
        cvec loc_els_p_;
    };
  }
}

#endif