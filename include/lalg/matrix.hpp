#include <memory>
#include "core/typedefs.hpp"

namespace qtnh {
  namespace lalg {
    using cvec = std::vector<qtnh::tel>;

    class ProcGrid {
      public:
        ProcGrid() = delete;
        ProcGrid(int nprows, int npcols);
        ~ProcGrid();

        constexpr std::pair<int, int> procDims() const { return { nprows_, npcols_ }; }
        constexpr std::pair<int, int> procIdxs() const { return { row_, col_ }; }
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

    class BlockMatrix {
      public:
        BlockMatrix() = delete;
        BlockMatrix(const ProcGrid& grid, int nrows, int ncols);
        BlockMatrix(const ProcGrid& grid, int nrows, int ncols, cvec* loc_els_p);
        ~BlockMatrix() = default;

        const ProcGrid& grid() const { return grid_; }
        qtnh::tel* data() { return loc_els_p_->data(); }
        
        constexpr std::pair<int, int> totDims() const { return { nrows_, ncols_ }; }
        constexpr std::pair<int, int> locDims() const { 
          return { 
            nrows_ / grid_.procDims().first, 
            ncols_ / grid_.procDims().second 
          }; 
        }
        
        constexpr std::array<int, 9> const descriptor() {
          return { 
            1,                          // DTYPE
            grid_.context(),            // CTXT
            nrows_,                     // M
            ncols_,                     // N
            nrows_ / locDims().first,   // MB
            ncols_ / locDims().second,  // NB
            1,                          // RSRC
            1,                          // CSRC
            nrows_ / locDims().first    // LLD
          };
        }

        std::unique_ptr<cvec> extractEls() { return std::move(loc_els_p_); }
      
      private:
        const ProcGrid& grid_;

        int nrows_;
        int ncols_;

        // Remember the elements need to be in column-major order. 
        std::unique_ptr<cvec> loc_els_p_;        
    };
  }
}
