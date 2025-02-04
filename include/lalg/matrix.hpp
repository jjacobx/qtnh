#include <memory>
#include "core/typedefs.hpp"

namespace qtnh {
  namespace lalg {
    using cvec = std::vector<qtnh::tel>;

    struct ProcGrid {
      int context;
      int row;
      int col;
    };

    class BlockMatrix {
      public:
        BlockMatrix() = delete;
        BlockMatrix(int nrows, int ncols, int nprows, int npcols);
        BlockMatrix(int nrows, int ncols, int nprows, int npcols, cvec* loc_els_p);
        ~BlockMatrix();

        constexpr std::pair<int, int> globDims() const { return { nrows_, ncols_ }; }
        constexpr std::pair<int, int> procDims() const { return { nprows_, npcols_ }; }
        
        constexpr std::array<int, 9> const descriptor() {
          return { 
            1,                // DTYPE
            grid_.context,    // CTXT
            nrows_,           // M
            ncols_,           // N
            nrows_ / nprows_, // MB
            ncols_ / npcols_, // NB
            1,                // RSRC
            1,                // CSRC
            nrows_ / nprows_  // LLD
          };
        }

        std::unique_ptr<cvec> extractEls() { return std::move(loc_els_p_); }
      
      private:
        int nrows_;
        int ncols_;

        int nprows_;
        int npcols_;

        // Remember the elements need to be in column-major order. 
        std::unique_ptr<cvec> loc_els_p_;

        ProcGrid grid_;
    };
  }
}
