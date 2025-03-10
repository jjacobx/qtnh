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
        ProcGrid(int nprows, int npcols, int offset);
        ~ProcGrid();

        constexpr mtup procDims() const { return { nprows_, npcols_ }; }
        constexpr mtup procIdxs() const { return { row_, col_ }; }
        constexpr bool active() const { return active_; }
        constexpr int context() const { return context_; }

        int getPNum(mtup idxs) const;
        mtup getPIdxs(int pnum) const;

      private:
        int nprows_;
        int npcols_;
        int offset_;

        bool active_ = false;
        int context_ = -1;
        int row_ = -1;
        int col_ = -1;
    };

    class BlockCyclicMatrix {
      public:
        BlockCyclicMatrix() = delete;
        BlockCyclicMatrix(const ProcGrid& grid, int m, int n, int mb, int nb);
        BlockCyclicMatrix(const ProcGrid& grid, int m, int n, int mb, int nb, cvec&& loc_els_p);
        ~BlockCyclicMatrix() = default;

        const ProcGrid& grid() const { return grid_; }
        qtnh::tel* data() { return loc_els_p_.data(); }

        constexpr mtup totDims() const { return { m_, n_ }; }
        constexpr mtup blkDims() const { return { mb_, nb_ }; }
        constexpr mtup disDims() const { return grid_.procDims(); }
        constexpr mtup cycDims() const { 
          return { m_ / disDims().first / mb_, n_ / disDims().second / nb_ };
        }

        constexpr std::array<int, 9> const desc9() {
          return { 
            1,                     // DTYPE
            grid_.context(),       // CTXT
            m_,                    // M
            n_,                    // N
            mb_,                   // MB
            nb_,                   // NB
            0,                     // RSRC
            0,                     // CSRC
            mb_ * cycDims().first  // LLD
          };
        }

        constexpr std::array<int, 11> const desc11() {
          return { 
            601,                   // DTYPE
            grid_.context(),       // CTXT
            m_,                    // M
            n_,                    // N
            mb_,                   // IMB
            nb_,                   // INB
            mb_,                   // MB
            nb_,                   // NB
            0,                     // RSRC
            0,                     // CSRC
            mb_ * cycDims().first  // LLD
          };
        }

        cvec&& extractEls() { return std::move(loc_els_p_); }

      private:
        const ProcGrid& grid_;

        int m_;
        int n_;
        int mb_;
        int nb_;

        // Remember the elements need to be in column-major order. 
        cvec loc_els_p_;
    };
  }

  namespace ops {
    template <typename T, std::size_t N>
    std::ostream& operator<<(std::ostream& out, const std::array<T, N>& o) {
      for (auto i = 0UL; i < N - 1; ++i) {
        out << o.at(i) << ", ";
      }

      out << o.at(N - 1);
      return out;
    }
  }
}

#endif