#ifndef __CON_DECOMP__
#define __CON_DECOMP__

#include "lalg/matrix.hpp"
#include "tensor/tensor.hpp"
#include "tensor/dense.hpp"
#include "tensor/ptuple.hpp"

namespace qtnh {
  struct DecParams {
    using split_pair = std::pair<qtnh::tidx_tup_st, qtnh::tidx_tup_st>;

    split_pair cyc_splits;
    split_pair dis_splits;
    split_pair loc_splits;
  };

  class Decomposer {
    public:
      Decomposer() = delete;
      Decomposer(qtnh::tptr tp, DecParams params);
      Decomposer(qtnh::tptr tp, DecParams params, PTupleSrc init_ptup);
      ~Decomposer() = default;

      void decompose();

      std::tuple<qtnh::tptr, qtnh::tptr, qtnh::tptr> extract_results() {
        return { std::move(tp_u_), std::move(tp_s_), std::move(tp_v_) };
      }

    private:
      qtnh::tptr tp_m_;
      DecParams params_;
      PTupleSrc ptup_;

      qtnh::tptr tp_u_;
      qtnh::tptr tp_s_;
      qtnh::tptr tp_v_;
  };
}

#endif