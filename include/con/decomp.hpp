#ifndef __CON_DECOMP__
#define __CON_DECOMP__

#include "lalg/matrix.hpp"
#include "tensor/tensor.hpp"
#include "tensor/dense.hpp"
#include "tensor/ptuple.hpp"

namespace qtnh {
  struct DecParams {
    lalg::mtup cyc_dims;
    lalg::mtup dis_dims;
    lalg::mtup loc_dims;
  };

  class Decomposer {
    public:
      Decomposer() = delete;
      Decomposer(qtnh::tptr tp, DecParams params);
      Decomposer(qtnh::tptr tp, DecParams params, PTupleSrc init_ptup);
      ~Decomposer() = default;

      void decompose();

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