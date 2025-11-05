#ifndef QTNH_TEN_DEC_BASE_HPP_INCLUDE
#define QTNH_TEN_DEC_BASE_HPP_INCLUDE

#include "ten/type/tensor.hpp"
#include "util/ptuple.hpp"

namespace qtnh {
  enum class DecType { SVD, QRD, LQD, QPD };

  struct DecParams {
    using split_pair = std::pair<qtnh::tidx_tup_st, qtnh::tidx_tup_st>;

    split_pair in_dis_splits;
    split_pair in_loc_splits;

    split_pair cyc_splits;
    split_pair dis_splits;
    split_pair loc_splits;

    // TODO: Derive above from parameters below and given tensor. 
    // qtnh::tidx_tup_st dis_split;
    // qtnh::tidx_tup_st loc_split;
    // std::size_t dis_block_size;
    // std::size_t loc_block_size;
  };

  class Decomposer {
    public:
      Decomposer() = delete;
      Decomposer(qtnh::tptr tp, DecParams params, bool skip_permute = false);
      Decomposer(qtnh::tptr tp, DecParams params, PTupleSrc init_ptup);
      ~Decomposer() = default;

      void decompose(DecType type = DecType::SVD);

      std::tuple<qtnh::tptr, qtnh::tptr, qtnh::tptr> extract_results() {
        return { std::move(tp_u_), std::move(tp_s_), std::move(tp_v_) };
      }

    private:
      qtnh::tptr tp_m_;
      DecParams params_;
      PTupleSrc ptup_;
      bool skip_permute_;

      qtnh::tptr tp_u_;
      qtnh::tptr tp_s_;
      qtnh::tptr tp_v_;
  };
}

#endif