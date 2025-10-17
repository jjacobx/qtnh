#include "blas/wrappers.hpp"
#include "ten/dec/base.hpp"
#include "util/vector.hpp"

#ifdef DEBUG
#include <iostream>
#endif

#include "util/ops.hpp"

namespace qtnh {
  std::vector<qtnh::tup_t> _split_tensor(Tensor* t, std::vector<qtnh::tidx_tup_st> rel_splits) {
    std::vector<qtnh::tup_t> tups(rel_splits.size());
    auto next_tup = PTupleSrc(t->totDims().size()).tup();

    for (auto i = 0UL; i < rel_splits.size(); ++i) {
      auto [tup1, tup2] = utils::split_vec(next_tup, rel_splits.at(i));
      tups.at(i) = tup1;
      next_tup = tup2;
    }

    return tups;
  };

  Decomposer::Decomposer(qtnh::tptr tp, DecParams params, bool skip_permute) 
  : tp_m_(std::move(tp))
  , params_(params)
  , ptup_(tp_m_->totDims().size())
  , skip_permute_(skip_permute)
  {
    std::vector<qtnh::tidx_tup_st> rel_splits1 {
      params_.in_dis_splits.first, params_.in_dis_splits.second, 
      params_.in_loc_splits.first, params_.in_loc_splits.second
    };

    auto tups1 = _split_tensor(tp_m_.get(), rel_splits1);
    IndexGroup ig1({ "d1", "d2", "l1", "l2" }, tups1);
    ig1.reorder({ "d1", "l1", "d2", "l2" });

    std::vector<qtnh::tidx_tup_st> rel_splits2 {
      params_.cyc_splits.first, params_.dis_splits.first, params_.loc_splits.first, 
      params_.cyc_splits.second, params_.dis_splits.second, params_.loc_splits.second
    };
    
    auto tups2 = _split_tensor(tp_m_.get(), rel_splits2);
    IndexGroup ig2({ "rc", "rd", "rb", "cc", "cd", "cb" }, tups2);
    ig2.reorder({ "rd", "cd", "cc", "cb", "rc", "rb" });

    ptup_ = ig2.ptup() * ig1.ptup();
  }
  
  Decomposer::Decomposer(qtnh::tptr tp, DecParams params, PTupleSrc init_ptup)
  : Decomposer(std::move(tp), params)
  {
    ptup_ = ptup_ * init_ptup.inv();
  }

  // Decomposition only works when distributed and block dimensions are the same
  // for rows and columns. Cycle dimensions can vary. 
  void Decomposer::decompose() {
    auto& env = tp_m_->bc().env();
    auto offset = tp_m_->bc().params().off;

    if (!skip_permute_) {
      tp_m_ = Tensor::permute(std::move(tp_m_), ptup_.toTar().tup());
    } else {
      // ! Fortran permute – better interface needed. 
      std::vector<qtnh::tidx_tup_st> rel_splits {
        params_.in_dis_splits.first + params_.in_dis_splits.second, 
        params_.in_loc_splits.first, params_.in_loc_splits.second
      };

      auto tups = _split_tensor(tp_m_.get(), rel_splits);
      IndexGroup ig({ "d", "lr", "lc" }, tups);
      ig.reorder({ "d", "lc", "lr" });
      tp_m_ = Tensor::permute(std::move(tp_m_), ig.ptup().toTar().tup());
    }

    tp_m_ = Tensor::rebcast(std::move(tp_m_), { 1, 1, offset });
    auto dtp = Tensor::convert<DenseTensor>(std::move(tp_m_));

    auto dis_dims = dtp->disDims();
    auto [dims_rd, dims_cd] = utils::split_dims(dis_dims, params_.dis_splits.first);
    auto size_rd = utils::dims_to_size(dims_rd);
    auto size_cd = utils::dims_to_size(dims_cd);
    
    using namespace lalg;
    ProcGrid pg(static_cast<int>(size_rd), static_cast<int>(size_cd), offset);

    // Fortran layout means rows and columns are inverted. 
    auto loc_split = params_.cyc_splits.second + params_.loc_splits.second;
    auto [dims_cl, dims_rl] = utils::split_dims(dtp->locDims(), loc_split);
    auto nrows = size_rd * utils::dims_to_size(dims_rl);
    auto ncols = size_cd * utils::dims_to_size(dims_cl);

    auto [dims_rc, dims_rb] = utils::split_dims(dims_rl, params_.cyc_splits.first);
    auto [dims_cc, dims_cb] = utils::split_dims(dims_cl, params_.cyc_splits.second);
    auto block = utils::dims_to_size(dims_rb);

    BlockCyclicMatrix m(pg, { int(nrows), int(ncols) }, { int(block), int(block) }, dtp->extractEls());

    auto [u, s, v] = PZGESVD(std::move(m));

    auto dims_xc = (ncols > nrows) ? dims_rc : dims_cc;
    auto dims_xd = (ncols > nrows) ? dims_rd : dims_cd;

    auto loc_dims_u = utils::concat_vecs(dims_xc, dims_cb, dims_rl);
    auto loc_dims_v = utils::concat_vecs(dims_cl, dims_xc, dims_cb);
    auto loc_dims_s = utils::concat_vecs(dims_xc, dims_xd, dims_cb);

    tp_u_ = DenseTensor::make(env, dis_dims, loc_dims_u, u.extractEls(), { 1, 1, offset });
    tp_s_ = DenseTensor::make(env, {}, loc_dims_s, std::move(s), { 1, 1, offset });
    tp_v_ = DenseTensor::make(env, dis_dims, loc_dims_v, v.extractEls(), { 1, 1, offset });

    // Permute S to correspond to U and V. 
    std::vector<tidx_tup_st> rel_splits_s = { dims_xc.size(), dims_xd.size(), dims_cb.size() };
    auto tups_s = _split_tensor(tp_s_.get(), rel_splits_s);
    IndexGroup ig_s({ "c", "d", "l" }, tups_s);
    ig_s.reorder({ "d", "c", "l" });
    tp_s_ = Tensor::permute(std::move(tp_s_), ig_s.ptup().toTar().tup());
    
    // Everything below is book-keeping to restore right index order. 
    // TODO: Wrap repeated parts into a function. 
    auto params_u = params_;
    auto params_v = params_;
    if (ncols > nrows) {
      params_u.cyc_splits.second = params_u.cyc_splits.first;
      params_u.in_dis_splits.second = params_u.in_dis_splits.first;
      params_u.in_loc_splits.second = params_u.in_loc_splits.first;
    } else {
      params_v.cyc_splits.first = params_v.cyc_splits.second;
      params_v.in_dis_splits.first = params_v.in_dis_splits.second;
      params_v.in_loc_splits.first = params_v.in_loc_splits.second;
    }

    std::vector<qtnh::tidx_tup_st> rel_splits_u1 {
      params_u.in_dis_splits.first, params_u.in_dis_splits.second, 
      params_u.in_loc_splits.first, params_u.in_loc_splits.second
    };
    std::vector<qtnh::tidx_tup_st> rel_splits_u2 {
      params_u.cyc_splits.first, params_u.dis_splits.first, params_u.loc_splits.first, 
      params_u.cyc_splits.second, params_u.dis_splits.second, params_u.loc_splits.second
    };
    std::vector <qtnh::tidx_tup_st> rel_splits_v1 {
      params_v.in_dis_splits.first, params_v.in_dis_splits.second, 
      params_v.in_loc_splits.first, params_v.in_loc_splits.second
    };
    std::vector<qtnh::tidx_tup_st> rel_splits_v2 {
      params_v.cyc_splits.first, params_v.dis_splits.first, params_v.loc_splits.first, 
      params_v.cyc_splits.second, params_v.dis_splits.second, params_v.loc_splits.second
    };

    auto tups_u1 = _split_tensor(tp_u_.get(), rel_splits_u1);
    auto tups_u2 = _split_tensor(tp_u_.get(), rel_splits_u2);
    auto tups_v1 = _split_tensor(tp_v_.get(), rel_splits_v1);
    auto tups_v2 = _split_tensor(tp_v_.get(), rel_splits_v2);

    IndexGroup ig_u1({ "d1", "d2", "l1", "l2" }, tups_u1);
    ig_u1.reorder({ "d1", "l1", "d2", "l2" });
    IndexGroup ig_u2({ "rc", "rd", "rb", "cc", "cd", "cb" }, tups_u2);
    ig_u2.reorder({ "rd", "cd", "cc", "cb", "rc", "rb" });

    IndexGroup ig_v1({ "d1", "d2", "l1", "l2" }, tups_v1);
    ig_v1.reorder({ "d1", "l1", "d2", "l2" });
    IndexGroup ig_v2({ "rc", "rd", "rb", "cc", "cd", "cb" }, tups_v2);
    ig_v2.reorder({ "rd", "cd", "cc", "cb", "rc", "rb" });

    auto ptup_u = (ig_u2.ptup() * ig_u1.ptup()).inv();
    auto ptup_v = (ig_v2.ptup() * ig_v1.ptup()).inv();

    if (!skip_permute_) {
      tp_u_ = Tensor::permute(std::move(tp_u_), ptup_u.toTar().tup());
      tp_v_ = Tensor::permute(std::move(tp_v_), ptup_v.toTar().tup());
    } else {
      std::vector<qtnh::tidx_tup_st> rel_splits_u {
        params_u.in_dis_splits.first + params_u.in_dis_splits.second, 
        params_u.in_loc_splits.first, params_u.in_loc_splits.second
      };
      std::vector <qtnh::tidx_tup_st> rel_splits_v {
        params_v.in_dis_splits.first + params_v.in_dis_splits.second, 
        params_v.in_loc_splits.first, params_v.in_loc_splits.second
      };

      auto tups_u = _split_tensor(tp_u_.get(), rel_splits_u);
      auto tups_v = _split_tensor(tp_v_.get(), rel_splits_v);

      IndexGroup ig_u({ "d", "lr", "lc" }, tups_u);
      IndexGroup ig_v({ "d", "lr", "lc" }, tups_v);
      ig_u.reorder({ "d", "lc", "lr" });
      ig_v.reorder({ "d", "lc", "lr" });

      tp_u_ = Tensor::permute(std::move(tp_u_), ig_u.ptup().inv().toTar().tup());
      tp_v_ = Tensor::permute(std::move(tp_v_), ig_v.ptup().inv().toTar().tup());
    }
  }
}
