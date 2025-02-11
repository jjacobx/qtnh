#include "con/decomp.hpp"
#include "lalg/routines.hpp"

namespace qtnh {
  Decomposer::Decomposer(qtnh::tptr tp, DecParams params) 
  : tp_m_(std::move(tp))
  , params_(params)
  , ptup_(tp_m_->totDims().size()) 
  {
    auto [tup_rc, tup_r1] = utils::split_vec(ptup_.tup(), params_.cyc_splits.first);
    auto [tup_rd, tup_r2] = utils::split_vec(tup_r1, params_.dis_splits.first);
    auto [tup_rb, tup_r3] = utils::split_vec(tup_r2, params_.loc_splits.first);
    auto [tup_cc, tup_r4] = utils::split_vec(tup_r3, params_.cyc_splits.second);
    auto [tup_cd, tup_cb] = utils::split_vec(tup_r4, params_.dis_splits.second);
    
    std::vector<std::string> labels { "rc", "rd", "rb", "cc", "cd", "cb" };
    std::vector<qtnh::tup_t> groups { tup_rc, tup_rd, tup_rb, tup_cc, tup_cd, tup_cb };
    IndexGroup ig(labels, groups);

    ig.reorder({ "rd", "cd", "cc", "cb", "rc", "rb" });
    ptup_ = ig.ptup();
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

    tp_m_ = Tensor::permute(std::move(tp_m_), ptup_.toTar().tup());
    auto dtp = Tensor::convert<DenseTensor>(std::move(tp_m_));

    auto dis_dims = dtp->disDims();
    auto [dims_rd, dims_cd] = utils::split_dims(dis_dims, params_.dis_splits.first);
    auto size_rd = utils::dims_to_size(dims_rd);
    auto size_cd = utils::dims_to_size(dims_cd);
    
    using namespace lalg;
    ProcGrid pg (static_cast<int>(size_rd), static_cast<int>(size_cd));

    // Fortran layout means rows and columns are inverted. 
    auto loc_split = params_.cyc_splits.second + params_.loc_splits.second;
    auto [dims_cl, dims_rl] = utils::split_dims(dtp->locDims(), loc_split);
    auto nrows = size_rd * utils::dims_to_size(dims_rl);
    auto ncols = size_cd * utils::dims_to_size(dims_cl);

    auto [dims_rc, dims_rb] = utils::split_dims(dims_rl, params_.cyc_splits.first);
    auto [dims_cc, dims_cb] = utils::split_dims(dims_cl, params_.cyc_splits.second);
    auto nblock = utils::dims_to_size(dims_rb);

    BlockCyclicMatrix m(pg, int(nrows), int(ncols), int(nblock), dtp->extractEls());

    auto [u, s, v] = PZGESVD(std::move(m));

    auto dims_xc = (ncols > nrows) ? dims_rc : dims_cc;
    auto dims_xl = utils::concat_dims(dims_xc, dims_cb);

    auto loc_dims_u = utils::concat_dims(dims_xl, dims_rl);
    auto loc_dims_v = utils::concat_dims(dims_cl, dims_xl);
    auto loc_dims_s = utils::concat_dims(dims_rd, dims_xl);

    tp_u_ = DenseTensor::make(env, dis_dims, loc_dims_u, u.extractEls());
    tp_s_ = DenseTensor::make(env, {}, loc_dims_s, std::move(s));
    tp_v_ = DenseTensor::make(env, dis_dims, loc_dims_v, v.extractEls());
  }
}
