#include "net/mps.hpp"

namespace qtnh {
  MPS::MPS(const QTNHEnv& env, std::size_t n_sites, qtnh::tidx site_dim) 
  : MPS(env, n_sites, qtnh::tidx_tup(n_sites, site_dim))
  {}

  MPS::MPS(const QTNHEnv& env, std::size_t n_sites, qtnh::tidx_tup site_dims) 
  : site_tensors_(n_sites)
  , site_norms_(n_sites, MPS_NORM::none)
  , site_dims_(site_dims)
  , dis_chi_(1)
  , loc_chi_(1)
  {
    for (auto i = 0UL; i < n_sites; ++i) {
      std::vector<qtnh::tel> els(site_dims.at(i));
      els.at(0) = 1.0;

      auto tp = DenseTensor::make(env, { 1, 1 }, { site_dims.at(i), 1, 1 }, std::move(els));
      site_tensors_.at(i) = std::move(tp);
    }
  }

  MPS::MPS(qtnh::tptr tp, MPS_NORM norm) {
    utils::throw_unimplemented();
  }

  void MPS::apply(std::unique_ptr<SymmTensorBase> tp, 
                  std::vector<std::size_t> sites) {

  }

  qtnh::tel MPS::self_overlap() {
    utils::throw_unimplemented();
    return 0;
  }

  qtnh::tel MPS::overlap(MPS& mps) {
    utils::throw_unimplemented();
    return 0;
  }
}
