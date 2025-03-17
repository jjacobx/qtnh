#include <algorithm>

#include "net/mps.hpp"
#include "ten/con/pair-defs.hpp"
#include "ten/dec/base.hpp"

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
    auto min_site = *std::min_element(sites.begin(), sites.end());
    auto max_site = *std::max_element(sites.begin(), sites.end());

    // Contract all sites within range (min, max). 
    tptr tp_res = std::move(site_tensors_.at(min_site));
    for (auto i = min_site + 1; i <= max_site; ++i) {
      tptr tp_tmp = std::move(site_tensors_.at(i));
      auto tot_size = tp_res->totDims().size();

      ConParams params({{ 1, tot_size - 1 }, { 0, 3 }});
      pcon con(std::move(tp_res), std::move(tp_tmp), params);
      tp_res = con.contract();
    }

    // Contract sites with applied tensor. 
    std::vector<wire> wires(sites.size());
    for (auto i = 0UL; i < sites.size(); ++i) {
      wires.at(i) = { sites.at(i) + 2, i };
    }
    
    pcon con(std::move(tp_res), std::move(tp), ConParams(wires));
    tp_res = con.contract();

    // Decompose back into site tensors. 
    for (auto i = min_site; i < max_site; ++i) {
      auto loc_size = tp_res->locDims().size();

      DecParams dp {
        { 1, 1 }, 
        { 2, loc_size - 3 }, 
        { 1, loc_size - 4 }, 
        { 1, 1 }, 
        { 1, 1 }
      };

      Decomposer dec(std::move(tp_res), dp);
      dec.decompose();

      auto [tp_u, tp_s, tp_v] = dec.extract_results();

      // Calculate SV and truncate. 
      // ...
      // * Temporary. 
      site_tensors_.at(i) = std::move(tp_u);
      tp_res = std::move(tp_v);
    }
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
