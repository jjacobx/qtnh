#include <algorithm>
#include <functional>
#include <iostream>

#include "net/mps.hpp"
#include "ten/con/pair-defs.hpp"
#include "ten/dec/base.hpp"
#include "util/ops.hpp"
#include "util/vector.hpp"

namespace qtnh {
  MPS::MPS(const QTNHEnv& env, std::size_t n_sites, qtnh::tidx site_dim, chi_pair chis) 
  : MPS(env, n_sites, qtnh::tidx_tup(n_sites, site_dim), chis)
  {}

  MPS::MPS(const QTNHEnv& env, std::size_t n_sites, qtnh::tidx_tup site_dims, chi_pair chis) 
  : site_tensors_(n_sites)
  , site_norms_(n_sites, MPS_NORM::none)
  , site_dims_(site_dims)
  , dis_chi_(chis.first)
  , loc_chi_(chis.second)
  {
    for (auto i = 0UL; i < n_sites; ++i) {
      std::vector<qtnh::tel> els(site_dims.at(i) * loc_chi_ * loc_chi_);
      if (env.proc_id == 0) els.at(0) = 1.0;

      site_tensors_.at(i) = DenseTensor::make(
        env, 
        { dis_chi_, dis_chi_ }, 
        { site_dims.at(i), loc_chi_, loc_chi_ }, 
        std::move(els)
      );
    }
  }

  MPS::MPS(qtnh::tptr tp, chi_pair chis, MPS_NORM norm) {
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

      ConParams params({{ 1, 0 }, { tot_size - 1, 3 }});
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

    tp_res->print_serial("Tres");
    if (utils::is_root()) std::cout << "AFTER OP\n";

    // Decompose back into site tensors. 
    for (auto i = min_site; i < max_site; ++i) {
      auto loc_size = tp_res->locDims().size();

      DecParams dp {
        { 1, 1 }, 
        { 2, loc_size - 2 }, 
        { 1, loc_size - 3 }, 
        { 1, 1 }, 
        { 1, 1 }
      };

      if (utils::is_root()) std::cout << 
        "dp.in_dis_splits" << dp.in_dis_splits << "\n" << 
        "dp.in_loc_splits" << dp.in_loc_splits << "\n" << 
        "dp.cyc_splits" << dp.cyc_splits << "\n" << 
        "dp.dis_splits" << dp.dis_splits << "\n" << 
        "dp.loc_splits" << dp.loc_splits << "\n";

      Decomposer dec(std::move(tp_res), dp);
      dec.decompose();

      auto [tp_u, tp_s, tp_v] = dec.extract_results();

      utils::barrier();
      if (utils::is_root()) std::cout << "U: " << tp_u->disDims() << ", " << tp_u->locDims() << "\n";
      if (utils::is_root()) std::cout << "S: " << tp_s->disDims() << ", " << tp_s->locDims() << "\n";
      if (utils::is_root()) std::cout << "V: " << tp_v->disDims() << ", " << tp_v->locDims() << "\n";

      // Truncate. 
      auto fmul = std::multiplies<tidx>();
      auto loc_dims_u = utils::combine_part(tp_u->locDims(), 2, 3, tidx(1.0), fmul);
      auto loc_dims_s = utils::combine_part(tp_s->locDims(), 1, 2, tidx(1.0), fmul);
      auto loc_dims_v = utils::combine_part(tp_v->locDims(), 0, 1, tidx(1.0), fmul);

      tp_u->reshape(tp_u->disDims(), loc_dims_u);
      tp_s->reshape(tp_s->disDims(), loc_dims_s);
      tp_v->reshape(tp_v->disDims(), loc_dims_v);

      tp_u = Tensor::truncate(std::move(tp_u), 4, loc_chi_);
      tp_s = Tensor::truncate(std::move(tp_s), 1, loc_chi_);
      tp_v = Tensor::truncate(std::move(tp_v), 2, loc_chi_);

      if (utils::is_root()) std::cout << "TRUNCATED\n";

      if (utils::is_root()) std::cout << "U: " << tp_u->disDims() << ", " << tp_u->locDims() << "\n";
      if (utils::is_root()) std::cout << "S: " << tp_s->disDims() << ", " << tp_s->locDims() << "\n";
      if (utils::is_root()) std::cout << "V: " << tp_v->disDims() << ", " << tp_v->locDims() << "\n";

      tp_u->print_serial("U");
      tp_s->print_serial("S");
      tp_v->print_serial("V");

      // Calculate SV. 
      auto&& els = tp_s->cast<DenseTensor>()->extractEls();
      tp_s = DiagTensor::make(
        tp_s->bc().env(), 
        {}, 
        { dis_chi_, loc_chi_, dis_chi_, loc_chi_ }, 
        false, 
        std::move(els)
      );

      tp_s = SymmTensorBase::rescatterIO(std::move(tp_s), 1);

      tp_s->print_serial("S (resc)");

      using repl_vec = std::vector<tidx_tup_st>;
      repl_vec phys_repls(loc_size - 3);
      std::iota(phys_repls.begin(), phys_repls.end(), 2);

      ConParams params(
        {{ 1, 0 }, { 3, 2 }}, 
        { 0, X, loc_size - 1, X }, 
        utils::concat_vecs(repl_vec { X, 1, X }, phys_repls, repl_vec { loc_size })
      );

      if (utils::is_root()) std::cout << "Wires: " << params.wires << "\n";
      if (utils::is_root()) std::cout << "Repls1: " << params.dimRepls1 << "\n";
      if (utils::is_root()) std::cout << "Repls2: " << params.dimRepls2 << "\n";
      
      pcon con(std::move(tp_s), std::move(tp_v), params);
      tptr tp_sv = con.contract();

      site_tensors_.at(i) = std::move(tp_u);
      tp_res = std::move(tp_sv);
    }

    site_tensors_.at(max_site) = std::move(tp_res);
  }

  qtnh::tel MPS::self_overlap() {
    utils::throw_unimplemented();
    return 0;
  }

  qtnh::tel MPS::overlap(MPS& mps) {
    utils::throw_unimplemented();
    return 0;
  }

  std::unique_ptr<DenseTensor> MPS::toDense() && {
    tptr tp_res = std::move(site_tensors_.at(0));
    for (auto i = 1UL; i < site_tensors_.size(); ++i) {
      tptr tp_tmp = std::move(site_tensors_.at(i));
      auto tot_size = tp_res->totDims().size();

      ConParams params({{ 1, 0 }, { tot_size - 1, 3 }});
      pcon con(std::move(tp_res), std::move(tp_tmp), params);
      tp_res = con.contract();
    }

    auto rank = tp_res->totDims().size();
    PTupleSrc ptup(rank);
    ptup.at(rank - 1) << int(rank - 2);

    tp_res = Tensor::permute(std::move(tp_res), ptup.toTar().tup());
    tp_res = Tensor::rebcast(std::move(tp_res), { 1, 1, tp_res->bc().params().off });

    const auto& bc = tp_res->bc();
    auto [void_dims, new_loc_dims] = utils::split_vec(tp_res->locDims(), 2);
    (void)void_dims; // Unused. 
    auto els = tp_res->cast<DenseTensor>()->extractEls();

    if (bc.gid() == 0) {
      els.resize(utils::dims_to_size(new_loc_dims));
    } else {
      els.clear();
    }

    return DenseTensor::make(bc.env(), {}, new_loc_dims, std::move(els), bc.params());
  }
}
