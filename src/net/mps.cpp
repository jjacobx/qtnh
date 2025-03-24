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
    using repl_vec = std::vector<tidx_tup_st>;
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
      auto rel_site = sites.at(i) - min_site;
      auto incr = rel_site == 0UL ? 0UL : 1UL;
      wires.at(i) = { 2 + rel_site + incr, i };
    }

    // Symmetric tensor contraction should replace wires in-place. 
    pcon con(std::move(tp_res), std::move(tp), ConParams(wires));
    tp_res = con.contract();

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

      Decomposer dec(std::move(tp_res), dp, true);
      dec.decompose();

      auto [tp_u, tp_s, tp_v] = dec.extract_results();

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

      auto dim_repls_2 = utils::concat_vecs(
        repl_vec { X, 1, X, 2 }, 
        utils::vec_incr(4UL, loc_size)
      );

      ConParams params(
        {{ 1, 0 }, { 3, 2 }}, 
        { 0, X, 3, X }, 
        dim_repls_2
      );
      
      pcon con(std::move(tp_s), std::move(tp_v), params);
      tptr tp_sv = con.contract();

      site_tensors_.at(i) = std::move(tp_u);
      tp_res = std::move(tp_sv);
    }

    site_tensors_.at(max_site) = std::move(tp_res);
  }

  qtnh::tel MPS::self_overlap() {
    //
    // XXX - XXX - XXX - XXX - XXX
    // XXX = XXX = XXX = XXX = XXX
    //  |     |     |     |     |
    // XXX - XXX - XXX - XXX - XXX
    // XXX = XXX = XXX = XXX = XXX 
    //

    // TODO: Start at lowest rank in MPS. 
    const auto& env = site_tensors_.at(0)->bc().env();
    std::vector<tel> els(loc_chi_ * loc_chi_ , 0);
    if (utils::is_root()) els.at(0) = 1.0;
    tptr tp_res = DenseTensor::make(env, { dis_chi_, dis_chi_ }, { loc_chi_, loc_chi_ }, std::move(els));

    for (auto i = 0UL; i < site_tensors_.size(); ++i) {
      tptr tp_up = site_tensors_.at(i)->copy();
      tptr tp_dn = site_tensors_.at(i)->copy();

      // Conjugate UP tensor. 
      for (auto i = 0UL; tp_up->bc().isActive() && i < tp_up->locSize(); ++i) {
        (*tp_up)[i] = std::conj((*tp_up)[i]);
      }

      ConParams params1({{ 0, 0 }, { 2, 3 }});
      pcon con1(std::move(tp_res), std::move(tp_up), params1);
      tp_res = con1.contract();

      ConParams params2({{ 0, 0 }, { 2, 3 }, { 3, 2 }});
      pcon con2(std::move(tp_res), std::move(tp_dn), params2);
      tp_res = con2.contract();
    }

    qtnh::tel res;
    if (utils::is_root()) {
      res = tp_res->at({ 0, 0, 0, 0 });
    }

    MPI_Bcast(&res, 1, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

    return res;
  }

  qtnh::tel MPS::overlap(MPS& mps) {
    utils::throw_unimplemented();
    return 0;
  }

  void MPS::renormalise() {
    auto div = self_overlap();
    div = std::pow(div, 1.0 / double(nSites()));

    for (auto i = 0UL; i < site_tensors_.size(); ++i) {
      tptr tp = std::move(site_tensors_.at(i));
      for (auto i = 0UL; tp->bc().isActive() && i < tp->locSize(); ++i) {
        (*tp)[i] = (*tp)[i] / div;
      }

      site_tensors_.at(i) = std::move(tp);
    }
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
    ptup.at(3) << 1;
    ptup.at(rank - 1) << int(rank - 4);

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

  void MPS::print() const {
    utils::barrier();

    if (utils::is_root()) {
      std::cout << "================================================================\n";
      std::cout << "MPS with N=" << site_tensors_.size() << 
        " chi=(" << dis_chi_ << "," << loc_chi_ << ")\n";
      std::cout << "Phys: " << site_dims_ << "\n";

      std::cout << "----------------------------------------------------------------\n";
      std::cout << "Sites: \n";
    }

    for (auto i = 0UL; i < site_tensors_.size(); ++i) {
      std::string label = "S" + std::to_string(i);
      site_tensors_.at(i)->print_serial(label, true, false);
    }

    if (utils::is_root()) {
      std::cout << "================================================================\n";
    }

    utils::barrier();
  }
}
