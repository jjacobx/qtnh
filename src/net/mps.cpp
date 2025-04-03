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
  , site_canons_(n_sites, SITE_CANON::none)
  , site_dims_(site_dims)
  , bond_dims_(n_sites - 1, 1)
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

  MPS::MPS(qtnh::tptr tp, chi_pair chis, SITE_CANON norm) {
    utils::throw_unimplemented();
  }

  std::size_t count_bond_dim(const std::vector<tel>& els, double tol = 1E-10) {
    auto counter = 0UL;
    for (auto& e : els) {
      if (std::abs(e) < tol) break;
      counter++;
    }

    return counter;
  }

  void MPS::apply(std::unique_ptr<SymmTensorBase> tp, 
                  std::vector<std::size_t> sites) {
    using repl_vec = std::vector<tidx_tup_st>;
    auto min_site = *std::min_element(sites.begin(), sites.end());
    auto max_site = *std::max_element(sites.begin(), sites.end());

    // leftCanonicalise(min_site);
    // rightCanonicalise(max_site);

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
      bond_dims_.at(i) = count_bond_dim(els);

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

      site_canons_.at(i) = SITE_CANON::left;
      site_tensors_.at(i) = std::move(tp_u);
      tp_res = std::move(tp_sv);
    }

    site_canons_.at(max_site) = SITE_CANON::none;
    site_tensors_.at(max_site) = std::move(tp_res);
  }

  void MPS::apply(const MPO& mpo, std::size_t from) {
    // ! Check if this is correct. 
    // leftCanonicalise(from);
    // rightCanonicalise(from + mpo.nSites() - 1);

    tptr tp_s1 = std::move(site_tensors_.at(from));
    tptr tp_op1 = mpo.at(0).copy();

    ConParams params1({{2, 0}});
    pcon con1(std::move(tp_s1), std::move(tp_op1), params1);
    tp_s1 = con1.contract();

    for (auto i = 1UL; i < mpo.nSites(); ++i) {
      auto is_last = i == (mpo.nSites() - 1);
      tptr tp_s2 = std::move(site_tensors_.at(from + i));
      tptr tp_op2 = mpo.at(i).copy();

      ConParams params2({{2, 0}});
      pcon con2(std::move(tp_s2), std::move(tp_op2), params2);
      tp_s2 = con2.contract();

      auto repls2 = is_last
        ? std::vector<tidx_tup_st> { X, 1, X, 5, 4, X }
        : std::vector<tidx_tup_st> { X, 1, X, 6, 4, X, 5 };
      ConParams params3({{1, 0}, {3, 2}, {5, 5}}, { 0, X, 3, X, 2, X }, repls2);
      pcon con3(std::move(tp_s1), std::move(tp_s2), params3);
      tptr tp_s3 = con3.contract();

      DecParams dp {
        { 1, 1 }, 
        { 2, is_last ? 2 : 3 }, 
        { 1, is_last ? 1 : 2 }, 
        { 1, 1 }, 
        { 1, 1 }
      };

      Decomposer dec(std::move(tp_s3), dp, true);
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
      bond_dims_.at(from + i - 1) = count_bond_dim(els);

      tp_s = DiagTensor::make(
        tp_s->bc().env(), 
        {}, 
        { dis_chi_, loc_chi_, dis_chi_, loc_chi_ }, 
        false, 
        std::move(els)
      );

      tp_s = SymmTensorBase::rescatterIO(std::move(tp_s), 1);

      auto params4 = is_last
       ? ConParams({{ 1, 0 }, { 3, 2 }}, { 0, X, 3, X }, { X, 1, X, 2, 4})
       : ConParams({{ 1, 0 }, { 3, 2 }}, { 0, X, 2, X }, { X, 1, X, 4, 5, 3});
      
      pcon con4(std::move(tp_s), std::move(tp_v), params4);
      tptr tp_sv = con4.contract();

      site_canons_.at(from + i - 1) = SITE_CANON::left;
      site_tensors_.at(from + i - 1) = std::move(tp_u);
      tp_s1 = std::move(tp_sv);
    }

    site_canons_.at(from + mpo.nSites() - 1) = SITE_CANON::none;
    site_tensors_.at(from + mpo.nSites() - 1) = std::move(tp_s1);
  }

  qtnh::tel MPS::overlap(MPS& mps) {
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
    tptr tp_res = DenseTensor::make(env, { mps.disChi(), disChi() }, 
                                    { mps.locChi(), locChi() }, std::move(els));

    for (auto i = 0UL; i < site_tensors_.size(); ++i) {
      tptr tp_up = mps.site(i).copy();
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

  qtnh::tel MPS::norm() {
    return overlap(*this);
  }

  void MPS::renormalise() {
    auto div = std::pow(norm(), 1.0 / double(nSites()));

    for (auto i = 0UL; i < site_tensors_.size(); ++i) {
      tptr tp = std::move(site_tensors_.at(i));
      for (auto i = 0UL; tp->bc().isActive() && i < tp->locSize(); ++i) {
        (*tp)[i] = (*tp)[i] / div;
      }

      site_tensors_.at(i) = std::move(tp);
    }
  }

  void MPS::leftCanonicalise(std::size_t to) {
    auto can_continue = true;
    for (auto i = 0UL; i + 1 < to; ++i) {
      if (can_continue && site_canons_.at(i) == SITE_CANON::left) {
        continue;
      } else {
        can_continue = false;
      }

      tptr tp1 = std::move(site_tensors_.at(i));
      tptr tp2 = std::move(site_tensors_.at(i + 1));

      ConParams params({{ 1, 0 }, { 4, 3 }});
      pcon con(std::move(tp1), std::move(tp2), params);
      tptr tp12 = con.contract();

      DecParams dp {{ 1, 1 }, { 2, 2 }, { 1, 1 }, { 1, 1 }, { 1, 1 }};
      Decomposer dec(std::move(tp12), dp, true);
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

      // TODO: Check for non-zero truncation. 
      tp_u = Tensor::truncate(std::move(tp_u), 4, loc_chi_);
      tp_s = Tensor::truncate(std::move(tp_s), 1, loc_chi_);
      tp_v = Tensor::truncate(std::move(tp_v), 2, loc_chi_);

      // Calculate SV. 
      auto&& els = tp_s->cast<DenseTensor>()->extractEls();
      bond_dims_.at(i) = count_bond_dim(els);

      tp_s = DiagTensor::make(
        tp_s->bc().env(), 
        {}, 
        { dis_chi_, loc_chi_, dis_chi_, loc_chi_ }, 
        false, 
        std::move(els)
      );

      tp_s = SymmTensorBase::rescatterIO(std::move(tp_s), 1);

      ConParams params_sv({{ 1, 0 }, { 3, 2 }}, { 0, X, 3, X }, { X, 1, X, 2, 4 });
      pcon con_sv(std::move(tp_s), std::move(tp_v), params_sv);

      site_canons_.at(i) = SITE_CANON::left;
      site_canons_.at(i + 1) = SITE_CANON::none;
      site_tensors_.at(i) = std::move(tp_u);
      site_tensors_.at(i + 1) = con_sv.contract();
    }
  }

  void MPS::rightCanonicalise(std::size_t to) {
    auto n = nSites();
    auto can_continue = true;
    for (auto i = 1UL; i < n - to; ++i) {
      if (can_continue && site_canons_.at(n - i) == SITE_CANON::right) {
        continue;
      } else {
        can_continue = false;
      }

      tptr tp1 = std::move(site_tensors_.at(n - i - 1));
      tptr tp2 = std::move(site_tensors_.at(n - i));

      ConParams params({{ 1, 0 }, { 4, 3 }});
      pcon con(std::move(tp1), std::move(tp2), params);
      tptr tp12 = con.contract();

      DecParams dp {{ 1, 1 }, { 2, 2 }, { 1, 1 }, { 1, 1 }, { 1, 1 }};
      Decomposer dec(std::move(tp12), dp, true);
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

      // TODO: Check for non-zero truncation. 
      tp_u = Tensor::truncate(std::move(tp_u), 4, loc_chi_);
      tp_s = Tensor::truncate(std::move(tp_s), 1, loc_chi_);
      tp_v = Tensor::truncate(std::move(tp_v), 2, loc_chi_);

      // Calculate US. 
      auto&& els = tp_s->cast<DenseTensor>()->extractEls();
      bond_dims_.at(n - i - 1) = count_bond_dim(els);

      tp_s = DiagTensor::make(
        tp_s->bc().env(), 
        {}, 
        { dis_chi_, loc_chi_, dis_chi_, loc_chi_ }, 
        false, 
        std::move(els)
      );

      tp_s = SymmTensorBase::rescatterIO(std::move(tp_s), 1);

      pcon con_us(std::move(tp_u), std::move(tp_s), ConParams({{ 1, 0 }, { 4, 2 }}));
      PTupleSrc ptup(tp_v->totDims().size());
      ptup.at(3) << 1;

      site_canons_.at(n - i) = SITE_CANON::right;
      site_canons_.at(n - i - 1) = SITE_CANON::none;
      site_tensors_.at(n - i) = Tensor::permute(std::move(tp_v), ptup.toTar().tup());
      site_tensors_.at(n - i - 1) = con_us.contract();
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

  std::ostream& operator<<(std::ostream& out, const SITE_CANON& o) {
    switch(o) {
      case SITE_CANON::left : out << "L"; break;
      case SITE_CANON::right: out << "R"; break;
      default               : out << "N";
    }

    return out;
  }

  void MPS::print() const {
    utils::barrier();

    if (utils::is_root()) {
      std::cout << "================================================================\n";
      std::cout << "MPS with N=" << site_tensors_.size() << 
        " chi=(" << dis_chi_ << "," << loc_chi_ << ")\n";
      std::cout << "Phys:  " << site_dims_ << "\n";
      std::cout << "Canons: " << site_canons_ << "\n";

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

  MPO::MPO(std::vector<qtnh::tptr>&& site_ops) 
  : site_ops_(std::move(site_ops))
  {}

  void MPO::print() const {
    utils::barrier();

    if (utils::is_root()) {
      std::cout << "================================================================\n";
      std::cout << "MPO with N=" << site_ops_.size() << ")\n";

      std::cout << "----------------------------------------------------------------\n";
      std::cout << "Sites: \n";
    }

    for (auto i = 0UL; i < site_ops_.size(); ++i) {
      std::string label = "O" + std::to_string(i);
      site_ops_.at(i)->print_serial(label, true, false);
    }

    if (utils::is_root()) {
      std::cout << "================================================================\n";
    }

    utils::barrier();
  }
}
