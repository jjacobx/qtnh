#include <algorithm>
#include <functional>
#include <iostream>
#include <random>

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

  std::size_t count_bond_dim(const std::vector<tel>& els, double tol = ZERO_TOL) {
    auto counter = 0UL;
    for (auto& e : els) {
      if (std::abs(e) < tol) break;
      counter++;
    }

    // Debug print. 
    if (utils::is_root()) std::cout << els << "\n";

    return counter;
  }

  void MPS::apply(tptr_symm tp, std::vector<std::size_t> sites) {
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
    tptr tp_site = std::move(site_tensors_.at(from));
    tptr tp_op = mpo.at(0).copy();

    // ! Doesn't work for 1-site MPO. 
    ConParams params({{ 2, 0 }}, { 0, 1, X, 3, 5 }, { X, 2, 4 });
    pcon con(std::move(tp_site), std::move(tp_op), params);
    tp_site = con.contract();

    for (auto i = 0UL; i + 1 < mpo.nSites(); ++i) {
      DecParams dp {
        { 1, 1 }, 
        { 2, 2 }, 
        { 1, 1 }, 
        { 1, 1 }, 
        { 1, 1 }
      };

      Decomposer dec(std::move(tp_site), dp, true);
      dec.decompose();

      auto [tp_u, tp_s, tp_v] = dec.extract_results();

      // Truncate. 
      tp_u = Tensor::truncate(std::move(tp_u), 4, 1);
      tp_s = Tensor::truncate(std::move(tp_s), 1, 1);
      tp_v = Tensor::truncate(std::move(tp_v), 2, 1);

      auto loc_dims_u = tp_u->locDims();
      auto loc_dims_v = tp_v->locDims();
      loc_dims_u.erase(loc_dims_u.begin() + 2);
      loc_dims_v.erase(loc_dims_v.begin());
      tp_u->reshape(tp_u->disDims(), loc_dims_u);
      tp_v->reshape(tp_v->disDims(), loc_dims_v);

      // tp_u->print_serial("U");
      // tp_s->print_serial("S");
      // tp_v->print_serial("V");

      // Calculate SV. 
      auto&& els = tp_s->cast<DenseTensor>()->extractEls();
      bond_dims_.at(from + i) = count_bond_dim(els);

      tp_s = DiagTensor::make(
        tp_s->bc().env(), 
        {}, 
        { dis_chi_, loc_chi_, dis_chi_, loc_chi_ }, 
        false, 
        std::move(els)
      );

      tp_s = SymmTensorBase::rescatterIO(std::move(tp_s), 1);

      params = ConParams({{ 1, 0 }, { 3, 2 }});
      con = pcon(std::move(tp_s), std::move(tp_v), params);
      tptr tp_sv = con.contract();

      // Apply op to next site. 
      tp_site = std::move(site_tensors_.at(from + i + 1));
      tp_op = mpo.at(i + 1).copy();

      auto is_last = (i + 2 == mpo.nSites());

      params = ConParams({{ 2, 0 }});
      con = pcon(std::move(tp_site), std::move(tp_op), params);
      tp_site = con.contract();

      // Apply SV to next state. 
      tidx_tup_ids repls1 { 0, X, 3, X, X };
      auto repls2 = is_last
        ? tidx_tup_ids { X, 1, X, 4, 2, X }
        : tidx_tup_ids { X, 1, X, 5, 2, X, 4 };
      params = ConParams({{ 1, 0 }, { 4, 2 }, { 3, 5 }}, repls1, repls2);
      con = pcon(std::move(tp_sv), std::move(tp_site), params);
      tp_site = con.contract();

      site_canons_.at(from + i) = SITE_CANON::left;
      site_tensors_.at(from + i) = std::move(tp_u);
    }

    site_canons_.at(from + mpo.nSites() - 1) = SITE_CANON::none;
    site_tensors_.at(from + mpo.nSites() - 1) = std::move(tp_site);
  }

  void MPS::apply_old(const MPO& mpo, std::size_t from) {
    tptr tp_s1 = std::move(site_tensors_.at(from));
    tptr tp_op1 = mpo.at(0).copy();

    ConParams params1({{2, 0}});
    pcon con1(std::move(tp_s1), std::move(tp_op1), params1);
    tp_s1 = con1.contract();

    // TODO: Factor site at a time instead of 2. 
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
      auto& t = *tp_up->cast<DenseTensor>();
      for (auto i = 0UL; tp_up->bc().isActive() && i < tp_up->locSize(); ++i) {
        t[i] = std::conj(t[i]);
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

  // TODO: Use QR decomposition. 
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

      // // TODO: Check for non-zero truncation. 
      // Unnecessary if decomposing a tensor at a time. 
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

  // TODO: Use RQ decomposition. 
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

      // // TODO: Check for non-zero truncation. 
      // Unnecessary if decomposing a tensor at a time. 
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

  std::map<MPS::sample_t, std::size_t> MPS::sample(std::size_t from, std::size_t to, std::size_t n) {
    std::map<sample_t, std::size_t> occs {{{}, n}};
    const auto& bc = site(0).bc();

    leftCanonicalise(from);
    rightCanonicalise(from);

    for (auto i = 0UL; bc.isActive() && i < to - from; ++i) {
      std::map<sample_t, std::size_t> occs_new;
      
      // Iterate all samples generated so far. 
      for (const auto& [samp, m] : occs) {
        tptr tp_site = site(from).copy();

        // Contract current sample. 
        for (auto j = 0UL; j < from + i; ++j) {
          auto pdim = siteDims().at(from + j);
          std::vector<tel> els(pdim, 0);
          els.at(samp.at(j)) = 1;

          tptr tp_proj = DenseTensor::make(bc.env(), {}, { pdim }, std::move(els));

          ConParams con_params({{ 2, 0 }});
          tp_site = pcon(std::move(tp_site), std::move(tp_proj), con_params).contract();
          
          tptr tp_site_next = site(from + j + 1).copy();
          con_params = ConParams({{ 1, 0 }, { 3, 3 }});
          tp_site = pcon(std::move(tp_site), std::move(tp_site_next), con_params).contract();
        }

        tptr tp_up = std::move(tp_site);
        tptr tp_dn = tp_up->copy();
  
        // Conjugate DN tensor. 
        auto& t = *tp_dn->cast<DenseTensor>();
        for (auto j = 0UL; j < tp_dn->locSize(); ++j) {
          t[j] = std::conj(t[j]);
        }
  
        ConParams con_params({{ 0, 0 }, { 1, 1 }, { 3, 3 }, { 4, 4 }});
        tptr tp_rho = pcon(std::move(tp_up), std::move(tp_dn), con_params).contract();
        
        BcParams bc_params { 1, uint(disChi() * disChi()), bc.params().off };
        tp_rho = Tensor::rebcast(std::move(tp_rho), bc_params);
        
        // Extract site value probabilities. 
        auto pdim = siteDims().at(from + i);
        std::vector<double> cumul_ps(pdim);
        auto sum = 0.0;
        for (auto j = 0UL; j < pdim; ++j) {
          auto p = tp_rho->at({ j, j }).real();
          cumul_ps.at(j) = sum + p;
          sum += p;
        }

        // TODO: Can this be moved? 
        std::random_device rd;
        std::mt19937 gen (rd());
        gen.seed(1);

        // Generate site value samples. 
        std::vector<std::size_t> val_freqs(pdim);
        std::uniform_real_distribution<> dis(0.0, sum);
        for (auto j = 0UL; j < m; ++j) {
          auto r = dis(gen);
          for (auto k = 0UL; k < pdim; ++k) {
            if (r < cumul_ps.at(k)) {
              ++val_freqs.at(k);
              break;
            }
          }
        }

        // Insert non-zero site value samples. 
        for (auto j = 0UL; j < pdim; ++j) {
          auto freq = val_freqs.at(j);
          if (freq > 0) {
            auto samp_new = samp;
            samp_new.push_back(j);
            occs_new.at(samp_new) = freq;
          }
        }
      }

      occs = occs_new;
    }

    return occs;
  }

  std::unique_ptr<DenseTensor> MPS::toDense() && {
    const auto& bc = site_tensors_.at(0)->bc();
    std::vector<tel> els(locChi(), 0);
    if (utils::is_root()) els.at(0) = 1.0;

    tptr tp_res = DenseTensor::make(bc.env(), { disChi() }, { locChi() }, std::move(els));
    tptr tp_last = tp_res->copy();

    // tptr tp_res = std::move(site_tensors_.at(0));
    for (auto i = 0UL; i < nSites(); ++i) {
      tptr tp_tmp = std::move(site_tensors_.at(i));
      // auto tot_size = tp_res->totDims().size();

      ConParams params({{ 0, 0 }, { i + 1, 3 }});
      pcon con(std::move(tp_res), std::move(tp_tmp), params);
      tp_res = con.contract();
    }

    ConParams params({{ 0, 0 }, { nSites() + 1, 1 }});
    pcon con(std::move(tp_res), std::move(tp_last), params);
    return Tensor::cast<DenseTensor>(con.contract());
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
  , pdim_(site_ops_.at(0)->locDims().at(0))
  {}

  void MPO::rightCanonicalise() {
    auto n = nSites();
    for (auto i = 1UL; i < n; ++i) {
      tptr tp = std::move(site_ops_.at(n - i));

      // Group physical dims together. 
      auto dims = tp->locDims();
      dims.erase(dims.begin(), dims.begin() + 2);
      dims.insert(dims.begin(), pdim_ * pdim_);
      tp->reshape({}, dims);

      std::vector<tidx_tup_st> ptup { 1, 0 };
      if (i > 1) ptup.push_back(2);
      tp = Tensor::permute(std::move(tp), ptup);

      auto c = (i > 1) ? 2UL : 1UL;
      DecParams dp {{ 0, 0 }, { 1, c }, { 1, c }, { 0, 0 }, { 0, 0 }};
      Decomposer dec(std::move(tp), dp, true);
      dec.decompose();

      auto [tp_u, tp_s, tp_v] = dec.extract_results();
      auto bond_size = tp_s->locSize();

      // TODO: Truncate elements close to 0. 

      ptup = { 1, 0 };
      if (i > 1) ptup.push_back(2);
      tp_v = Tensor::permute(std::move(tp_v), ptup);
      
      // Ungroup physical dims and update. 
      dims = tp_v->locDims();
      dims.erase(dims.begin(), dims.begin() + 1);
      dims.insert(dims.begin(), { pdim_, pdim_ });
      tp_v->reshape({}, dims);

      site_ops_.at(n - i) = std::move(tp_v);

      // Calculate US. 
      auto&& els = tp_s->cast<DenseTensor>()->extractEls();

      tp_s = DiagTensor::make(
        tp_s->bc().env(), 
        {}, 
        { bond_size, bond_size }, 
        false, 
        std::move(els)
      );

      pcon con_us(std::move(tp_u), std::move(tp_s), ConParams({{ 1, 0 }}));
      auto tp_us = con_us.contract();

      auto w = (n - i > 1) ? 3UL : 2UL;
      ConParams params({{ w, 0 }});
      pcon con_prev(std::move(site_ops_.at(n - i - 1)), std::move(tp_us), params);
      site_ops_.at(n - i - 1) = con_prev.contract();
    }
  }

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
