#include <algorithm>
#include <functional>
#include <iostream>
#include <random>

#include "net/bcmps.hpp"
#include "ten/con/pair-defs.hpp"
#include "ten/dec/base.hpp"
#include "util/ops.hpp"
#include "util/vector.hpp"

namespace qtnh {
  BCMPS::BCMPS(const QTNHEnv& env, std::size_t n_sites, qtnh::tidx site_dim, chi_triple chis) 
  : BCMPS(env, n_sites, qtnh::tidx_tup(n_sites, site_dim), chis)
  {}

  BCMPS::BCMPS(const QTNHEnv& env, std::size_t n_sites, qtnh::tidx_tup site_dims, chi_triple chis) 
  : site_tensors_(n_sites)
  , site_canons_(n_sites, SITE_CANON::none)
  , site_dims_(site_dims)
  , bond_dims_(n_sites - 1, 1)
  , cyc_chi_(chis.at(0))
  , dis_chi_(chis.at(1))
  , blk_chi_(chis.at(2))
  {
    for (auto i = 0UL; i < n_sites; ++i) {
      auto size = site_dims.at(i) * locChi() * locChi();
      std::vector<qtnh::tel> els(size);
      if (env.proc_id == 0) els.at(0) = 1.0;

      site_tensors_.at(i) = DenseTensor::make(
        env, 
        { dis_chi_, dis_chi_ }, 
        { site_dims.at(i), cyc_chi_, blk_chi_, cyc_chi_, blk_chi_ }, 
        std::move(els)
      );
    }
  }


  BCMPS::BCMPS(qtnh::tptr, chi_triple, SITE_CANON) {
    utils::throw_unimplemented();
  }
  
  BCMPS::BCMPS(std::vector<qtnh::tptr>&& sites)
  : site_tensors_(std::move(sites))
  , site_canons_(site_tensors_.size(), SITE_CANON::none)
  , site_dims_(site_tensors_.size())
  , bond_dims_(site_tensors_.size() - 1, 1)
  , cyc_chi_(site_tensors_.at(0)->locDims().at(1))
  , dis_chi_(site_tensors_.at(0)->disDims().at(0))
  , blk_chi_(site_tensors_.at(0)->locDims().at(2))
  {
    for (auto i = 0UL; i < nSites(); ++i) {
      site_dims_.at(i) = site_tensors_.at(i)->locDims().at(0);
    }
  }

  BCMPS BCMPS::rand(const QTNHEnv& env, std::size_t n_sites, qtnh::tidx site_dim, chi_triple chis, std::size_t bond_dim) {
    std::mt19937 gen(2025);
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    std::vector<tptr> sites;
    BCMPS mps(env, n_sites, site_dim, chis);

    for (auto i = 0UL; i < n_sites; ++i) {
      auto tp = Tensor::cast<DenseTensor>(std::move(mps.site_tensors_.at(i)));
      auto loc_size = std::min(bond_dim, chis.at(0) * chis.at(2));
      auto dis_size = bond_dim / loc_size;

      // Global site-wise element calculation to ensure repeatability. 
      std::vector<tel> els(site_dim * bond_dim * bond_dim);
      for (auto j = 0UL; j < els.size(); ++j) {
        auto rel = dis(gen), img = dis(gen);
        els.at(j) = tel { rel, img };
      }

      for (auto j = 0UL; j < loc_size; ++j) {
        if (i == 0UL && j > 0UL) break;

        for (auto k = 0UL; k < loc_size; ++k) {
          if (i + 1 == n_sites && k > 0UL) break;

          auto p = env.proc_id / chis.at(1);
          auto q = env.proc_id % chis.at(1);

          if (p < dis_size && q < dis_size) {
            for (auto l = 0UL; l < site_dim; ++l) {
              auto pos = l * bond_dim * bond_dim + 
                (p * chis.at(1) + j) * bond_dim + 
                q * chis.at(1) + k;
              
              auto c1 = j / chis.at(2);
              auto c2 = k / chis.at(2);
              auto b1 = j % chis.at(2);
              auto b2 = k % chis.at(2);

              tp->at({ p, q, l, c1, b1, c2, b2 }) = els.at(pos);
            }
          }
        }
      }

      mps.site_tensors_.at(i) = std::move(tp);
    }

    mps.leftCanonicalise(n_sites - 1);
    mps.rightCanonicalise(0);
    mps.renormalise();

    return mps.copy();
  }

  BCMPS BCMPS::copy() {
    std::vector<tptr> sites;
    for (auto i = 0UL; i < nSites(); ++i) {
      sites.push_back(site_tensors_.at(i)->copy());
    }

    return BCMPS(std::move(sites));
  }

  void BCMPS::apply(tptr_symm tp, std::vector<std::size_t> sites, bool update_dims) {
    #ifdef DEBUG
      utils::barrier();
      if (utils::is_root()) {
        std::cout << "\nStarting: Apply MPS-tensor\n";
      }
    #endif

    using repl_vec = std::vector<tidx_tup_st>;
    auto min_site = *std::min_element(sites.begin(), sites.end());
    auto max_site = *std::max_element(sites.begin(), sites.end());

    // Contract all sites within range (min, max). 
    tptr tp_res = std::move(site_tensors_.at(min_site));
    for (auto i = min_site + 1; i <= max_site; ++i) {
      tptr tp_tmp = std::move(site_tensors_.at(i));
      auto tot_size = tp_res->totDims().size();

      ConParams params({{ 1, 0 }, { tot_size - 2, 3 }, { tot_size - 1, 4 }});
      pcon con(std::move(tp_res), std::move(tp_tmp), params);
      tp_res = con.contract();
    }

    // Contract sites with applied tensor. 
    std::vector<wire> wires(sites.size());
    for (auto i = 0UL; i < sites.size(); ++i) {
      auto rel_site = sites.at(i) - min_site;
      auto incr = rel_site == 0UL ? 0UL : 2UL;
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
        { 3, loc_size - 3 }, 
        { 2, loc_size - 4 }, 
        { 1, 1 }, 
        { 1, 1 }
      };

      Decomposer dec(std::move(tp_res), dp, true);
      auto type = update_dims ? DecType::SVD : DecType::QPD;
      dec.decompose(type);

      auto [tp_u, tp_s, tp_v] = dec.extract_results();
      tptr tp_sv;

      if (update_dims) {
        // Truncate. 
        tp_u = Tensor::truncate(std::move(tp_u), 5, 1);
        tp_s = Tensor::truncate(std::move(tp_s), 1, 1);
        tp_v = Tensor::truncate(std::move(tp_v), 2, 1);

        auto loc_dims_u = tp_u->locDims();
        auto loc_dims_v = tp_v->locDims();
        loc_dims_u.erase(loc_dims_u.begin() + 3);
        loc_dims_v.erase(loc_dims_v.begin());
        tp_u->reshape(tp_u->disDims(), loc_dims_u);
        tp_v->reshape(tp_v->disDims(), loc_dims_v);

        // tp_u->print_serial("U");
        // tp_s->print_serial("S");
        // tp_v->print_serial("V");

        // Calculate SV. 
        auto&& els = tp_s->cast<DenseTensor>()->extractEls();
        bond_dims_.at(i) = count_bond_dim(els);

        tp_s = DiagTensor::make(
          tp_s->bc().env(), 
          {}, 
          { dis_chi_, cyc_chi_, blk_chi_, dis_chi_, cyc_chi_, blk_chi_ }, 
          false, 
          std::move(els)
        );

        tp_s = SymmTensorBase::rescatterIO(std::move(tp_s), 1);

        auto dim_repls_2 = utils::concat_vecs(
          repl_vec { X, 1, X, X, 2 }, 
          utils::vec_incr(5UL, loc_size)
        );

        ConParams params(
          {{ 1, 0 }, { 3, 2 }}, 
          { 0, X, 3, 4, X, X }, 
          dim_repls_2
        );
        
        pcon con(std::move(tp_s), std::move(tp_v), params);
        tp_sv = con.contract();
      } else {
        // Truncate. 
        tp_u = Tensor::truncate(std::move(tp_u), 5, 1);
        tp_v = Tensor::truncate(std::move(tp_v), 2, 1);

        auto loc_dims_u = tp_u->locDims();
        auto loc_dims_v = tp_v->locDims();
        loc_dims_u.erase(loc_dims_u.begin() + 3);
        loc_dims_v.erase(loc_dims_v.begin());
        tp_u->reshape(tp_u->disDims(), loc_dims_u);
        tp_v->reshape(tp_v->disDims(), loc_dims_v);

        // tp_u->print_serial("Q");
        // tp_v->print_serial("R");
        // tp_s->print_serial("Pt");

        std::vector<wire> wires(max_site - i + 3);
        wires.at(0) = { 1, 0 };
        for (auto i = 1UL; i < wires.size(); ++i) {
          wires.at(i) = { i + 3, i + 1 };
        }

        auto repl1 = utils::concat_vecs(
          repl_vec { 0, X, 3, 4 }, 
          repl_vec(max_site - i + 2, X)
        );

        // ! This is very likely to fail... 
        auto repl2 = utils::concat_vecs(
          repl_vec { X, 1 }, 
          repl_vec(max_site - i + 2, X), 
          repl_vec { 2 }, 
          utils::vec_incr(5UL, max_site - i + 5)
        );

        ConParams params(wires, repl1, repl2);
        con = pcon(std::move(tp_v), std::move(tp_s), params);
        tp_sv = con.contract();

        // tp_sv->print_serial("SV");
      }

      site_canons_.at(i) = SITE_CANON::left;
      site_tensors_.at(i) = std::move(tp_u);
      tp_res = std::move(tp_sv);

      // Truncate. 
      // auto fmul = std::multiplies<tidx>();
      // auto loc_dims_u = utils::combine_part(tp_u->locDims(), 3, 5, tidx(1.0), fmul);
      // auto loc_dims_s = utils::combine_part(tp_s->locDims(), 1, 3, tidx(1.0), fmul);
      // auto loc_dims_v = utils::combine_part(tp_v->locDims(), 0, 2, tidx(1.0), fmul);

      // tp_u->reshape(tp_u->disDims(), loc_dims_u);
      // tp_s->reshape(tp_s->disDims(), loc_dims_s);
      // tp_v->reshape(tp_v->disDims(), loc_dims_v);

      // tp_u = Tensor::truncate(std::move(tp_u), 5, locChi());
      // tp_s = Tensor::truncate(std::move(tp_s), 1, locChi());
      // tp_v = Tensor::truncate(std::move(tp_v), 2, locChi());

      // // Calculate SV. 
      // auto&& els = tp_s->cast<DenseTensor>()->extractEls();
      // bond_dims_.at(i) = count_bond_dim(els);

      // tp_s = DiagTensor::make(
      //   tp_s->bc().env(), 
      //   {}, 
      //   { dis_chi_, cyc_chi_, blk_chi_, dis_chi_, cyc_chi_, blk_chi_ }, 
      //   false, 
      //   std::move(els)
      // );

      // tp_s = SymmTensorBase::rescatterIO(std::move(tp_s), 1);

      // auto dim_repls_2 = utils::concat_vecs(
      //   repl_vec { X, 1, X, X, 2 }, 
      //   utils::vec_incr(5UL, loc_size)
      // );

      // ConParams params(
      //   {{ 1, 0 }, { 3, 2 }}, 
      //   { 0, X, 3, 4, X, X }, 
      //   dim_repls_2
      // );
      
      // pcon con(std::move(tp_s), std::move(tp_v), params);
      // tptr tp_sv = con.contract();

      // site_canons_.at(i) = SITE_CANON::left;
      // site_tensors_.at(i) = std::move(tp_u);
      // tp_res = std::move(tp_sv);
    }

    site_canons_.at(max_site) = SITE_CANON::none;
    site_tensors_.at(max_site) = std::move(tp_res);

    #ifdef DEBUG
      utils::barrier();
      if (utils::is_root()) {
        std::cout << "\nFinished: Apply MPS-tensor\n";
      }
    #endif
  }

  void BCMPS::apply(const MPO& mpo, std::size_t from, bool update_dims) {
    #ifdef DEBUG
      utils::barrier();
      if (utils::is_root()) {
        std::cout << "\nStarting: Apply MPS-MPO\n";
      }
    #endif

    tptr tp_site = std::move(site_tensors_.at(from));
    tptr tp_op = mpo.at(0).copy();

    // ! Doesn't work for 1-site MPO. 
    ConParams params({{ 2, 0 }}, { 0, 1, X, 3, 4, 6, 7 }, { X, 2, 5 });
    pcon con(std::move(tp_site), std::move(tp_op), params);
    tp_site = con.contract();

    for (auto i = 0UL; i + 1 < mpo.nSites(); ++i) {
      DecParams dp {
        { 1, 1 }, 
        { 3, 3 }, 
        { 2, 2 }, 
        { 1, 1 }, 
        { 1, 1 }
      };

      // tp_site->print_serial("Original");

      Decomposer dec(std::move(tp_site), dp, true);

      auto type = update_dims ? DecType::SVD : DecType::QPD;
      dec.decompose(type);
      
      auto [tp_u, tp_s, tp_v] = dec.extract_results();
      tptr tp_sv;

      if (update_dims) {
        // Truncate. 
        tp_u = Tensor::truncate(std::move(tp_u), 5, 1);
        tp_s = Tensor::truncate(std::move(tp_s), 1, 1);
        tp_v = Tensor::truncate(std::move(tp_v), 2, 1);

        auto loc_dims_u = tp_u->locDims();
        auto loc_dims_v = tp_v->locDims();
        loc_dims_u.erase(loc_dims_u.begin() + 3);
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
          { dis_chi_, cyc_chi_, blk_chi_, dis_chi_, cyc_chi_, blk_chi_ }, 
          false, 
          std::move(els)
        );

        tp_s = SymmTensorBase::rescatterIO(std::move(tp_s), 1);

        params = ConParams({{ 1, 0 }, { 4, 2 }, { 5, 3 }});
        con = pcon(std::move(tp_s), std::move(tp_v), params);
        tp_sv = con.contract();
      } else {
        // params = ConParams({{ 1, 0 }, { 5, 2 }, { 6, 3 }, { 7, 4 }});
        // con = pcon(tp_u->copy(), tp_v->copy(), params);
        // tptr tp_check = con.contract();
        // con = pcon(std::move(tp_check), tp_s->copy(), params);
        // tp_check = con.contract();

        // tp_check->print_serial("Check");

        // Truncate. 
        tp_u = Tensor::truncate(std::move(tp_u), 5, 1);
        tp_v = Tensor::truncate(std::move(tp_v), 2, 1);

        auto loc_dims_u = tp_u->locDims();
        auto loc_dims_v = tp_v->locDims();
        loc_dims_u.erase(loc_dims_u.begin() + 3);
        loc_dims_v.erase(loc_dims_v.begin());
        tp_u->reshape(tp_u->disDims(), loc_dims_u);
        tp_v->reshape(tp_v->disDims(), loc_dims_v);

        // tp_u->print_serial("Q");
        // tp_v->print_serial("R");
        // tp_s->print_serial("Pt");

        params = ConParams({{ 1, 0 }, { 4, 2 }, { 5, 3 }, { 6, 4 }});
        con = pcon(std::move(tp_v), std::move(tp_s), params);
        tp_sv = con.contract();

        // tp_sv->print_serial("SV");
      }

      // Apply op to next site. 
      tp_site = std::move(site_tensors_.at(from + i + 1));
      tp_op = mpo.at(i + 1).copy();

      auto is_last = (i + 2 == mpo.nSites());

      params = ConParams({{ 2, 0 }});
      con = pcon(std::move(tp_site), std::move(tp_op), params);
      tp_site = con.contract();

      // Apply SV to next state. 
      tidx_tup_ids repls1 { 0, X, 3, 4, X, X, X };
      auto repls2 = is_last
        ? tidx_tup_ids { X, 1, X, X, 5, 6, 2, X }
        : tidx_tup_ids { X, 1, X, X, 6, 7, 2, X, 5 };
      params = ConParams({{ 1, 0 }, { 5, 2 }, { 6, 3 }, { 4, 7 }}, repls1, repls2);
      con = pcon(std::move(tp_sv), std::move(tp_site), params);
      tp_site = con.contract();

      site_canons_.at(from + i) = SITE_CANON::left;
      site_tensors_.at(from + i) = std::move(tp_u);
    }

    site_canons_.at(from + mpo.nSites() - 1) = SITE_CANON::none;
    site_tensors_.at(from + mpo.nSites() - 1) = std::move(tp_site);

    #ifdef DEBUG
      utils::barrier();
      if (utils::is_root()) {
        std::cout << "Finished: Apply MPS-MPO\n\n";
      }
    #endif
  }

  qtnh::tel BCMPS::overlap(BCMPS& mps) {
    //
    // XXX - XXX - XXX - XXX - XXX
    // XXX = XXX = XXX = XXX = XXX
    //  |     |     |     |     |
    // XXX - XXX - XXX - XXX - XXX
    // XXX = XXX = XXX = XXX = XXX 
    //

    // TODO: Start at lowest rank in MPS. 
    const auto& env = site_tensors_.at(0)->bc().env();
    std::vector<tel> els(mps.locChi() * locChi() , 0);
    if (utils::is_root()) els.at(0) = 1.0;
    tptr tp_res = DenseTensor::make(
      env, 
      { mps.disChi(), disChi() }, 
      { mps.cycChi(), mps.blkChi(), cycChi(), blkChi() }, 
      std::move(els)
    );

    for (auto i = 0UL; i < site_tensors_.size(); ++i) {
      tptr tp_up = mps.site(i).copy();
      tptr tp_dn = site_tensors_.at(i)->copy();

      // Conjugate UP tensor. 
      auto& t = *tp_up->cast<DenseTensor>();
      for (auto i = 0UL; tp_up->bc().isActive() && i < tp_up->locSize(); ++i) {
        t[i] = std::conj(t[i]);
      }

      ConParams params1({{ 0, 0 }, { 2, 3 }, { 3, 4 }});
      pcon con1(std::move(tp_res), std::move(tp_up), params1);
      tp_res = con1.contract();

      ConParams params2({{ 0, 0 }, { 2, 3 }, { 3, 4 }, { 4, 2 }});
      pcon con2(std::move(tp_res), std::move(tp_dn), params2);
      tp_res = con2.contract();
    }

    qtnh::tel res;
    if (utils::is_root()) {
      res = tp_res->at({ 0, 0, 0, 0, 0, 0 });
    }

    MPI_Bcast(&res, 1, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

    return res;
  }

  qtnh::tel BCMPS::norm() {
    return std::sqrt(overlap(*this));
  }

  void BCMPS::renormalise() {
    auto div = std::pow(norm(), 1.0 / double(nSites()));

    for (auto i = 0UL; i < site_tensors_.size(); ++i) {
      auto tp = Tensor::cast<DenseTensor>(std::move(site_tensors_.at(i)));
      for (auto i = 0UL; tp->bc().isActive() && i < tp->locSize(); ++i) {
        (*tp)[i] = (*tp)[i] / div;
      }

      site_tensors_.at(i) = std::move(tp);
    }
  }

  void BCMPS::leftCanonicalise(std::size_t to) {
    #ifdef DEBUG
      utils::barrier();
      if (utils::is_root()) {
        std::cout << "\nStarting: Left-canonicalise\n";
      }
    #endif

    auto can_continue = true;
    for (auto i = 0UL; i + 1 < to; ++i) {
      if (can_continue && site_canons_.at(i) == SITE_CANON::left) {
        continue;
      } else {
        can_continue = false;
      }

      tptr tp = std::move(site_tensors_.at(i));
      
      DecParams dp {{ 1, 1 }, { 3, 2 }, { 2, 1 }, { 1, 1 }, { 1, 1 }};
      Decomposer dec(std::move(tp), dp, true);
      dec.decompose(DecType::QRD);  // TODO: Use QR decomposition. 

      auto [tp_u, tp_s, tp_v] = dec.extract_results();

      // Calculate SV. 
      auto&& els = tp_s->cast<DenseTensor>()->extractEls();
      // bond_dims_.at(i) = count_bond_dim(els);

      tp_s = DiagTensor::make(
        tp_s->bc().env(), 
        {}, 
        { dis_chi_, cyc_chi_, blk_chi_, dis_chi_, cyc_chi_, blk_chi_ }, 
        false, 
        std::move(els)
      );

      tp_s = SymmTensorBase::rescatterIO(std::move(tp_s), 1);
      auto params = ConParams({{ 1, 0 }, { 4, 2 }, { 5, 3 }});
      auto con = pcon(std::move(tp_s), std::move(tp_v), params);
      tptr tp_sv = con.contract();

      tp = std::move(site_tensors_.at(i + 1));
      params = ConParams(
        {{ 1, 0 }, { 4, 3 }, { 5, 4 }}, 
        { 0, X, 3, 4, X, X }, 
        { X, 1, 2, X, X, 5, 6 }
      );

      con = pcon(std::move(tp_sv), std::move(tp), params);
      
      site_canons_.at(i) = SITE_CANON::left;
      site_canons_.at(i + 1) = SITE_CANON::none;
      site_tensors_.at(i) = std::move(tp_u);
      site_tensors_.at(i + 1) = con.contract();
    }

    #ifdef DEBUG
      utils::barrier();
      if (utils::is_root()) {
        std::cout << "\nFinished: Left-canonicalise\n";
      }
    #endif
  }

  
  void BCMPS::rightCanonicalise(std::size_t to) {
    #ifdef DEBUG
      utils::barrier();
      if (utils::is_root()) {
        std::cout << "\nStarting: Right-canonicalise\n";
      }
    #endif

    auto n = nSites();
    auto can_continue = true;
    for (auto i = 1UL; i < n - to; ++i) {
      if (can_continue && site_canons_.at(n - i) == SITE_CANON::right) {
        continue;
      } else {
        can_continue = false;
      }

      tptr tp = std::move(site_tensors_.at(n - i));
      auto ptup = PTupleSrc(tp->totDims().size());
      ptup.at(2) >> 2;

      tp = Tensor::permute(std::move(tp), ptup.toTar().tup());

      DecParams dp {{ 1, 1 }, { 2, 3 }, { 1, 2 }, { 1, 1 }, { 1, 1 }};
      Decomposer dec(std::move(tp), dp, true);
      dec.decompose(DecType::LQD);  // TODO: Use LQ decomposition. 

      auto [tp_u, tp_s, tp_v] = dec.extract_results();

      // Calculate US. 
      auto&& els = tp_s->cast<DenseTensor>()->extractEls();
      // bond_dims_.at(n - i - 1) = count_bond_dim(els);

      tp_s = DiagTensor::make(
        tp_s->bc().env(), 
        {}, 
        { dis_chi_, cyc_chi_, blk_chi_, dis_chi_, cyc_chi_, blk_chi_ }, 
        false, 
        std::move(els)
      );

      tp_s = SymmTensorBase::rescatterIO(std::move(tp_s), 1);
      auto params = ConParams({{ 1, 0 }, { 4, 2 }, { 5, 3 }});
      auto con = pcon(std::move(tp_u), std::move(tp_s), params);
      tptr tp_us = con.contract();

      tp = std::move(site_tensors_.at(n - i - 1));
      params = ConParams({{ 1, 0 }, { 5, 2 }, { 6, 3 }});
      con = pcon(std::move(tp), std::move(tp_us), params);

      // Wrong index order in V. 
      ptup = PTupleSrc(tp_v->totDims().size());
      ptup.at(4) << 2;

      site_canons_.at(n - i) = SITE_CANON::right;
      site_canons_.at(n - i - 1) = SITE_CANON::none;
      site_tensors_.at(n - i) = Tensor::permute(std::move(tp_v), ptup.toTar().tup());
      site_tensors_.at(n - i - 1) = con.contract();
    }

    #ifdef DEBUG
      utils::barrier();
      if (utils::is_root()) {
        std::cout << "\nFinished: Right-canonicalise\n";
      }
    #endif
  }

  void BCMPS::swap(std::size_t k) {
    tptr tp1 = std::move(site_tensors_.at(k));
    tptr tp2 = std::move(site_tensors_.at(k + 1));

    // Swap physical indices via index replacement. 
    ConParams c_par(
      {{ 1, 0 }, { 5, 3 }, { 6, 4 }}, 
      { 0, X, 5, 3, 4, X, X }, 
      { X, 1, 2, X, X, 6, 7 }
    );

    auto con = pcon(std::move(tp1), std::move(tp2), c_par);
    tptr tp12 = con.contract();

    DecParams d_par {
      { 1, 1 }, 
      { 3, 3 }, 
      { 2, 2 }, 
      { 1, 1 }, 
      { 1, 1 }
    };

    Decomposer dec(std::move(tp12), d_par, true);
    dec.decompose(DecType::QPD);
    auto [tp_q, tp_pt, tp_r] = dec.extract_results();

    // Truncate. 
    tp_q = Tensor::truncate(std::move(tp_q), 5, 1);
    tp_r = Tensor::truncate(std::move(tp_r), 2, 1);

    auto loc_dims_q = tp_q->locDims();
    auto loc_dims_r = tp_r->locDims();
    loc_dims_q.erase(loc_dims_q.begin() + 3);
    loc_dims_r.erase(loc_dims_r.begin());
    tp_q->reshape(tp_q->disDims(), loc_dims_q);
    tp_r->reshape(tp_r->disDims(), loc_dims_r);

    c_par = ConParams(
      {{ 1, 0 }, { 4, 2 }, { 5, 3 }, { 6, 4 }}, 
      { 0, X, 3, 4, X, X, X }, 
      { X, 1, X, X, X, 2, 5, 6 }
    );

    con = pcon(std::move(tp_r), std::move(tp_pt), c_par);
    site_tensors_.at(k) = std::move(tp_q);
    site_tensors_.at(k + 1) = con.contract();
    site_canons_.at(k) = SITE_CANON::left;
    site_canons_.at(k + 1) = SITE_CANON::none;
  }

  void BCMPS::permute(PTupleTar ptup) {
    auto tup = ptup.tup();
    while (true) {
      auto sorted = true;
      for (auto i = 0UL; i < tup.size() - 1; ++i) {
        if (tup.at(i) > tup.at(i + 1)) {
          sorted = false;
          
          // ? Is this necessary? 
          leftCanonicalise(i + 1);
          rightCanonicalise(i);

          swap(i);
          std::swap(tup.at(i), tup.at(i + 1));
        }
      }

      if (sorted) return;
    }
  }

  std::map<BCMPS::sample_t, std::size_t> BCMPS::sample(std::size_t from, std::size_t to, std::size_t n) {
    leftCanonicalise(from);
    rightCanonicalise(from);

    std::mt19937 gen(2025);

    std::map<sample_t, std::size_t> occs {{{}, n}};
    const auto& bc = site(0).bc();

    for (auto i = 0UL; bc.isActive() && i < to - from; ++i) {
      std::map<sample_t, std::size_t> occs_new;
      
      // Iterate all samples generated so far. 
      for (const auto& [samp, m] : occs) {
        tptr tp_site = site(from).copy();
        tp_site = Tensor::permute(std::move(tp_site), { 0, 1, 4, 2, 3, 5, 6 });

        // Contract current sample. 
        for (auto j = 0UL; j < from + i; ++j) {
          auto pdim = siteDims().at(from + j);
          std::vector<tel> els(pdim, 0);
          els.at(samp.at(j)) = 1;

          tptr tp_proj = DenseTensor::make(bc.env(), {}, { pdim }, std::move(els));

          ConParams con_params({{ 4, 0 }});
          tp_site = pcon(std::move(tp_site), std::move(tp_proj), con_params).contract();
          
          tptr tp_site_next = site(from + j + 1).copy();
          con_params = ConParams({{ 1, 0 }, { 4, 3 }, { 5, 4 }});
          tp_site = pcon(std::move(tp_site), std::move(tp_site_next), con_params).contract();
        }

        tptr tp_up = std::move(tp_site);
        tptr tp_dn = tp_up->copy();
  
        // Conjugate DN tensor. 
        auto& t = *tp_dn->cast<DenseTensor>();
        for (auto j = 0UL; j < tp_dn->locSize(); ++j) {
          t[j] = std::conj(t[j]);
        }
        
        ConParams con_params({{ 0, 0 }, { 1, 1 }, { 2, 2 }, { 3, 3 }, { 5, 5 }, { 6, 6 }});
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
            occs_new.insert({ samp_new, freq });
          }
        }
      }

      occs = occs_new;
    }

    return occs;
  }

  std::unique_ptr<DenseTensor> BCMPS::toDense() && {
    const auto& bc = site_tensors_.at(0)->bc();
    std::vector<tel> els(locChi(), 0);
    if (utils::is_root()) els.at(0) = 1.0;

    tptr tp_res = DenseTensor::make(
      bc.env(), 
      { disChi() }, 
      { cycChi(), blkChi() }, 
      std::move(els)
    );
    tptr tp_last = tp_res->copy();

    // tptr tp_res = std::move(site_tensors_.at(0));
    for (auto i = 0UL; i < nSites(); ++i) {
      tptr tp_tmp = std::move(site_tensors_.at(i));
      // auto tot_size = tp_res->totDims().size();

      ConParams params({{ 0, 0 }, { i + 1, 3 }, { i + 2, 4 }});
      pcon con(std::move(tp_res), std::move(tp_tmp), params);
      tp_res = con.contract();
    }

    ConParams params({{ 0, 0 }, { nSites() + 1, 1 }, { nSites() + 2, 2 }});
    pcon con(std::move(tp_res), std::move(tp_last), params);
    return Tensor::cast<DenseTensor>(con.contract());
  }

  void BCMPS::print() const {
    utils::barrier();

    if (utils::is_root()) {
      std::cout << "================================================================\n";
      std::cout << "MPS with N=" << site_tensors_.size() << 
        " chi=(" << cyc_chi_ << "," << dis_chi_ << "," << blk_chi_ << ")\n";
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
}
