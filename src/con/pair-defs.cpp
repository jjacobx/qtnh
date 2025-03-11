#include <algorithm>
#include <iostream>
#include <numeric>

#include "con/pair-defs.hpp"
#include "core/utils.hpp"
#include "lalg/wrappers.hpp"
#include "tensor/indexing.hpp"
#include "tensor/ptuple.hpp"


namespace qtnh {
  void _local_contraction(Tensor* tp1, Tensor* tp2, DenseTensor* tp3, 
                          TIndexing ti1, TIndexing ti2, TIndexing ti3) {
    auto it3 = ti3.keep("local").num("local").begin();
    for (auto idxs1 : ti1.tup("local")) {
      for (auto idxs2 : ti2.tup("local")) {
        qtnh::tel el3 = 0.0;

        #ifdef DEBUG
          using namespace qtnh::ops;
          std::cout << t3.bc().env.proc_id << " | t3[" << *it3 << "] = ";
        #endif

        auto it1 = ti1.num("closed", idxs1);
        auto it2 = ti2.num("closed", idxs2);
        while(it1 != it1.end() && it2 != it2.end()) {
          #ifdef DEBUG
            std::cout << "t1[" << *it1 << "] * t2[" << *it2 << "]";
          #endif

          el3 += (*tp1)[*it1] * (*tp2)[*it2];
          ++it1; ++it2;

          #ifdef DEBUG
            if (it1 != it1.end() && it2 != it2.end()) std::cout << " + ";
          #endif
        }

        (*tp3)[*it3] = el3;
        ++it3;


        #ifdef DEBUG
          std::cout << " = " << el3  << std::endl;
        #endif
      }
    }
  }

  void _repad(const lalg::ProcGrid& pg1, const lalg::ProcGrid& pg2, 
              std::vector<qtnh::tel>& els, int loc_size) {
    using namespace ops;

    int pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    auto psrc = pg1.getPNum(pg2.procIdxs());
    auto ptar = pg2.getPNum(pg1.procIdxs());

    if (psrc >= 0 && psrc == ptar) return;
    
    if (ptar > -1) {
      MPI_Ssend(els.data(), loc_size, MPI_DOUBLE_COMPLEX, ptar, 0, MPI_COMM_WORLD);
    }

    if (psrc > -1) {
      els.resize(loc_size);
      MPI_Recv(els.data(), loc_size, MPI_DOUBLE_COMPLEX, psrc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (pg2.active()) {
      els.resize(loc_size);
      std::fill(els.begin(), els.end(), 0);
    } else {
      els.resize(0);
    }
    
    return;
  }

  template<> qtnh::tptr PairContractor<DenseTensor, DenseTensor>::contract_scalapack() {
    // #ifdef DEBUG
    if (utils::is_root())
      std::cout << "STARTING DENSE-DENSE CONTRACTION USING SCALAPACK\n";
    // #endif

    auto ws = params_.wires;
    // auto& env = tp1_->bc().env();

    auto ndis1 = tp1_->disDims().size();
    auto nloc1 = tp1_->locDims().size();
    auto ndis2 = tp2_->disDims().size();
    auto nloc2 = tp2_->locDims().size();

    PTupleSrc ptup1(ndis1 + nloc1);
    PTupleSrc ptup2(ndis2 + nloc2);

    auto n_dis_ws = 0UL, n_loc_ws = 0UL;
    auto dis_ws_size = 1UL;
    for (auto w : ws) {
      auto [w1, w2] = w;
      // Get current positions of given indices. 
      auto i = ptup1.inv().tup().at(w1);
      auto j = ptup2.inv().tup().at(w2);
      
      // Move first tensor dims to back, 
      // and second tensor dims to front. 
      if (w1 < ndis1) {
        dis_ws_size *= tp1_->disDims().at(w1);
        ptup1.at(i) >> int(ndis1 - i - 1);
        ptup2.at(j) << int(j - n_dis_ws++);
      } else {
        ptup1.at(i) >> int(ndis1 + nloc1 - i - 1);
        ptup2.at(j) << int(j - ndis2 - n_loc_ws++);
      }
    }

    auto dis_size_1 = tp1_->disSize() / dis_ws_size;
    auto dis_size_2 = tp2_->disSize() / dis_ws_size;

    auto use_bt = dis_ws_size > dis_size_1 && dis_ws_size > dis_size_2;

    auto grows = std::max(dis_size_1, dis_ws_size);
    auto gcols = std::max(dis_size_2, dis_ws_size);
    if (use_bt) {
      grows = std::max(dis_size_1, dis_size_2);
      gcols = dis_ws_size;
    }

    

    // Save permuted dims. 
    auto dims1 = utils::split_vec_rel(ptup1.apply(tp1_->totDims()), 
                                      ndis1 - n_dis_ws, n_dis_ws, nloc1 - n_loc_ws);
    auto dims2 = utils::split_vec_rel(ptup2.apply(tp2_->totDims()), 
                                      n_dis_ws, ndis2 - n_dis_ws, n_loc_ws);

    // Convert to column-major. 
    auto gs1 = utils::split_vec_rel(PTupleSrc(ndis1 + nloc1).tup(), 
                                    ndis1 - n_dis_ws, n_dis_ws, nloc1 - n_loc_ws);
    auto gs2 = utils::split_vec_rel(PTupleSrc(ndis2 + nloc2).tup(), 
                                    n_dis_ws, ndis2 - n_dis_ws, n_loc_ws);
    
    IndexGroup ig1({ "rd", "cd", "rb", "cb" }, utils::arr_to_vec(gs1));
    IndexGroup ig2({ "rd", "cd", "rb", "cb" }, utils::arr_to_vec(gs2));
    ig1.reorder({ "rd", "cd", "cb", "rb" });
    ig2.reorder({ "cd", "rd", "rb", "cb" });

    ptup1 = ig1.ptup() * ptup1;
    ptup2 = ig2.ptup() * ptup2;

    if (utils::is_root()) {
      using namespace ops;
      std::cout << ptup1.tup() << "\n";
      std::cout << ptup2.tup() << "\n";
    }

    tp1_ = Tensor::cast<DenseTensor>(Tensor::permute(std::move(tp1_), ptup1.toTar().tup()));
    tp2_ = Tensor::cast<DenseTensor>(Tensor::permute(std::move(tp2_), ptup2.toTar().tup()));

    // Calculate matrix parameters. 
    // TODO: Use B^T if contracted distributed dimensions are larger. 
    std::array<int, 4> sizes1, sizes2;
    auto fun = [](auto dims) { return int(utils::dims_to_size(dims)); };
    std::transform(dims1.begin(), dims1.end(), sizes1.begin(), fun);
    std::transform(dims2.begin(), dims2.end(), sizes2.begin(), fun);

    auto md = sizes1.at(0), nd = sizes2.at(1), kd = sizes1.at(1);
    auto ml = sizes1.at(2), nl = sizes2.at(3), kl = sizes1.at(3);
    auto pm = std::max(nd, md);
    auto pn = std::max(nd, kd);

    // auto md = std::max(sizes1.at(0), sizes2.at(1));
    // auto nd = std::max(sizes1.at(1), sizes2.at(0));
    // auto m = md * sizes1.at(2);
    // auto n = md * sizes2.at(3);
    // auto k = nd * sizes1.at(3);

    // TODO: Implement below directly with MPI Routines
    // TODO: e.g. _pad_els(els, nd_rows, nd_cols, to_rows, to_cols)
    // auto params1 = tp1_->bc().params();
    // params1.str = qtnh::uint(sizes1.at(1));
    // auto tp_tmp = DenseTensor::make(env, { std::size_t(sizes1.at(0)) }, 
    //                                 { std::size_t(sizes1.at(2) * sizes1.at(3)) }, 
    //                                 tp1_->extractEls(), params1);
    // params1.str = qtnh::uint(nd);
    // tp_tmp = Tensor::cast<DenseTensor>(Tensor::rebcast(std::move(tp_tmp), params1));
    // auto els1 = tp_tmp->extractEls();

    // auto params2 = tp2_->bc().params();
    // params2.str = qtnh::uint(sizes2.at(0));
    // auto tp_tmp = DenseTensor::make(env, { std::size_t(sizes2.at(1)) }, 
    //                                 { std::size_t(sizes2.at(2) * sizes2.at(3)) }, 
    //                                 tp2_->extractEls(), params2);
    // params2.str = qtnh::uint(nd);
    // tp_tmp = Tensor::cast<DenseTensor>(Tensor::rebcast(std::move(tp_tmp), params2));
    // auto els2 = tp_tmp->extractEls();

    if (utils::is_root()) {
      using namespace ops;
      std::cout << dims1.at(0) << dims1.at(1) << dims1.at(2) << dims1.at(3) << "\n";
      std::cout << dims2.at(0) << dims2.at(1) << dims2.at(2) << dims2.at(3) << "\n";
      std::cout << gs1.at(0) << gs1.at(1) << gs1.at(2) << gs1.at(3) << "\n";
      std::cout << gs2.at(0) << gs2.at(1) << gs2.at(2) << gs2.at(3) << "\n";
      std::cout << ig1.ptup().tup() << "\n";
      std::cout << ig2.ptup().tup() << "\n";
      std::cout << "md = " << md <<
        ", nd = " << nd <<
        ", kd = " << kd <<
        ", ml = " << ml <<
        ", nl = " << nl <<
        ", kl = " << kl <<
        ", pm = " << pm <<
        ", pn = " << pn << "\n";
      std::cout << "m1_b = (" << sizes1.at(2) << ", " << sizes1.at(3) << 
        "), m2_b = (" << sizes2.at(2) << ", " << sizes2.at(3) << ")\n";
    }

    using namespace lalg;
    ProcGrid pg1(md, kd);
    ProcGrid pg2(nd, kd);
    ProcGrid pg3(nd, md);
    ProcGrid pg_all(pm, pn);
    
    auto els1 = tp1_->extractEls();
    auto els2 = tp2_->extractEls();

    _repad(pg1, pg_all, els1, ml * kl);
    _repad(pg2, pg_all, els2, nl * kl);

    utils::barrier();
    std::cout << tp1_->bc().env().proc_id << " | els1 = ";
    for (auto e : els1) {
      std::cout << e << ", ";
    }
    std::cout << "\n";

    std::cout << tp1_->bc().env().proc_id << " | els2 = ";
    for (auto e : els2) {
      std::cout << e << ", ";
    }
    std::cout << "\n";
    utils::barrier();

    // Zero-pad inactive processes in the grid. 
    // TODO: Zero-pad active processes too, to match contracted dimensions. 
    // ! Problem: zero-padding columns requires different distribution! 
    // if (pg.active()) {
    //   std::cout << tp1_->locSize() << ", " << (m * k) / (md * nd) << "\n";
    //   std::cout << tp2_->locSize() << ", " << (k * n) / (md * nd) << "\n";
    //   els1.resize((m * k) / (md * nd), 0);
    //   els2.resize((k * n) / (md * nd), 0);
    // }

    // Create matrices and multiply. 
    BlockCyclicMatrix m1(pg_all, { pm * ml, pn * kl }, { ml, kl }, std::move(els1));
    BlockCyclicMatrix m2(pg_all, { pm * nl, pn * kl }, { nl, kl }, std::move(els2));

    if (utils::is_root()) {
      using namespace ops;
      std::cout << "M1.blk = " << m1.blkDims() << "\n";
      std::cout << "M2.blk = " << m2.blkDims() << "\n";
      std::cout << "M1.dis = " << m1.disDims() << "\n";
      std::cout << "M2.dis = " << m2.disDims() << "\n";
      std::cout << "M1.cyc = " << m1.cycDims() << "\n";
      std::cout << "M2.cyc = " << m2.cycDims() << "\n";
    }

    auto m3 = PZGEMM(std::move(m1), std::move(m2), false, true);

    utils::barrier();
    std::cout << "GEMM COMPLETE\n";

    if (utils::is_root()) {
      using namespace ops;
      std::cout << "M3.blk = " << m3.blkDims() << "\n";
      std::cout << "M3.dis = " << m3.disDims() << "\n";
      std::cout << "M3.cyc = " << m3.cycDims() << "\n";
    }

    auto dis_dims3 = utils::concat_dims(dims1.at(0), dims2.at(1));
    auto loc_dims3 = utils::concat_dims(dims2.at(3), dims1.at(2)); // column-major
    
    auto new_els = m3.extractEls();
    std::cout << tp1_->bc().env().proc_id << " | els3 = ";
    for (auto e : new_els) {
      std::cout << e << ", ";
    }
    std::cout << "\n";

    _repad(pg_all, pg3, new_els, ml * nl);

    tptr tp3 = DenseTensor::make(tp1_->bc().env(), dis_dims3, loc_dims3, std::move(new_els));

    // Permute back to row-major. 
    PTupleSrc ptup3(tp3->totDims().size());
    auto gs3 = utils::split_vec_rel(ptup3.tup(), dims1.at(0).size(), dims2.at(1).size(), 
                                    dims1.at(2).size(), dims2.at(3).size());

    IndexGroup ig3({ "rd", "cd", "rb", "cb" }, utils::arr_to_vec(gs3));
    ig3.reorder({ "rd", "cd", "cb", "rb" });
    tp3 = Tensor::permute(std::move(tp3), ig3.ptup().inv().toTar().tup());

    if (utils::is_root()) {
      using namespace ops;
      std::cout << ig1.ptup().tup() << "\n";
      std::cout << ig2.ptup().tup() << "\n";
      std::cout << ig3.ptup().tup() << "\n";
      // std::cout << "m = " << m <<
      //   ", n = " << n <<
      //   ", k = " << k <<
      //   ", md = " << md <<
      //   ", nd = " << nd << "\n";
      std::cout << dis_dims3 << ", " << loc_dims3 << "\n";
    }

    return tp3;
  }

  template<> qtnh::tptr PairContractor<DenseTensor, DenseTensor>::contract() {
    #ifdef DEBUG
      if (utils::is_root())
        std::cout << "STARTING DENSE-DENSE CONTRACTION\n";
    #endif

    auto ws = params_.wires;

    auto ndis1 = tp1_->disDims().size();
    auto nloc1 = tp1_->locDims().size();
    auto ndis2 = tp2_->disDims().size();
    auto nloc2 = tp2_->locDims().size();

    // STEP 1: Permute distributed contracted dims. 
    std::vector<qtnh::tidx_tup_st> ptup1(tp1_->totDims().size());
    std::vector<qtnh::tidx_tup_st> ptup2(tp2_->totDims().size());
    std::iota(ptup1.begin(), ptup1.end(), 0);
    std::iota(ptup2.begin(), ptup2.end(), 0);

    std::sort(ws.begin(), ws.end(), utils::wirecomp::second);

    auto ndis_cons = 0u;
    for (auto w : ws) {
      if (w.first < ndis1) {
        if (w.first < ndis1) ptup1.at(w.first) = ndis1 - ndis_cons - 1;
        for (auto i = w.first + 1; i < ndis1; ++i) {
          if (ptup1.at(i) < ndis1 - ndis_cons) --ptup1.at(i);
        }

        // Wires are sorted by second, so this is guaranteed to update all previous values. 
        if (w.second < ndis2) ptup2.at(w.second) = 0;
        for (auto i = w.second; i > 0; --i) {
          ++ptup2.at(i - 1);
        }

        ++ndis_cons;
      }
    }

    // * Temporary – establish default index replacements. 
    // * This might have to be moved somewhere else. 
    if (params_.useDefRepls) {
      params_.dimRepls1 = std::vector<qtnh::tidx_tup_st>(tp1_->totDims().size(), UINT16_MAX);
      std::sort(params_.wires.begin(), params_.wires.end(), utils::wirecomp::first);

      for  (auto i = 0u, j = 0u; i < tp1_->totDims().size(); ++i) {
        if ((j < params_.wires.size()) && (i == params_.wires.at(j).first)) {
          ++j;
        } else {
          params_.dimRepls1.at(i) = i - j;
          if (i >= tp1_->disDims().size()) {
            params_.dimRepls1.at(i) += (tp2_->disDims().size() - ndis_cons);
          }
        }
      }

      params_.dimRepls2 = std::vector<qtnh::tidx_tup_st>(tp2_->totDims().size(), UINT16_MAX);
      std::sort(params_.wires.begin(), params_.wires.end(), utils::wirecomp::second);

      for  (auto i = 0u, j = 0u; i < tp2_->totDims().size(); ++i) {
        if ((j < params_.wires.size()) && (i == params_.wires.at(j).second)) {
          ++j;
        } else {
          params_.dimRepls2.at(i) = tp1_->disDims().size() - ndis_cons + i - j;
          if (i >= tp2_->disDims().size()) {
            params_.dimRepls2.at(i) = tp1_->totDims().size() - params_.wires.size() + i - j;
          }
        }
      }
    }

    #ifdef DEBUG
      using namespace ops;
      if (utils::is_root()) {
        std::cout << "T1 dimension replacements: " << params_.dimRepls1 << "\n";
        std::cout << "T2 dimension replacements: " << params_.dimRepls2 << "\n";
      }
    #endif

    tp1_ = Tensor::cast<DenseTensor>(Tensor::permute(std::move(tp1_), ptup1));
    tp2_ = Tensor::cast<DenseTensor>(Tensor::permute(std::move(tp2_), ptup2));

    // Update dimension replacements after permutations. 
    // ! params_.dimRepls1 are unavailable. 
    auto dim_repls1_p = utils::permute_vec(params_.dimRepls1, ptup1);
    auto dim_repls2_p = utils::permute_vec(params_.dimRepls2, ptup2);

    // STEP 2: Align by broadcast. 
    auto dis_dims1 = tp1_->disDims();
    auto dis_dims2 = tp2_->disDims();
    dis_dims1.erase(dis_dims1.end() - ndis_cons, dis_dims1.end());
    dis_dims2.erase(dis_dims2.begin(), dis_dims2.begin() + ndis_cons);

    auto align_str = (qtnh::uint)utils::dims_to_size(dis_dims2);
    auto align_cyc = (qtnh::uint)utils::dims_to_size(dis_dims1);
    auto align_off = std::min(tp1_->bc().params().off, tp2_->bc().params().off);
    tp1_ = Tensor::cast<DenseTensor>(Tensor::rebcast(std::move(tp1_), { align_str, 1, align_off }));
    tp2_ = Tensor::cast<DenseTensor>(Tensor::rebcast(std::move(tp2_), { 1, align_cyc, align_off }));

    // STEP 3: Contract local wires. 
    std::vector<TIFlag> ifls1(ndis1, { "distributed", 0 });
    std::vector<TIFlag> ifls2(ndis2, { "distributed", 0 });
    ifls1.insert(ifls1.end(), nloc1, { "local", 0 });
    ifls2.insert(ifls2.end(), nloc2, { "local", 0 });

    for (auto i = 0u; i < ws.size(); ++i) {
      if (ws.at(i).first < ndis1) {
        ifls1.at(ws.at(i).first) = { "reduced", static_cast<int>(i) };
        ifls2.at(ws.at(i).second) = { "reduced", static_cast<int>(i) };
      } else {
        ifls1.at(ws.at(i).first) = { "closed", static_cast<int>(i) };
        ifls2.at(ws.at(i).second) = { "closed", static_cast<int>(i) };
      }
    }

    TIndexing ti1(tp1_->totDims(), ifls1);
    TIndexing ti2(tp2_->totDims(), ifls2);
    auto ti3_dis = TIndexing::app(ti1.keep("distributed"), ti2.keep("distributed"));
    auto ti3_loc = TIndexing::app(ti1.keep("local"), ti2.keep("local"));

    std::vector<qtnh::tidx_tup_st> ptup_loc(ti3_loc.dims().size());
    std::iota(ptup_loc.begin(), ptup_loc.end(), 0);

    for (auto i = tp1_->disDims().size(), j = i; i < tp1_->totDims().size(); ++i) {
      if (ifls1.at(i).label != "closed") {
        ptup_loc.at(i - j) = dim_repls1_p.at(i) - ti3_dis.dims().size();
      } else {
        ++j;
      }
    }

    auto split = ti1.keep("local").dims().size();
    for (auto i = tp2_->disDims().size(), j = i; i < tp2_->totDims().size(); ++i) {
      if (ifls2.at(i).label != "closed") {
        ptup_loc.at(split + i - j) = dim_repls2_p.at(i) - ti3_dis.dims().size();
      } else {
        ++j;
      }
    }

    qtnh::tidx_tup dims_ti3_loc(ti3_loc.dims().size());
    std::vector<TIFlag> ifls_ti3_loc(ti3_loc.ifls().size());
    for (auto i = 0u; i < ti3_loc.dims().size(); ++i) {
      dims_ti3_loc.at(ptup_loc.at(i)) = ti3_loc.dims().at(i);
      ifls_ti3_loc.at(ptup_loc.at(i)) = { "local",  static_cast<int>(i) };
    }

    auto ti3 = TIndexing::app(
      ti1.keep("distributed"), 
      ti1.keep("reduced"), 
      ti2.keep("distributed"), 
      TIndexing(dims_ti3_loc, ifls_ti3_loc)
    );

    std::size_t loc_size = 0;
    if (tp1_->bc().isActive() && tp2_->bc().isActive()) {
      loc_size = utils::dims_to_size(ti3.cut("distributed").dims());
    }

    auto els = std::vector<qtnh::tel>(loc_size);
    DenseTensor t3 { 
      tp1_->bc().env(), 
      ti3.cut("local").dims(), 
      ti3.keep("local").dims(), 
      std::move(els), 
      { 1, 1, align_off } 
    };

    ti1 = ti1.cut("distributed").cut("reduced");
    ti2 = ti2.cut("distributed").cut("reduced");

    if (t3.bc().isActive()) {
      _local_contraction(tp1_.get(), tp2_.get(), &t3, ti1, ti2, ti3);

      if (ndis_cons > 0) {
        // STEP 4: All-reduce distributed wires. 
        auto dis_idxs = utils::i_to_idxs(t3.bc().gid(), t3.disDims());
        for (auto i = 0u; i < dis_idxs.size(); ++i) {
          if (ti3.ifls().at(i).label == "reduced") dis_idxs.at(i) = 0;
        }
        
        auto colour = int(utils::idxs_to_i(dis_idxs, t3.disDims()));

        // ! Expect MPI memory limit issues. 
        // ! Can be performed multiple times with offset for larger arrays. 
        MPI_Comm allr_comm;
        MPI_Comm_split(t3.bc().gcomm(), colour, t3.bc().gid(), &allr_comm);
        MPI_Allreduce(MPI_IN_PLACE, t3.loc_els_.data(), int(loc_size), 
                      MPI_C_DOUBLE_COMPLEX, MPI_SUM, allr_comm);
        MPI_Comm_free(&allr_comm);
      }
    }

    // STEP 5: Convert virtual index to stretch factor. 
    std::vector<qtnh::tidx_tup_st> ptup3(t3.totDims().size());
    std::iota(ptup3.begin(), ptup3.end(), 0);
    
    for (auto i = 0u; i < ndis1 - ndis_cons; ++i) {
      ptup3.at(i) = dim_repls1_p.at(i);
    } 
    for (auto i = 1u; i <= ndis_cons; ++i) {
      ptup3.at(ndis1 - i) = t3.disDims().size() - i;
    }
    for (auto i = 0u; i < ndis2 - ndis_cons; ++i) {
      ptup3.at(ndis1 + i) = dim_repls2_p.at(ndis_cons + i);
    }

    t3._permute_internal(&t3, ptup3);

    // Last n distributed indices are virtual. 
    auto [new_dis_dims, virtual_dims] = utils::split_dims(t3.disDims(), t3.disDims().size() - ndis_cons);
    BcParams new_params { static_cast<qtnh::uint>(utils::dims_to_size(virtual_dims)), 1, align_off };

    return DenseTensor::make(t3.bc().env(), new_dis_dims, t3.locDims(), std::move(t3.loc_els_), new_params);
  }

  template<> qtnh::tptr PairContractor<DenseTensor, SymmTensor>::contract() {
    #ifdef DEBUG
      if (utils::is_root())
        std::cout << "STARTING DENSE-SYMM CONTRACTION\n";
    #endif

    std::size_t input_count = 0;
    for (auto w : params_.wires) {
      auto is_input_dis = w.second < (tp2_->disDims().size() / 2);
      auto is_input_loc = (w.second >= tp2_->disDims().size()) && 
                          (w.second < (tp2_->disDims().size() + tp2_->locDims().size() / 2));
      if (is_input_dis || is_input_loc) {
        ++input_count;
      }
    }

    auto dis_imbal = 0;

    if ((dis_imbal == 0) && (input_count == tp2_->disDims().size() / 2 + tp2_->locDims().size() / 2)) {
      params_.dimRepls1 = std::vector<qtnh::tidx_tup_st>(tp1_->totDims().size(), UINT16_MAX);
      std::sort(params_.wires.begin(), params_.wires.end(), utils::wirecomp::first);

      for  (auto i = 0u, j = 0u; i < tp1_->totDims().size(); ++i) {
        if ((j < params_.wires.size()) && (i == params_.wires.at(j).first)) {
          ++j;
        } else {
          params_.dimRepls1.at(i) = i;
        }
      }

      params_.dimRepls2 = std::vector<qtnh::tidx_tup_st>(tp2_->totDims().size(), UINT16_MAX);
      std::sort(params_.wires.begin(), params_.wires.end(), utils::wirecomp::second);

      std::vector<qtnh::tidx_tup_st> from_dims(params_.wires.size());
      for (auto i = 0u; i < params_.wires.size(); ++i) {
        from_dims.at(i) = params_.wires.at(i).first;
      }

      for (auto i = 0u, j = 0u, k = 0u; i < tp2_->totDims().size(); ++i) {
        if ((j < params_.wires.size()) && (i == params_.wires.at(j).second)) {
          ++j;
        } else if (k < params_.wires.size()) {
          params_.dimRepls2.at(i) = from_dims.at(k++);
        }
      }

      // * Might need to retain this when function completes. 
      params_.useDefRepls = false;
    }

    // * Potentially expensive conversion. 
    PairContractor<DenseTensor, DenseTensor> dcon(
      std::move(tp1_), 
      Tensor::convert<DenseTensor>(std::move(tp2_)), 
      params_
    );

    auto tp_res = dcon.contract();
    params_ = dcon.params();

    return tp_res;
  }

  qtnh::tptr _contract_dense_diag(qtnh::tptr tp1, qtnh::tptr tp2, ConParams& params) {
    auto ws = params.wires;

    auto ndis1 = tp1->disDims().size();
    //auto nloc1 = tp1->locDims().size();
    auto ndis_out2 = tp2->disDims().size() / 2;
    //auto nloc_out2 = tp2->locDims().size() / 2;

    // STEP 1: Permute distributed contracted dims. 
    std::vector<qtnh::tidx_tup_st> ptup1(tp1->totDims().size());
    std::vector<qtnh::tidx_tup_st> ptup2(tp2->totDims().size());
    std::iota(ptup1.begin(), ptup1.end(), 0);
    std::iota(ptup2.begin(), ptup2.end(), 0);

    std::sort(ws.begin(), ws.end(), utils::wirecomp::second);

    std::size_t ndis_cons = 0;
    for (auto w : ws) {
      if (w.first < ndis1) {
        if (w.first < ndis1) ptup1.at(w.first) = ndis1 - ndis_cons - 1;
        for (qtnh::tidx_tup_st i = w.first + 1; i < ndis1; ++i) {
          if (ptup1.at(i) < ndis1 - ndis_cons) --ptup1.at(i);
        }

        // Wires are sorted by second, so this is guaranteed to update all previous values. 
        if (w.second < ndis_out2) ptup2.at(w.second) = 0;
        for (qtnh::tidx_tup_st i = w.second; i > 0; --i) {
          ++ptup2.at(i - 1);
        }

        ++ndis_cons;
      }
    }

    tp1 = Tensor::permute(std::move(tp1), ptup1);
    tp2 = SymmTensorBase::permuteIO(std::move(tp2), ptup2);

    // Update dimension replacements after permutations. 
    auto dim_repls1_p = utils::permute_vec(params.dimRepls1, ptup1);
    auto dim_repls2_p = utils::permute_vec(params.dimRepls2, ptup2);


    // STEP 2: Align by broadcast. 
    auto dis_dims1 = tp1->disDims();
    auto dis_dims_out2 = utils::split_dims(tp2->disDims(), ndis_out2).first;
    dis_dims1.erase(dis_dims1.end() - ndis_cons, dis_dims1.end());
    dis_dims_out2.erase(dis_dims_out2.begin(), dis_dims_out2.begin() + ndis_cons);

    auto align_str = (qtnh::uint)utils::dims_to_size(dis_dims_out2);
    auto align_cyc = (qtnh::uint)utils::dims_to_size(dis_dims1);
    auto align_off = std::min(tp1->bc().params().off, tp2->bc().params().off);
    tp1 = Tensor::rebcast(std::move(tp1), { align_str, 1, align_off });
    tp2 = Tensor::rebcast(std::move(tp2), { 1, align_cyc, align_off });

    return tp1;
  }
}
