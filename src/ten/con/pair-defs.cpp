#include <algorithm>
#include <iostream>
#include <numeric>

#include "blas/wrappers.hpp"
#include "ten/con/pair-defs.hpp"
#include "ten/util/indexing.hpp"
#include "ten/util/ops.hpp"
#include "ten/util/ptuple.hpp"
#include "ten/util/utils.hpp"
#include "ten/util/vector.hpp"

namespace qtnh {
  void _local_contraction(Tensor* tp1, Tensor* tp2, DenseTensor* tp3, 
                          TIndexing ti1, TIndexing ti2, TIndexing ti3) {
    auto it3 = ti3.keep("local").num("local").begin();
    for (auto idxs1 : ti1.tup("local")) {
      for (auto idxs2 : ti2.tup("local")) {
        qtnh::tel el3 = 0.0;

        #ifdef DEBUG
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

  template<> qtnh::tptr PairContractor<DenseTensor, DenseTensor>::contract_gemm() {
    #ifdef DEBUG
    if (utils::is_root())
      std::cout << "STARTING DENSE-DENSE CONTRACTION (GEMM METHOD)\n";
    #endif

    auto ws = params_.wires;
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

    // Calculate grid size. 
    auto dis_size_1 = tp1_->disSize() / dis_ws_size;
    auto dis_size_2 = tp2_->disSize() / dis_ws_size;

    auto use_bt = dis_ws_size > dis_size_1 && dis_ws_size > dis_size_2;
    auto grows = int(std::max(dis_size_1, dis_ws_size));
    auto gcols = int(std::max(dis_size_2, dis_ws_size));
    if (use_bt) {
      grows = int(std::max(dis_size_1, dis_size_2));
      gcols = int(dis_ws_size);
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
    ig2.reorder({ "rd", "cd", "cb", "rb" });
    if (use_bt) ig2.reorder({ "cd", "rd", "rb", "cb" });

    ptup1 = ig1.ptup() * ptup1;
    ptup2 = ig2.ptup() * ptup2;

    // Align and permute tensors. 
    auto offset = std::min(tp1_->bc().params().off, tp2_->bc().params().off);
    tp1_ = Tensor::cast<DenseTensor>(Tensor::rebcast(std::move(tp1_), { 1, 1, offset }));
    tp2_ = Tensor::cast<DenseTensor>(Tensor::rebcast(std::move(tp2_), { 1, 1, offset }));

    tp1_ = Tensor::cast<DenseTensor>(Tensor::permute(std::move(tp1_), ptup1.toTar().tup()));
    tp2_ = Tensor::cast<DenseTensor>(Tensor::permute(std::move(tp2_), ptup2.toTar().tup()));

    // Calculate matrix parameters. 
    std::array<int, 4> sizes1, sizes2;
    auto to_size = [](auto dims) { return int(utils::dims_to_size(dims)); };
    std::transform(dims1.begin(), dims1.end(), sizes1.begin(), to_size);
    std::transform(dims2.begin(), dims2.end(), sizes2.begin(), to_size);

    auto md = sizes1.at(0), nd = sizes2.at(1), kd = sizes1.at(1);
    auto ml = sizes1.at(2), nl = sizes2.at(3), kl = sizes1.at(3);

    using namespace lalg;
    auto pg1 = ProcGrid(md, kd, offset);
    auto pg2 = use_bt ? ProcGrid(nd, kd, offset) : ProcGrid(kd, nd, offset);
    auto pg3 = ProcGrid(md, nd, offset);
    auto pg_all = ProcGrid(grows, gcols, offset);

    // Create matrices and move them to larger grid. 
    auto m1 = BlockCyclicMatrix(pg1, { md * ml, kd * kl }, { ml, kl }, tp1_->extractEls());
    auto m2 = use_bt
      ? BlockCyclicMatrix(pg2, { nd * nl, kd * kl }, { nl, kl }, tp2_->extractEls())
      : BlockCyclicMatrix(pg2, { kd * kl, nd * nl }, { kl, nl }, tp2_->extractEls());

    m1 = std::move(m1).toGrid(pg_all);
    m2 = std::move(m2).toGrid(pg_all);
    
    // Calculate matrix product. 
    auto m3 = PZGEMM(std::move(m1), std::move(m2), false, use_bt);
    m3 = std::move(m3).toGrid(pg3);

    auto dis_dims3 = utils::concat_dims(dims1.at(0), dims2.at(1));
    auto loc_dims3 = utils::concat_dims(dims2.at(3), dims1.at(2)); // column-major

    tptr tp3 = DenseTensor::make(tp1_->bc().env(), dis_dims3, loc_dims3, 
                                 m3.extractEls(), { 1, 1, offset});

    // Permute back to row-major. 
    PTupleSrc ptup3(tp3->totDims().size());
    auto gs3 = utils::split_vec_rel(ptup3.tup(), dims1.at(0).size(), dims2.at(1).size(), 
                                    dims1.at(2).size(), dims2.at(3).size());

    IndexGroup ig3({ "rd", "cd", "rb", "cb" }, utils::arr_to_vec(gs3));
    ig3.reorder({ "rd", "cd", "cb", "rb" });

    // Apply dimension replacements. 
    auto dim_repls = utils::concat_vecs(params_.dimRepls1, params_.dimRepls2);
    PTupleSrc ptup_repls(dim_repls.size());

    auto gs_repls = utils::split_vec_rel(ptup_repls.tup(), ndis1, nloc1, ndis2);
    IndexGroup ig_repls({ "d1", "l1", "d2", "l2" }, utils::arr_to_vec(gs_repls));
    ig_repls.reorder({ "d1", "d2", "l1", "l2" });
    dim_repls = ig_repls.ptup().apply(dim_repls);
    tup_t remap(tp3->totDims().size());

    for (auto i = 0UL, j = 0UL; i < dim_repls.size(); ++i) {
      if (dim_repls.at(i) < qtnh::X) {
        remap.at(i - j) = dim_repls.at(i);
      } else {
        ++j;
      }
    }

    auto ptup_final = PTupleTar(remap) * ig3.ptup().inv().toTar();
    tp3 = Tensor::permute(std::move(tp3), ptup_final.tup());

    return tp3;
  }

  template<> qtnh::tptr PairContractor<DenseTensor, DenseTensor>::contract_direct() {
    #ifdef DEBUG
      if (utils::is_root())
        std::cout << "STARTING DENSE-DENSE CONTRACTION (DIRECT METHOD)\n";
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

  template<> qtnh::tptr PairContractor<DenseTensor, DenseTensor>::contract() {
    auto n_dis_ws = 0UL;
    for (auto [w1, w2] : params_.wires) {
      if (w1 < tp1_->disDims().size()) n_dis_ws++;
    }

    // Calculate default index replacements. 
    if (params_.useDefRepls) {
      params_.dimRepls1 = std::vector<qtnh::tidx_tup_st>(tp1_->totDims().size(), qtnh::X);
      std::sort(params_.wires.begin(), params_.wires.end(), utils::wirecomp::first);

      for  (auto i = 0u, j = 0u; i < tp1_->totDims().size(); ++i) {
        if ((j < params_.wires.size()) && (i == params_.wires.at(j).first)) {
          ++j;
        } else {
          params_.dimRepls1.at(i) = i - j;
          if (i >= tp1_->disDims().size()) {
            params_.dimRepls1.at(i) += (tp2_->disDims().size() - n_dis_ws);
          }
        }
      }

      params_.dimRepls2 = std::vector<qtnh::tidx_tup_st>(tp2_->totDims().size(), qtnh::X);
      std::sort(params_.wires.begin(), params_.wires.end(), utils::wirecomp::second);

      for  (auto i = 0u, j = 0u; i < tp2_->totDims().size(); ++i) {
        if ((j < params_.wires.size()) && (i == params_.wires.at(j).second)) {
          ++j;
        } else {
          params_.dimRepls2.at(i) = tp1_->disDims().size() - n_dis_ws + i - j;
          if (i >= tp2_->disDims().size()) {
            params_.dimRepls2.at(i) = tp1_->totDims().size() - params_.wires.size() + i - j;
          }
        }
      }
    }

    #ifdef DEBUG
      if (utils::is_root()) {
        std::cout << "T1 dimension replacements: " << params_.dimRepls1 << "\n";
        std::cout << "T2 dimension replacements: " << params_.dimRepls2 << "\n";
      }
    #endif

    #ifdef CON_GEMM
      return contract_gemm();
    #else
      return contract_direct();
    #endif
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
      params_.dimRepls1 = std::vector<qtnh::tidx_tup_st>(tp1_->totDims().size(), qtnh::X);
      std::sort(params_.wires.begin(), params_.wires.end(), utils::wirecomp::first);

      for  (auto i = 0u, j = 0u; i < tp1_->totDims().size(); ++i) {
        if ((j < params_.wires.size()) && (i == params_.wires.at(j).first)) {
          ++j;
        } else {
          params_.dimRepls1.at(i) = i;
        }
      }

      params_.dimRepls2 = std::vector<qtnh::tidx_tup_st>(tp2_->totDims().size(), qtnh::X);
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
