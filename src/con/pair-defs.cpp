#include <algorithm>
#include <iostream>
#include <numeric>

#include "con/pair-defs.hpp"
#include "core/utils.hpp"
#include "tensor/indexing.hpp"


namespace qtnh {
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
    auto align_off = std::min(tp1_->bc().off, tp2_->bc().off);
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
    if (tp1_->bc().active && tp2_->bc().active) {
      loc_size = utils::dims_to_size(ti3.cut("distributed").dims());
    }

    auto els = std::vector<qtnh::tel>(loc_size);
    DenseTensor t3 { 
      tp1_->bc().env, 
      ti3.cut("local").dims(), 
      ti3.keep("local").dims(), 
      std::move(els), 
      { 1, 1, align_off } 
    };

    ti1 = ti1.cut("distributed").cut("reduced");
    ti2 = ti2.cut("distributed").cut("reduced");

    if (t3.bc().active) {
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

            el3 += (*tp1_)[*(it1++)] * (*tp2_)[*(it2++)];

            #ifdef DEBUG
              if (it1 != it1.end() && it2 != it2.end()) std::cout << " + ";
            #endif
          }

          t3[*(it3++)] = el3;

          #ifdef DEBUG
            std::cout << " = " << el3  << std::endl;
          #endif
        }
      }

      // STEP 4: All-reduce distributed wires. 
      auto dis_idxs = utils::i_to_idxs(t3.bc().group_id, t3.disDims());
      for (auto i = 0u; i < dis_idxs.size(); ++i) {
        if (ti3.ifls().at(i).label == "reduced") dis_idxs.at(i) = 0;
      }
      
      auto colour = utils::idxs_to_i(dis_idxs, t3.disDims());

      // ! Expect MPI memory limit issues. 
      // ! Can be performed multiple times with offset for larger arrays. 
      MPI_Comm allr_comm;
      MPI_Comm_split(t3.bc().group_comm, colour, t3.bc().group_id, &allr_comm);
      MPI_Allreduce(MPI_IN_PLACE, t3.loc_els_.data(), loc_size, MPI_C_DOUBLE_COMPLEX, MPI_SUM, allr_comm);
      MPI_Comm_free(&allr_comm);
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

    return DenseTensor::make(t3.bc().env, new_dis_dims, t3.locDims(), std::move(t3.loc_els_), new_params);
  }

  template<> qtnh::tptr PairContractor<DenseTensor, SymmTensor>::contract() {
    #ifdef DEBUG
      if (utils::is_root())
        std::cout << "STARTING DENSE-SYMM CONTRACTION\n";
    #endif

    std::size_t input_count = 0;
    for (auto w : params_.wires) {
      if (w.second < (tp2_->disDims().size() / 2) || ((w.second >= tp2_->disDims().size()) && (w.second < (tp2_->disDims().size() + tp2_->locDims().size() / 2)))) {
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
    auto nloc1 = tp1->locDims().size();
    auto ndis_out2 = tp2->disDims().size() / 2;
    auto nloc_out2 = tp2->locDims().size() / 2;

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
    auto align_off = std::min(tp1->bc().off, tp2->bc().off);
    tp1 = Tensor::rebcast(std::move(tp1), { align_str, 1, align_off });
    tp2 = Tensor::rebcast(std::move(tp2), { 1, align_cyc, align_off });

    return tp1;
  }
}
