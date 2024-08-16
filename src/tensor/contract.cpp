#include <algorithm>
#include <iostream>
#include <numeric>

#include "core/utils.hpp"
#include "tensor/tensor.hpp"
#include "tensor/dense.hpp"
#include "tensor/symm.hpp"
#include "tensor/indexing.hpp"

namespace qtnh {
  qtnh::tptr _contract_dense(qtnh::tptr tp1, qtnh::tptr tp2, ConParams& params) {
    auto ws = params.wires;

    auto ndis1 = tp1->disDims().size();
    auto nloc1 = tp1->locDims().size();
    auto ndis2 = tp2->disDims().size();
    auto nloc2 = tp2->locDims().size();

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
        if (w.second < ndis2) ptup2.at(w.second) = 0;
        for (qtnh::tidx_tup_st i = w.second; i > 0; --i) {
          ++ptup2.at(i - 1);
        }

        ++ndis_cons;
      }
    }

    tp1 = Tensor::permute(std::move(tp1), ptup1);
    tp2 = Tensor::permute(std::move(tp2), ptup2);

    // Update dimension replacements after permutations. 
    auto dim_repls1_p = utils::permute_vec(params.dimRepls1, ptup1);
    auto dim_repls2_p = utils::permute_vec(params.dimRepls2, ptup2);

    // STEP 2: Align by broadcast. 
    auto dis_dims1 = tp1->disDims();
    auto dis_dims2 = tp2->disDims();
    dis_dims1.erase(dis_dims1.end() - ndis_cons, dis_dims1.end());
    dis_dims2.erase(dis_dims2.begin(), dis_dims2.begin() + ndis_cons);

    auto align_str = (qtnh::uint)utils::dims_to_size(dis_dims2);
    auto align_cyc = (qtnh::uint)utils::dims_to_size(dis_dims1);
    auto align_off = std::min(tp1->bc().off, tp2->bc().off);
    tp1 = Tensor::rebcast(std::move(tp1), { align_str, 1, align_off });
    tp2 = Tensor::rebcast(std::move(tp2), { 1, align_cyc, align_off });

    // STEP 3: Contract local wires. 
    std::vector<TIFlag> ifls1(ndis1, { "distributed", 0 });
    std::vector<TIFlag> ifls2(ndis2, { "distributed", 0 });
    ifls1.insert(ifls1.end(), nloc1, { "local", 0 });
    ifls2.insert(ifls2.end(), nloc2, { "local", 0 });

    for (std::size_t i = 0; i < ws.size(); ++i) {
      if (ws.at(i).first < ndis1) {
        ifls1.at(ws.at(i).first) = { "reduced", (int)i };
        ifls2.at(ws.at(i).second) = { "reduced", (int)i };
      } else {
        ifls1.at(ws.at(i).first) = { "closed", (int)i };
        ifls2.at(ws.at(i).second) = { "closed", (int)i };
      }
    }

    TIndexing ti1(tp1->totDims(), ifls1);
    TIndexing ti2(tp2->totDims(), ifls2);
    auto ti3_dis = TIndexing::app(ti1.keep("distributed"), ti2.keep("distributed"));
    auto ti3_loc = TIndexing::app(ti1.keep("local"), ti2.keep("local"));

    std::vector<qtnh::tidx_tup_st> ptup_loc(ti3_loc.dims().size());
    std::iota(ptup_loc.begin(), ptup_loc.end(), 0);

    for (std::size_t i = tp1->disDims().size(), j = i; i < tp1->totDims().size(); ++i) {
      if (ifls1.at(i).label != "closed") {
        ptup_loc.at(i - j) = dim_repls1_p.at(i) - ti3_dis.dims().size();
      } else {
        ++j;
      }
    }

    auto split = ti1.keep("local").dims().size();
    for (std::size_t i = tp2->disDims().size(), j = i; i < tp2->totDims().size(); ++i) {
      if (ifls2.at(i).label != "closed") {
        ptup_loc.at(split + i - j) = dim_repls2_p.at(i) - ti3_dis.dims().size();
      } else {
        ++j;
      }
    }

    qtnh::tidx_tup dims_ti3_loc(ti3_loc.dims().size());
    std::vector<TIFlag> ifls_ti3_loc(ti3_loc.ifls().size());
    for (std::size_t i = 0; i < ti3_loc.dims().size(); ++i) {
      dims_ti3_loc.at(ptup_loc.at(i)) = ti3_loc.dims().at(i);
      ifls_ti3_loc.at(ptup_loc.at(i)) = { "local", (int)i };
    }

    auto ti3 = TIndexing::app(
      ti1.keep("distributed"), 
      ti1.keep("reduced"), 
      ti2.keep("distributed"), 
      TIndexing(dims_ti3_loc, ifls_ti3_loc)
    );

    std::size_t loc_size = 0;
    if (tp1->bc().active && tp2->bc().active) {
      loc_size = utils::dims_to_size(ti3.cut("distributed").dims());
    }

    auto els = std::vector<qtnh::tel>(loc_size);
    DenseTensor t3(tp1->bc().env, ti3.cut("local").dims(), ti3.keep("local").dims(), std::move(els), { 1, 1, align_off });

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

            el3 += (*tp1)[*(it1++)] * (*tp2)[*(it2++)];

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
      for (std::size_t i = 0; i < dis_idxs.size(); ++i) {
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
    
    for (std::size_t i = 0; i < ndis1 - ndis_cons; ++i) {
      ptup3.at(i) = dim_repls1_p.at(i);
    } 
    for (std::size_t i = 1; i <= ndis_cons; ++i) {
      ptup3.at(ndis1 - i) = t3.disDims().size() - i;
    }
    for (std::size_t i = 0; i < ndis2 - ndis_cons; ++i) {
      ptup3.at(ndis1 + i) = dim_repls2_p.at(ndis_cons + i);
    }

    t3._permute_internal(&t3, ptup3);

    // Last n distributed indices are virtual. 
    auto [new_dis_dims, virtual_dims] = utils::split_dims(t3.disDims(), t3.disDims().size() - ndis_cons);
    BcParams new_params(utils::dims_to_size(virtual_dims), 1, align_off);

    return DenseTensor::make(t3.bc().env, new_dis_dims, t3.locDims(), std::move(t3.loc_els_), new_params);
  }

  // TODO: Implement this in a separate file using enums
  tptr Tensor::contract(tptr tp1, tptr tp2, ConParams& params) {
    std::size_t n_dis_ws = 0;

    // Validate contraction dimensions
    for (auto& w : params.wires) {
      if (tp1->totDims().at(w.first) != tp2->totDims().at(w.second)) {
        throw std::invalid_argument("Incompatible contraction dimensions.");
      }
      if ((w.first < tp1->disDims().size()) != (w.second < tp2->disDims().size())) {
        throw std::invalid_argument("Attempted to contract distributed and local dimensions.");
      }

      if (w.first < tp1->disDims().size()) ++n_dis_ws;
    }

    // Create default index replacements. 
    if (params.useDefRepls) {
      params.dimRepls1 = std::vector<qtnh::tidx_tup_st>(tp1->totDims().size(), UINT16_MAX);
      std::sort(params.wires.begin(), params.wires.end(), utils::wirecomp::first);

      for  (std::size_t i = 0, j = 0; i < tp1->totDims().size(); ++i) {
        if ((j < params.wires.size()) && (i == params.wires.at(j).first)) {
          ++j;
        } else {
          params.dimRepls1.at(i) = i - j;
          if (i >= tp1->disDims().size()) {
            params.dimRepls1.at(i) += (tp2->disDims().size() - n_dis_ws);
          }
        }
      }

      params.dimRepls2 = std::vector<qtnh::tidx_tup_st>(tp2->totDims().size(), UINT16_MAX);
      std::sort(params.wires.begin(), params.wires.end(), utils::wirecomp::second);

      for  (std::size_t i = 0, j = 0; i < tp2->totDims().size(); ++i) {
        if ((j < params.wires.size()) && (i == params.wires.at(j).second)) {
          ++j;
        } else {
          params.dimRepls2.at(i) = tp1->disDims().size() - n_dis_ws + i - j;
          if (i >= tp2->disDims().size()) {
            params.dimRepls2.at(i) = tp1->totDims().size() - params.wires.size() + i - j;
          }
        }
      }
    }

    if (tp2->cast<SymmTensorBase>() != nullptr) {
      auto tp2_symm = Tensor::cast<SymmTensorBase>(std::move(tp2));

      std::size_t input_count = 0;
      for (auto w : params.wires) {
        if (w.second < tp2_symm->disInDims().size() || ((w.second >= tp2_symm->disDims().size()) && (w.second < (tp2_symm->disDims().size() + tp2_symm->locInDims().size())))) {
          ++input_count;
        }
      }

      auto dis_imbal = (int)tp2_symm->disInDims().size() - (int)tp2_symm->disOutDims().size();

      if ((dis_imbal == 0) && (input_count == tp2_symm->disInDims().size() + tp2_symm->locInDims().size())) {
        params.dimRepls1 = std::vector<qtnh::tidx_tup_st>(tp1->totDims().size(), UINT16_MAX);
        std::sort(params.wires.begin(), params.wires.end(), utils::wirecomp::first);

        for  (std::size_t i = 0, j = 0; i < tp1->totDims().size(); ++i) {
          if ((j < params.wires.size()) && (i == params.wires.at(j).first)) {
            ++j;
          } else {
            params.dimRepls1.at(i) = i;
          }
        }

        params.dimRepls2 = std::vector<qtnh::tidx_tup_st>(tp2_symm->totDims().size(), UINT16_MAX);
        std::sort(params.wires.begin(), params.wires.end(), utils::wirecomp::second);

        std::vector<qtnh::tidx_tup_st> from_dims(params.wires.size());
        for (std::size_t i = 0; i < params.wires.size(); ++i) {
          from_dims.at(i) = params.wires.at(i).first;
        }

        for (std::size_t i = 0, j = 0, k = 0; i < tp2_symm->totDims().size(); ++i) {
          if ((j < params.wires.size()) && (i == params.wires.at(j).second)) {
            ++j;
          } else if (k < params.wires.size()) {
            params.dimRepls2.at(i) = from_dims.at(k++);
          }
        }
      }

      #ifdef DEBUG
        using namespace ops;
        if (utils::is_root()) {
          std::cout << "Starting symmetric contraction\n";
          std::cout << "T1 dimension replacements: " << params.dimRepls1 << "\n";
          std::cout << "T2 dimension replacements: " << params.dimRepls2 << "\n";
        }
      #endif

      return _contract_dense(std::move(tp1), std::move(tp2_symm), params);
    }

    #ifdef DEBUG
      using namespace ops;
      if (utils::is_root()) {
        std::cout << "Starting dense contraction\n";
        std::cout << "T1 dimension replacements: " << params.dimRepls1 << "\n";
        std::cout << "T2 dimension replacements: " << params.dimRepls2 << "\n";
      }
    #endif

    return _contract_dense(std::move(tp1), std::move(tp2), params);
  }
}
