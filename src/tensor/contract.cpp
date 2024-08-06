#include <algorithm>
#include <iostream>
#include <numeric>

#include "core/utils.hpp"
#include "tensor/tensor.hpp"
#include "tensor/dense.hpp"
#include "tensor/symm.hpp"
#include "tensor/indexing.hpp"

namespace qtnh {
  qtnh::tptr _contract_dense(qtnh::tptr t1p, qtnh::tptr t2p, std::vector<qtnh::wire> ws, bool in_place = false) {
    auto ndis1 = t1p->disDims().size();
    auto nloc1 = t1p->locDims().size();
    auto ndis2 = t2p->disDims().size();
    auto nloc2 = t2p->locDims().size();

    // STEP 1: Permute distributed contracted dims. 
    std::vector<qtnh::tidx_tup_st> ptup1(t1p->totDims().size());
    std::vector<qtnh::tidx_tup_st> ptup2(t2p->totDims().size());
    std::iota(ptup1.begin(), ptup1.end(), 0);
    std::iota(ptup2.begin(), ptup2.end(), 0);

    std::sort(ws.begin(), ws.end(), utils::wirecomp::first);

    std::size_t ndis_cons = 0;
    for (auto w : ws) {
      if (w.first < ndis1) {
        std::size_t i1, i2;
        for (i1 = 0; i1 < ndis1; ++i1) {
          if (ptup1.at(i1) == w.first) break; } 
        for (i2 = 0; i2 < ndis2; ++i2) {
          if (ptup2.at(i2) == w.second) break; }

        ptup1.erase(ptup1.begin() + i1);
        ptup2.erase(ptup2.begin() + i2);

        // Ensure same ordering; -1 is necessary as one element has been erased. 
        ptup1.insert(ptup1.begin() + ndis1 - ndis_cons - 1, w.first);
        ptup2.insert(ptup2.begin(), w.second);

        ndis_cons++;
      }
    }

    t1p = Tensor::permute(std::move(t1p), ptup1);
    t2p = Tensor::permute(std::move(t2p), ptup2);

    // STEP 2: Align by broadcast. 
    auto dis_dims1 = t1p->disDims();
    auto dis_dims2 = t2p->disDims();
    dis_dims1.erase(dis_dims1.end() - ndis_cons, dis_dims1.end());
    dis_dims2.erase(dis_dims2.begin(), dis_dims2.begin() + ndis_cons);

    auto align_str = (qtnh::uint)utils::dims_to_size(dis_dims2);
    auto align_cyc = (qtnh::uint)utils::dims_to_size(dis_dims1);
    auto align_off = std::min(t1p->bc().off, t2p->bc().off);
    t1p = Tensor::rebcast(std::move(t1p), { align_str, 1, align_off });
    t2p = Tensor::rebcast(std::move(t2p), { 1, align_cyc, align_off });

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

    TIndexing ti1(t1p->totDims(), ifls1);
    TIndexing ti2(t2p->totDims(), ifls2);
    auto ti3_loc = TIndexing::app(ti1.keep("local"), ti2.keep("local"));

    std::vector<qtnh::tidx_tup_st> ptup_loc(ti3_loc.dims().size());
    std::iota(ptup_loc.begin(), ptup_loc.end(), 0);

    // TODO: Move this to a separate function. Dense contraction should directly accept permutation tuples. 
    if (in_place) {
      std::sort(ws.begin(), ws.end(), utils::wirecomp::second);
      std::vector<qtnh::wire> ws_loc(ws.begin() + ndis_cons, ws.end());

      auto ti1_loc_size = ti1.keep("local").dims().size();
      for (std::size_t i = 0; (i < ws_loc.size() ) && (ptup_loc.size() > ti1_loc_size); ++i) {
        ptup_loc.at(ti1_loc_size + i) = ws_loc.at(i).first - ndis1;
        for (std::size_t j = 0; j < ti1_loc_size; ++j) {
          if (ptup_loc.at(j) >= ws_loc.at(i).first) ++ptup_loc.at(j);
        }
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
    if (t1p->bc().active && t2p->bc().active) {
      loc_size = utils::dims_to_size(ti3.cut("distributed").dims());
    }

    auto els = std::vector<qtnh::tel>(loc_size);
    DenseTensor t3(t1p->bc().env, ti3.cut("local").dims(), ti3.keep("local").dims(), std::move(els), { 1, 1, align_off });

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

            el3 += (*t1p)[*(it1++)] * (*t2p)[*(it2++)];

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
    
    for (std::size_t i = 1; i <= ndis_cons; ++i) {
      ptup3.at(ndis1 - i) = t3.disDims().size() - i;
    }
    for (auto i = ndis1; i < t3.disDims().size(); ++i) {
      ptup3.at(i) -= ndis_cons;
    }

    if (in_place) {
      std::sort(ws.begin(), ws.end(), utils::wirecomp::second);
      std::vector<qtnh::wire> ws_dis(ws.begin(), ws.begin() + ndis_cons);

      for (std::size_t i = 0; (i < ws_dis.size()) && (ndis2 - ndis_cons > 0); ++i) {
        ptup3.at(ndis1 + i) = ws_dis.at(i).first;
        for (std::size_t j = 0; j < ndis1 - ndis_cons; ++j) {
          if (ptup3.at(j) >= ws_dis.at(i).first) ++ptup3.at(j);
        }
      }
    }

    t3._permute_internal(&t3, ptup3);

    // Last n distributed indices are virtual. 
    auto [new_dis_dims, virtual_dims] = utils::split_dims(t3.disDims(), t3.disDims().size() - ndis_cons);
    BcParams new_params(utils::dims_to_size(virtual_dims), 1, align_off);

    return DenseTensor::make(t3.bc().env, new_dis_dims, t3.locDims(), std::move(t3.loc_els_), new_params);
  }

  // TODO: Implement this in a separate file using enums
  tptr Tensor::contract(tptr t1u, tptr t2u, const std::vector<qtnh::wire>& ws) {
    // Validate contraction dimensions
    for (auto& w : ws) {
      if (t1u->totDims().at(w.first) != t2u->totDims().at(w.second)) {
        throw std::invalid_argument("Incompatible contraction dimensions.");
      }
      if ((w.first < t1u->disDims().size()) != (w.second < t2u->disDims().size())) {
        throw std::invalid_argument("Attempted to contract distributed and local dimensions.");
      }
    }

    if (t2u->cast<SymmTensorBase>() != nullptr) {
      std::cout << "Symmetric contraction\n";
      t2u = Tensor::cast<SymmTensorBase>(std::move(t2u));
      return _contract_dense(std::move(t1u), std::move(t2u), ws, true);
    }

    std::cout << "Dense contraction\n";
    return _contract_dense(std::move(t1u), std::move(t2u), ws);
  }
}
