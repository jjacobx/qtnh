#include <cassert>
#include <functional>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <stdexcept>

#include "core/utils.hpp"
#include "tensor/dense.hpp"
#include "tensor/indexing.hpp"
#include "tensor/special.hpp"

namespace qtnh {
  DenseTensor::DenseTensor(std::vector<qtnh::tel> els)
    : WritableTensor(), loc_els(els) {}

  qtnh::tel DenseTensor::operator[](const qtnh::tidx_tup &loc_idxs) const {
    auto i = utils::idxs_to_i(loc_idxs, loc_dims);
    return loc_els.at(i);
  }

  qtnh::tel& DenseTensor::operator[](const qtnh::tidx_tup& loc_idxs) {
    auto i = utils::idxs_to_i(loc_idxs, loc_dims);
    return loc_els.at(i);
  }

  TIndexing _get_indexing(Tensor* tp, const std::vector<qtnh::wire>& ws, bool second) {
    auto loc_dims = tp->getLocDims();
    auto dist_dims = tp->getDistDims();

    qtnh::tifl_tup ifls(loc_dims.size(), { TIdxT::open, 0 });

    qtnh::tidx_tup_st tag = 0;
    for (auto w : ws) {
      ifls.at((second ? w.second : w.first) - dist_dims.size()) = qtnh::tifl{ TIdxT::closed, tag++ };
    }

    return TIndexing(loc_dims, ifls);
  }

  qtnh::tidx_tup _concat_dims(qtnh::tidx_tup dist_dims1, qtnh::tidx_tup dist_dims2, qtnh::tidx_tup loc_dims) {
    auto dist_dims = dist_dims1;
    dist_dims.insert(dist_dims.end(), dist_dims2.begin(), dist_dims2.end());

    auto dims = dist_dims;
    dims.insert(dims.end(), loc_dims.begin(), loc_dims.end());

    return dims;
  }

  void _set_els(Tensor* t1p, Tensor* t2p, DenseTensor* t3p, TIndexing ti1, TIndexing ti2, TIndexing ti3, qtnh::tidx_tup_st ws_size) {
    auto it = ti3.begin();
    for (auto idxs1 : ti1) {
      for (auto idxs2 : ti2) {
        qtnh::tidx_tup_st tag = 0;
        qtnh::tel el3 = 0.0;

        #ifdef DEBUG
          using namespace qtnh::ops;
          std::cout << "t3[" << *it << "] = ";
        #endif

        while(t3p->isActive()) {
          auto el1 = (*t1p)[idxs1];
          auto el2 = (*t2p)[idxs2];
          el3 += el1 * el2;

          #ifdef DEBUG
            std::cout << "t1[" << idxs1 << "] * t2[" << idxs2 << "]";
          #endif

          if (ti1.isLast(idxs1, TIdxT::closed, tag) && ti2.isLast(idxs2, TIdxT::closed, tag)) {
            tag++;
            if (tag >= ws_size) {
              (*t3p)[*it] = el3;

              #ifdef DEBUG
                std::cout << " = " << el3 << std::endl;
              #endif

              break;
            }     
          }

          #ifdef DEBUG
            std::cout << " + ";
          #endif

          ti1.next(idxs1, TIdxT::closed, tag);
          ti2.next(idxs2, TIdxT::closed, tag);
        }

        for (qtnh::tidx_tup_st t = 0; t < ws_size; ++t) {
          ti1.reset(idxs1, TIdxT::closed, t);
        }

        ++it;
      }
    }
  }

  void _local_swap(DenseTensor* tp, qtnh::tidx_tup_st loc_idx1, qtnh::tidx_tup_st loc_idx2) {
    auto loc_dims = tp->getLocDims();
    qtnh::tifl_tup ifls(loc_dims.size(), { TIdxT::open, 0 });
    ifls.at(loc_idx1) = ifls.at(loc_idx2) = { TIdxT::closed, 0 };
    TIndexing ti(loc_dims, ifls);

    for (auto idxs : ti) {
      auto idxs1 = idxs;
      auto idxs2 = idxs;

      // Swaps only invoked n * (n - 1) / 2 times, instead of n * n
      for (qtnh::tidx i = 0; i < loc_dims.at(loc_idx1) - 1; ++i) {
        idxs1.at(loc_idx1) = idxs2.at(loc_idx1) = i;
        for (qtnh::tidx j = i + 1; j < loc_dims.at(loc_idx2); ++j) {
          idxs1.at(loc_idx2) = idxs2.at(loc_idx1) = j;
          std::swap((*tp)[idxs1], (*tp)[idxs2]);
        }
      }
    }

    return;
  }

  SDenseTensor::SDenseTensor(const QTNHEnv& env, qtnh::tidx_tup dims, std::vector<qtnh::tel> els)
    : SDenseTensor(env, dims, els, DEF_STENSOR_BCAST) {}

  SDenseTensor::SDenseTensor(const QTNHEnv& env, qtnh::tidx_tup dims, std::vector<qtnh::tel> els, bool bcast)
    : Tensor(env), DenseTensor(els), SharedTensor(dims) {
    if (loc_els.size() != getLocSize()) {
      throw std::invalid_argument("Invalid length of elements.");
    }

    #ifdef DEBUG
      MPI_Barrier(MPI_COMM_WORLD);
    #endif

    // Use broadcast flag to ensure elements are the same on all ranks
    if (bcast) {
      loc_els.resize(getSize());
      MPI_Bcast(loc_els.data(), getSize(), MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    }
    
  }

  void SDenseTensor::setEl(const qtnh::tidx_tup& idxs, qtnh::tel el) {
    setLocEl(idxs, el);
    return;
  }

  void SDenseTensor::setLocEl(const qtnh::tidx_tup& loc_idxs, qtnh::tel el) {
    if (!active) return;
    (*this)[loc_idxs] = el;
    return;
  }

  void SDenseTensor::swap(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) {
    if (dims.at(idx1) != dims.at(idx2)) {
      throw std::runtime_error("Asymmetric swaps are not allowed");
    }

    if (idx1 == idx2) return;

    _local_swap(this, idx1, idx2);
    return;
  }

  Tensor* SDenseTensor::contract_disp(Tensor* tp, const std::vector<qtnh::wire>& ws) {
    return tp->contract(this, ws);
  }

  Tensor* SDenseTensor::contract(ConvertTensor* tp, const std::vector<qtnh::wire>& ws) {
    return tp->contract(this, utils::invert_wires(ws));
  }

  Tensor* SDenseTensor::contract(SDenseTensor* tp, const std::vector<qtnh::wire>& ws) {
    #ifdef DEBUG
      std::cout << "Contracting SDense with SDense" << std::endl;
    #endif

    TIndexing ti1 = _get_indexing(this, ws, 0);
    TIndexing ti2 = _get_indexing(tp, ws, 1);
    TIndexing ti3 = TIndexing::app(ti1.cut(TIdxT::closed), ti2.cut(TIdxT::closed));

    auto dims3 = ti3.getDims();
    auto nloc = utils::dims_to_size(dims3);
    std::vector<qtnh::tel> els3(nloc, 0.0);

    auto t3p = new SDenseTensor(this->env, dims3, els3);

    _set_els(this, tp, t3p, ti1, ti2, ti3, ws.size());

    return t3p;
  }

  Tensor* SDenseTensor::contract(DDenseTensor* tp, const std::vector<qtnh::wire>& ws) {
    #ifdef DEBUG
      std::cout << "Contracting DDense with SDense" << std::endl;
    #endif

    auto ti1 = _get_indexing(this, ws, 0);
    auto ti2 = _get_indexing(tp, ws, 1);
    auto ti3 = TIndexing::app(ti1.cut(TIdxT::closed), ti2.cut(TIdxT::closed));

    std::size_t nloc = utils::dims_to_size(ti3.getDims());
    std::vector<qtnh::tel> els3(nloc, 0.0);

    auto dims3 = _concat_dims(dist_dims, tp->getDistDims(), ti3.getDims());
    auto nidx = tp->getDistDims().size();

    auto t3p = new DDenseTensor(this->env, dims3, els3, nidx);

    _set_els(this, tp, t3p, ti1, ti2, ti3, ws.size());

    return t3p;
  }

  DDenseTensor* SDenseTensor::distribute(tidx_tup_st nidx) {
    qtnh::tidx_tup loc_dims(dims.begin() + nidx, dims.end());
    qtnh::tidx_tup dist_dims(dims.begin(), dims.begin() + nidx);

    auto nloc = utils::dims_to_size(loc_dims);
    auto ndist = utils::dims_to_size(dist_dims);

    if (env.num_processes < ndist) {
      throw std::runtime_error("Not enough ranks to distribute provided indices");
    }

    auto els = loc_els;
    if (env.proc_id < ndist) {
      els.erase(els.begin(), els.begin() + env.proc_id * nloc);
      els.erase(els.begin() + nloc, els.end());
    } else {
      els.clear();
    }

    auto dt = new DDenseTensor(env, dims, els, nidx);
    return dt;
  }

  DDenseTensor::DDenseTensor(const QTNHEnv& env, qtnh::tidx_tup dims, std::vector<qtnh::tel> els, qtnh::tidx_tup_st n_dist_idxs)
    : Tensor(env, utils::split_dims(dims, n_dist_idxs).second, utils::split_dims(dims, n_dist_idxs).first), DenseTensor(els) {
    if (env.proc_id >= getDistSize()) {
      active = false;
    }
 
    if (active && loc_els.size() != getLocSize()) {
      throw std::invalid_argument("Invalid length of elements.");
    }
  }

  std::optional<qtnh::tel> DDenseTensor::getEl(const qtnh::tidx_tup& idxs) const {
    qtnh::tidx_tup loc_idxs(idxs.begin(), idxs.begin() + dist_dims.size());
    qtnh::tidx_tup dist_idxs(idxs.begin() + dist_dims.size(), idxs.end());
    auto rank = utils::idxs_to_i(dist_idxs, getDistDims());

    if (env.proc_id == rank) {
      return getLocEl(loc_idxs);
    } else {
      return {};
    }
  }

  std::optional<qtnh::tel> DDenseTensor::getLocEl(const qtnh::tidx_tup& loc_idxs) const {
    if (active) {
      return (*this)[loc_idxs];
    } else {
      return {};
    }
  }

  void DDenseTensor::setEl(const qtnh::tidx_tup& glob_idxs, qtnh::tel el) {
    qtnh::tidx_tup loc_idxs(glob_idxs.begin(), glob_idxs.begin() + dist_dims.size());
    qtnh::tidx_tup dist_idxs(glob_idxs.begin() + dist_dims.size(), glob_idxs.end());
    auto rank = utils::idxs_to_i(dist_idxs, getDistDims());

    if (env.proc_id == rank) {
      setLocEl(loc_idxs, el);
    }

    return;
  }

  void DDenseTensor::setLocEl(const qtnh::tidx_tup& loc_idxs, qtnh::tel el) {
    if (!active) return;
    (*this)[loc_idxs] = el;
    return;
  }

  void DDenseTensor::swap(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) {
    if (idx1 > idx2) std::swap(idx1, idx2);

    // Case: asymmetric swap
    if (dims.at(idx1) != dims.at(idx2)) {
      throw std::runtime_error("Asymmetric swaps are not allowed");
    }

    // Case: same-index swap
    if (idx1 == idx2) return;

    // Case: local swap
    if (idx1 >= dist_dims.size()) {
      if (active) _local_swap(this, idx1 - dist_dims.size(), idx2 - dist_dims.size());
      return;
    }

    // Case: mixed local/distributed swap
    if (idx1 < dist_dims.size() && idx2 >= dist_dims.size()) {
      // Splitting communication causes global synchronisation
      // Need to split active and inactive processes before quitting
      MPI_Comm active_group;
      MPI_Comm_split(MPI_COMM_WORLD, active, env.proc_id, &active_group);
      if (!active) return;

      qtnh::tidx_tup trail_dims(dims.begin() + idx2 + 1, dims.end());
      auto block_length = utils::dims_to_size(trail_dims);
      auto stride = dims.at(idx2) * block_length;

      qtnh::tidx_tup mid_loc_dims(dims.begin() + dist_dims.size(), dims.begin() + idx2);
      auto num_blocks = utils::dims_to_size(mid_loc_dims);

      qtnh::tidx_tup mid_dist_dims(dist_dims.begin() + idx1 + 1, dist_dims.end());
      auto dist_stride = utils::dims_to_size(mid_dist_dims);

      auto dist_idxs = utils::i_to_idxs(env.proc_id, dist_dims);
      auto rank_idx = dist_idxs.at(idx1);

      MPI_Datatype strided, restrided;
      MPI_Type_vector(num_blocks, block_length, stride, MPI_C_DOUBLE_COMPLEX, &strided);
      MPI_Type_create_resized(strided, 0, block_length * sizeof(qtnh::tel), &restrided);
      MPI_Type_commit(&restrided);

      MPI_Comm swap_group;
      MPI_Comm_split(active_group, env.proc_id - rank_idx * dist_stride, env.proc_id, &swap_group);

      int swap_rank, swap_size;
      MPI_Comm_rank(swap_group, &swap_rank);
      MPI_Comm_size(swap_group, &swap_size);

      std::vector<qtnh::tel> new_els(loc_els.size());
      for (std::size_t i = 0; i < dims.at(idx1); ++i) {
        // TODO: Consider MPI message size limit
        // * A scatter might already take it into account
        MPI_Scatter(loc_els.data(), 1, restrided, new_els.data() + i * block_length, 1, restrided, i, swap_group);
      }

      // ! new_els should not be copied, and original loc_els should be destroyed
      loc_els = std::move(new_els);
      MPI_Type_free(&restrided);
      return;
    }

    // Case: distributed swap
    if (idx2 < dist_dims.size()) {
      if (!active) return;

      auto target_idxs = utils::i_to_idxs(env.proc_id, dist_dims);
      std::swap(target_idxs.at(idx1), target_idxs.at(idx2));
      auto target = utils::idxs_to_i(target_idxs, dist_dims);

      std::vector<qtnh::tel> new_els(loc_els.size());
      // TODO: Consider MPI message size limit – not a scatter
      MPI_Sendrecv(loc_els.data(), loc_els.size(), MPI_C_DOUBLE_COMPLEX, target, 0, 
                   new_els.data(), new_els.size(), MPI_C_DOUBLE_COMPLEX, target, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      
      // ! new_els should not be copied, and original loc_els should be destroyed
      loc_els = std::move(new_els);
      return;
    }

    // * This should not be reached
    utils::throw_unimplemented();
    return;
  }

  Tensor* DDenseTensor::contract_disp(Tensor* tp, const std::vector<qtnh::wire>& ws) {
    return tp->contract(this, ws);
  }

  Tensor* DDenseTensor::contract(SDenseTensor* tp, const std::vector<qtnh::wire>& ws) {
    #ifdef DEBUG
      std::cout << "Contracting DDense with SDense" << std::endl;
    #endif

    auto ti1 = _get_indexing(this, ws, 0);
    auto ti2 = _get_indexing(tp, ws, 1);
    auto ti3 = TIndexing::app(ti1.cut(TIdxT::closed), ti2.cut(TIdxT::closed));

    auto nloc = utils::dims_to_size(ti3.getDims());
    std::vector<qtnh::tel> els3(nloc, 0.0);

    auto dims3 = _concat_dims(dist_dims, tp->getDistDims(), ti3.getDims());
    auto nidx = dist_dims.size();

    auto t3 = new DDenseTensor(this->env, dims3, els3, nidx);

    _set_els(this, tp, t3, ti1, ti2, ti3, ws.size());

    return t3;
  }

  Tensor* DDenseTensor::contract(ConvertTensor* tp, const std::vector<qtnh::wire>& ws) {
    return tp->contract(this, utils::invert_wires(ws));
  }

  Tensor* DDenseTensor::contract(DDenseTensor* tp, const std::vector<qtnh::wire>& ws) {
    #ifdef DEBUG
      std::cout << "Contracting DDense with DDense" << std::endl;
    #endif

    auto ti1 = _get_indexing(this, ws, 0);
    auto ti2 = _get_indexing(tp, ws, 1);
    auto ti3 = TIndexing::app(ti1.cut(TIdxT::closed), ti2.cut(TIdxT::closed));

    auto nloc = utils::dims_to_size(ti3.getDims());
    std::vector<qtnh::tel> els3(0);

    auto dims3 = _concat_dims(dist_dims, tp->getDistDims(), ti3.getDims());
    auto nidx = dist_dims.size() + tp->getDistDims().size();

    auto n1 = utils::dims_to_size(dist_dims);
    auto n2 = utils::dims_to_size(tp->getDistDims());

    if (env.proc_id < n1 * n2) els3.assign(nloc, 0.0);
    auto t3 = new DDenseTensor(this->env, dims3, els3, nidx);

    this->rep_all(n2);
    tp->rep_each(n1);

    _set_els(this, tp, t3, ti1, ti2, ti3, ws.size());

    return t3;
  }

  void DDenseTensor::scatter(tidx_tup_st n) {
    auto dist_dims2 = qtnh::tidx_tup(loc_dims.begin(), loc_dims.begin() + n);
    auto shift = utils::dims_to_size(dist_dims2);

    if (active && env.proc_id != 0) {
      MPI_Ssend(loc_els.data(), getLocSize(), MPI_C_DOUBLE_COMPLEX, env.proc_id * shift, 0, MPI_COMM_WORLD);
    }

    if (env.proc_id % shift == 0 && env.proc_id < shift * getDistSize() && env.proc_id != 0) {
      loc_els.resize(getLocSize());
      MPI_Recv(loc_els.data(), getLocSize(), MPI_C_DOUBLE_COMPLEX, env.proc_id / shift, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      active = true;
    }

    MPI_Comm scatt_comm;
    MPI_Comm_split(MPI_COMM_WORLD, env.proc_id / shift, env.proc_id, &scatt_comm);
    if (env.proc_id < shift * getDistSize()) {
      auto nnew = getLocSize() / shift;

      // Resize only receiving buffers
      if (!active) loc_els.resize(nnew);

      MPI_Scatter(loc_els.data(), nnew, MPI_C_DOUBLE_COMPLEX, loc_els.data(), nnew, MPI_C_DOUBLE_COMPLEX, 0, scatt_comm);

      // Resize everything
      loc_els.resize(nnew);
      active = true;
    }

    loc_dims.erase(loc_dims.begin(), loc_dims.begin() + n);
    dist_dims.insert(dist_dims.end(), dist_dims2.begin(), dist_dims2.end());

    return;
  }

  void DDenseTensor::gather(tidx_tup_st n) {
    auto loc_dims2 = qtnh::tidx_tup(dist_dims.end() - n, dist_dims.end());
    auto shift = utils::dims_to_size(loc_dims2);
    auto nnew = getLocSize() * shift;

    MPI_Comm gath_comm;
    MPI_Comm_split(MPI_COMM_WORLD, env.proc_id / shift, env.proc_id, &gath_comm);
    if (env.proc_id < getDistSize()) {
      int gath_rank;
      MPI_Comm_rank(gath_comm, &gath_rank);

      if (gath_rank == 0) loc_els.resize(nnew);
      MPI_Gather(loc_els.data(), getLocSize(), MPI_C_DOUBLE_COMPLEX, loc_els.data(), getLocSize(), MPI_C_DOUBLE_COMPLEX, 0, gath_comm);
    }

    if (env.proc_id < getDistSize() && env.proc_id % shift == 0 && env.proc_id != 0) {
      MPI_Ssend(loc_els.data(), nnew, MPI_C_DOUBLE_COMPLEX, env.proc_id / shift, 0, MPI_COMM_WORLD);
    }

    if (env.proc_id < getDistSize() / shift && env.proc_id != 0) {
      loc_els.resize(nnew);
      MPI_Recv(loc_els.data(), nnew, MPI_C_DOUBLE_COMPLEX, env.proc_id * shift, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      active = true;
    } else if (env.proc_id != 0) {
      loc_els.clear();
      active = false;
    }

    loc_dims.insert(loc_dims.begin(), loc_dims2.begin(), loc_dims2.end());
    dist_dims.erase(dist_dims.end() - n, dist_dims.end());

    return;
  }

  SDenseTensor* DDenseTensor::share() {
    gather(dist_dims.size());
    loc_els.resize(getSize());

    // Elements will be broadcasted by the constructor
    auto st = new SDenseTensor(env, dims, loc_els, true);
    return st;
  }

  void DDenseTensor::rep_all(std::size_t n) {
    std::vector<MPI_Request> send_reqs(n, MPI_REQUEST_NULL);
    for (std::size_t i = 1; active && (i < n); ++i) {
      MPI_Isend(loc_els.data(), getLocSize(), MPI_C_DOUBLE_COMPLEX, env.proc_id + i * getDistSize(), 0, MPI_COMM_WORLD, &send_reqs.at(i));
    }

    MPI_Request recv_req = MPI_REQUEST_NULL;
    if (!active && env.proc_id < n * getDistSize()) {
      loc_els.resize(getLocSize());
      MPI_Irecv(loc_els.data(), getLocSize(), MPI_C_DOUBLE_COMPLEX, env.proc_id % getDistSize(), 0, MPI_COMM_WORLD, &recv_req);
      active = true;
    }

    dims.insert(dims.begin(), n);
    dist_dims.insert(dist_dims.begin(), n);

    MPI_Waitall(n, send_reqs.data(), MPI_STATUS_IGNORE);
    MPI_Wait(&recv_req, MPI_STATUS_IGNORE);

    return;
  }

  void DDenseTensor::rep_each(std::size_t n) {
    std::vector<MPI_Request> send_reqs(n, MPI_REQUEST_NULL);
    for (std::size_t i = 0; active && (i < n); ++i) {
      // Rank 0 sending data to itself causes a deadlock
      if (i == 0 && env.proc_id == 0) continue;
      MPI_Isend(loc_els.data(), getLocSize(), MPI_C_DOUBLE_COMPLEX, n * env.proc_id + i, 0, MPI_COMM_WORLD, &send_reqs.at(i));
    }

    MPI_Waitall(n, send_reqs.data(), MPI_STATUS_IGNORE);

    MPI_Request recv_req = MPI_REQUEST_NULL;
    if (env.proc_id != 0 && env.proc_id < n * getDistSize()) {
      loc_els.resize(getLocSize());
      MPI_Irecv(loc_els.data(), getLocSize(), MPI_C_DOUBLE_COMPLEX, env.proc_id / n, 0, MPI_COMM_WORLD, &recv_req);
      active = true;
    }

    dims.insert(dims.begin() + getDistDims().size(), n);
    dist_dims.insert(dist_dims.end(), n);

    MPI_Wait(&recv_req, MPI_STATUS_IGNORE);

    return;
  }
}
