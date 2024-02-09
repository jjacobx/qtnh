#include <cassert>
#include <functional>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <stdexcept>

#include "dense-tensor.hpp"
#include "indexing.hpp"

using namespace qtnh::ops;

namespace qtnh {
  TIndexing _get_indexing(Tensor* t, const std::vector<qtnh::wire>& wires, bool second) {
    auto loc_dims = t->getLocDims();
    auto dist_dims = t->getDistDims();

    std::vector<TIdxFlag> flags(loc_dims.size(), TIdxFlag::open);

    for (auto w : wires) {
      flags.at((second ? w.second : w.first) - dist_dims.size()) = TIdxFlag::closed;
    }

    return TIndexing(loc_dims, flags);
  }

  qtnh::tidx_tup _concat_dims(qtnh::tidx_tup dist_dims1, qtnh::tidx_tup dist_dims2, qtnh::tidx_tup loc_dims) {
    auto dist_dims = dist_dims1;
    dist_dims.insert(dist_dims.end(), dist_dims2.begin(), dist_dims2.end());

    auto dims = dist_dims;
    dims.insert(dims.end(), loc_dims.begin(), loc_dims.end());

    return dims;
  }

  void _set_els(Tensor* t1, Tensor* t2, DenseTensor* t3, TIndexing ti1, TIndexing ti2, TIndexing ti3) {
    auto it = ti3.begin();
    for (auto idxs1 : ti1) {
      for (auto idxs2 : ti2) {
        qtnh::tel el3 = 0.0;

        while(t3->isActive()) {
          auto el1 = (*t1)[idxs1];
          auto el2 = (*t2)[idxs2];
          el3 += el1 * el2;

          if (ti1.isLast(idxs1, TIdxFlag::closed) && ti2.isLast(idxs2, TIdxFlag::closed)) {
            (*t3)[*it] = el3;

            #ifdef DEBUG
              std::cout << "t3[" << *it << "] = " << (*t3)[*it] << std::endl;
            #endif

            break;
          }

          ti1.next(idxs1, TIdxFlag::closed);
          ti2.next(idxs2, TIdxFlag::closed);
        }

        ti1.reset(idxs1, TIdxFlag::closed);
        ++it;
      }
    }
  }

  std::optional<qtnh::tel> DenseTensor::getLocEl(const qtnh::tidx_tup& loc_idxs) const {
    if (!active) {
      return {};
    } else { 
      return (*this)[loc_idxs];
    }
  }

  qtnh::tel DenseTensor::operator[](const qtnh::tidx_tup& loc_idxs) const {
    auto i = idxs_to_i(loc_idxs, dims);
    return loc_els.at(i);
  }

  SDenseTensor::SDenseTensor(const QTNHEnv& env, const qtnh::tidx_tup& dims, std::vector<qtnh::tel> els)
    : SDenseTensor(env, dims, els, DEF_STENSOR_BCAST) {}

  SDenseTensor::SDenseTensor(const QTNHEnv& env, const qtnh::tidx_tup& dims, std::vector<qtnh::tel> els, bool bcast)
    : DenseTensor(env, dims, els) {
    if (loc_els.size() != getSize()) {
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

  std::optional<qtnh::tel> SDenseTensor::getEl(const qtnh::tidx_tup& glob_idxs) const {
    return getLocEl(glob_idxs);
  }

  void SDenseTensor::setEl(const qtnh::tidx_tup& glob_idxs, qtnh::tel el) {
    setLocEl(glob_idxs, el);
    return;
  }

  void SDenseTensor::setLocEl(const qtnh::tidx_tup& loc_idxs, qtnh::tel el) {
    if (!active) return;
    (*this)[loc_idxs] = el;
    return;
  }

  qtnh::tel& SDenseTensor::operator[](const qtnh::tidx_tup& loc_idxs) {
    auto i = idxs_to_i(loc_idxs, loc_dims);
    return loc_els.at(i);
  }

  void SDenseTensor::swap(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) {
    if (dims.at(idx1) != dims.at(idx2)) {
      throw std::runtime_error("Asymmetric swaps are not allowed");
    }

    tidx_flags flags(dims.size(), TIdxFlag::open);
    flags.at(idx1) = flags.at(idx2) = TIdxFlag::closed;
    TIndexing ti(dims, flags);

    for (auto idxs : ti) {
      auto idxs1 = idxs;
      auto idxs2 = idxs;

      for (qtnh::tidx i = 0; i < dims.at(idx1) - 1; ++i) {
        idxs1.at(idx1) = idxs2.at(idx1) = i;
        for (qtnh::tidx j = i + 1; j < dims.at(idx2); ++j) {
          idxs1.at(idx2) = idxs2.at(idx1) = j;
          std::swap((*this)[idxs1], (*this)[idxs2]);
        }
      }
    }

    return;
  }

  Tensor* SDenseTensor::contract_disp(Tensor* t, const std::vector<qtnh::wire>& wires) {
    return t->contract(this, wires);
  }

  Tensor* SDenseTensor::contract(SDenseTensor* t, const std::vector<qtnh::wire>& wires) {
    #ifdef DEBUG
      std::cout << "Contracting SDense with SDense" << std::endl;
    #endif

    TIndexing ti1 = _get_indexing(this, wires, 0);
    TIndexing ti2 = _get_indexing(t, wires, 1);
    TIndexing ti3 = TIndexing::app(ti1.cut(TIdxFlag::closed), ti2.cut(TIdxFlag::closed));

    auto dims3 = ti3.getDims();
    auto nloc = dims_to_size(dims3);
    std::vector<qtnh::tel> els3(nloc, 0.0);

    auto t3 = new SDenseTensor(this->env, dims3, els3);

    _set_els(this, t, t3, ti1, ti2, ti3);

    return t3;
  }

  Tensor* SDenseTensor::contract(DDenseTensor* t, const std::vector<qtnh::wire>& wires) {
    #ifdef DEBUG
      std::cout << "Contracting DDense with SDense" << std::endl;
    #endif

    auto ti1 = _get_indexing(this, wires, 0);
    auto ti2 = _get_indexing(t, wires, 1);
    auto ti3 = TIndexing::app(ti1.cut(TIdxFlag::closed), ti2.cut(TIdxFlag::closed));

    std::size_t nloc = dims_to_size(ti3.getDims());
    std::vector<qtnh::tel> els3(nloc, 0.0);

    auto dims3 = _concat_dims(dist_dims, t->getDistDims(), ti3.getDims());
    auto nidx = t->getDistDims().size();

    auto t3 = new DDenseTensor(this->env, dims3, els3, nidx);

    _set_els(this, t, t3, ti1, ti2, ti3);

    return t3;
  }

  DDenseTensor SDenseTensor::distribute(tidx_tup_st nidx) {
    qtnh::tidx_tup loc_dims(dims.begin() + nidx, dims.end());
    qtnh::tidx_tup dist_dims(dims.begin(), dims.begin() + nidx);

    auto nloc = dims_to_size(loc_dims);
    auto ndist = dims_to_size(dist_dims);

    auto els = loc_els;
    if (env.proc_id < ndist) {
      els.erase(els.begin(), els.begin() + env.proc_id * nloc);
      els.erase(els.begin() + nloc, els.end());
    } else {
      els.clear();
    }

    return DDenseTensor(env, dims, els, nidx);
  }

  DDenseTensor::DDenseTensor(const QTNHEnv& env, const qtnh::tidx_tup& dims, std::vector<qtnh::tel> els, qtnh::tidx_tup_st n_dist_idxs)
    : DenseTensor(env, dims, els) {
    loc_dims = qtnh::tidx_tup(dims.begin() + n_dist_idxs, dims.end());
    dist_dims = qtnh::tidx_tup(dims.begin(), dims.begin() + n_dist_idxs);

    if (env.proc_id >= getDistSize()) {
      active = false;
    }
 
    if (active && els.size() != getLocSize()) {
      throw std::invalid_argument("Invalid length of elements.");
    }
  }

  std::optional<qtnh::tel> DDenseTensor::getEl(const qtnh::tidx_tup& glob_idxs) const {
    qtnh::tidx_tup loc_idxs(glob_idxs.begin(), glob_idxs.begin() + dist_dims.size());
    qtnh::tidx_tup dist_idxs(glob_idxs.begin() + dist_dims.size(), glob_idxs.end());
    auto rank = idxs_to_i(dist_idxs, getDistDims());

    if (env.proc_id == rank) {
      return getLocEl(loc_idxs);
    } else {
      return {};
    }
  }

  void DDenseTensor::setEl(const qtnh::tidx_tup& glob_idxs, qtnh::tel el) {
    qtnh::tidx_tup loc_idxs(glob_idxs.begin(), glob_idxs.begin() + dist_dims.size());
    qtnh::tidx_tup dist_idxs(glob_idxs.begin() + dist_dims.size(), glob_idxs.end());
    auto rank = idxs_to_i(dist_idxs, getDistDims());

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

  qtnh::tel& DDenseTensor::operator[](const qtnh::tidx_tup& loc_idxs) {
    auto i = idxs_to_i(loc_idxs, loc_dims);
    return loc_els.at(i);
  }

  void DDenseTensor::swap(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) {
    if (dims.at(idx1) != dims.at(idx2)) {
      throw std::runtime_error("Asymmetric swaps are not allowed");
    }

    if (idx1 < dist_dims.size() || idx2 < dist_dims.size()) {
      throw std::runtime_error("Swaps with distributed indices are not implemented");
    }

    throw_unimplemented();
  }

  Tensor* DDenseTensor::contract_disp(Tensor* t, const std::vector<qtnh::wire>& wires) {
    return t->contract(this, wires);
  }

  Tensor* DDenseTensor::contract(SDenseTensor* t, const std::vector<qtnh::wire>& wires) {
    #ifdef DEBUG
      std::cout << "Contracting DDense with SDense" << std::endl;
    #endif

    auto ti1 = _get_indexing(this, wires, 0);
    auto ti2 = _get_indexing(t, wires, 1);
    auto ti3 = TIndexing::app(ti1.cut(TIdxFlag::closed), ti2.cut(TIdxFlag::closed));

    auto nloc = dims_to_size(ti3.getDims());
    std::vector<qtnh::tel> els3(nloc, 0.0);

    auto dims3 = _concat_dims(dist_dims, t->getDistDims(), ti3.getDims());
    auto nidx = dist_dims.size();

    auto t3 = new DDenseTensor(this->env, dims3, els3, nidx);

    _set_els(this, t, t3, ti1, ti2, ti3);

    return t3;
  }

  Tensor* DDenseTensor::contract(DDenseTensor* t, const std::vector<qtnh::wire>& wires) {
    #ifdef DEBUG
      std::cout << "Contracting DDense with DDense" << std::endl;
    #endif

    auto ti1 = _get_indexing(this, wires, 0);
    auto ti2 = _get_indexing(t, wires, 1);
    auto ti3 = TIndexing::app(ti1.cut(TIdxFlag::closed), ti2.cut(TIdxFlag::closed));

    auto nloc = dims_to_size(ti3.getDims());
    std::vector<qtnh::tel> els3(0);

    auto dims3 = _concat_dims(dist_dims, t->getDistDims(), ti3.getDims());
    auto nidx = dist_dims.size() + t->getDistDims().size();

    auto n1 = dims_to_size(dist_dims);
    auto n2 = dims_to_size(t->getDistDims());

    if (env.proc_id < n1 * n2) els3.assign(nloc, 0.0);
    auto t3 = new DDenseTensor(this->env, dims3, els3, nidx);

    this->rep_all(n2);
    t->rep_each(n1);

    _set_els(this, t, t3, ti1, ti2, ti3);

    return t3;
  }

  void DDenseTensor::scatter(tidx_tup_st n) {
    auto dist_dims2 = qtnh::tidx_tup(loc_dims.begin(), loc_dims.begin() + n);
    auto shift = dims_to_size(dist_dims2);

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
    auto shift = dims_to_size(loc_dims2);
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

  SDenseTensor DDenseTensor::share() {
    gather(dist_dims.size());
    loc_els.resize(getSize());

    // Elements will be broadcasted by the constructor
    return SDenseTensor(env, dims, loc_els, true);
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
