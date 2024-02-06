#include <cassert>
#include <functional>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <stdexcept>

#include "dense-tensor.hpp"
#include "indexing.hpp"

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
            break;
          }

          ti1.next(idxs1, TIdxFlag::closed);
          ti2.next(idxs2, TIdxFlag::closed);
        }

        #ifdef DEBUG
          std::cout << "t3[" << *it << "] = " << t3.getLocEl(*it).value() << std::endl;
        #endif

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
    : DenseTensor(env, dims, els) {
    // TODO: Broadcast from rank 0
    if (els.size() != getLocSize()) {
      throw std::invalid_argument("Invalid length of elements.");
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
    auto i = idxs_to_i(loc_idxs, dims);
    return loc_els.at(i);
  }

  void SDenseTensor::swap(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) {
    throw "Unimplemented funciton!";
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
    : DenseTensor(env, dims, els), n_dist_idxs(n_dist_idxs) {
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
    qtnh::tidx_tup loc_idxs(glob_idxs.begin(), glob_idxs.begin() + n_dist_idxs);
    qtnh::tidx_tup dist_idxs(glob_idxs.begin() + n_dist_idxs, glob_idxs.end());
    auto rank = idxs_to_i(dist_idxs, getDistDims());

    if (env.proc_id == rank) {
      return getLocEl(loc_idxs);
    } else {
      return {};
    }
  }

  void DDenseTensor::setEl(const qtnh::tidx_tup& glob_idxs, qtnh::tel el) {
    qtnh::tidx_tup loc_idxs(glob_idxs.begin(), glob_idxs.begin() + n_dist_idxs);
    qtnh::tidx_tup dist_idxs(glob_idxs.begin() + n_dist_idxs, glob_idxs.end());
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
    auto i = idxs_to_i(loc_idxs, dims);
    return loc_els.at(i);
  }

  void DDenseTensor::swap(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) {
    throw "Unimplemented funciton!";
  }

  Tensor* DDenseTensor::contract_disp(Tensor* t, const std::vector<qtnh::wire>& wires) {
    return t->contract(this, wires);
  }

  Tensor* DDenseTensor::contract(SDenseTensor* t, const std::vector<qtnh::wire>& wires) {
    #ifdef DEBUG
      std::cout << "Contracting DDense with SDense" << std::endl;
    #endif

    TIndexing ti1 = _get_indexing(this, wires, 0);
    TIndexing ti2 = _get_indexing(t, wires, 1);
    TIndexing ti3 = TIndexing::app(ti1.cut(TIdxFlag::closed), ti2.cut(TIdxFlag::closed));

    std::size_t nloc = dims_to_size(ti3.getDims());
    std::vector<qtnh::tel> els3(nloc, 0.0);

    auto dims3 = _concat_dims(dist_dims, t->getDistDims(), ti3.getDims());
    auto nidx = dist_dims.size();

    auto t3 = new DDenseTensor(this->env, dims3, els3, nidx);

    _set_els(this, t, t3, ti1, ti2, ti3);

    return t3;
  }

  Tensor* DDenseTensor::contract(DDenseTensor* t, const std::vector<qtnh::wire>& wires) {
    std::cout << "Contracting DDense with DDense" << std::endl;
    return nullptr;
  }

  // Tensor* DDenseTensor::contract(DDenseTensor* t, const std::vector<qtnh::wire>& wires) {
  //   #ifdef DEBUG
  //     std::cout << "Contracting DDense with DDense" << std::endl;
  //   #endif

  //   auto n_dist_idxs1 = n_dist_idxs;
  //   auto n_dist_idxs2 = t->getDistDims().size();

  //   auto dims1 = this->getDims();
  //   auto ddims1 = this->getDims();
  //   dims1.erase(dims1.begin(), dims1.begin() + n_dist_idxs1);
  //   ddims1.erase(ddims1.begin() + n_dist_idxs1, ddims1.end());
  //   auto dims2 = t->getDims();
  //   auto ddims2 = t->getDims();
  //   dims2.erase(dims2.begin(), dims2.begin() + n_dist_idxs2);
  //   ddims2.erase(ddims2.begin() + n_dist_idxs2, ddims2.end());

  //   std::vector<TIdxFlag> flags1(dims1.size(), TIdxFlag::open);
  //   std::vector<TIdxFlag> flags2(dims2.size(), TIdxFlag::open);

  //   for (auto w : wires) {
  //     flags1.at(w.first - n_dist_idxs1) = TIdxFlag::closed;
  //     flags2.at(w.second - n_dist_idxs2) = TIdxFlag::closed;
  //   }

  //   TIndexing ti1(dims1, flags1);
  //   TIndexing ti2(dims2, flags2);
  //   TIndexing ti3 = TIndexing::app(ti1.cut(TIdxFlag::closed), ti2.cut(TIdxFlag::closed));

  //   auto dims3 = ti3.getDims();
  //   std::size_t n = std::accumulate(dims3.begin(), dims3.end(), 1, std::multiplies<qtnh::tidx>());
  //   std::vector<qtnh::tel> els3(n, 0.0);

  //   auto els1_send = this->getLocEls();
  //   auto els1_recv = std::vector<qtnh::tel>(els1_send.size());
  //   auto els2_send = t->getLocEls();
  //   auto els2_recv = std::vector<qtnh::tel>(els2_send.size());

  //   std::size_t n1 = std::accumulate(ddims1.begin(), ddims1.end(), 1, std::multiplies<qtnh::tidx>());
  //   std::size_t n2 = std::accumulate(ddims2.begin(), ddims2.end(), 1, std::multiplies<qtnh::tidx>());
  //   std::size_t len1 = std::accumulate(dims1.begin(), dims1.end(), 1, std::multiplies<qtnh::tidx>());
  //   std::size_t len2 = std::accumulate(dims2.begin(), dims2.end(), 1, std::multiplies<qtnh::tidx>());

  //   std::vector<MPI_Request> reqs1(n2);
  //   std::vector<MPI_Request> reqs2(n1);
  //   for (int i = 0; i < n2 && this->isActive(); ++i) {
  //     MPI_Isend(els1_send.data(), len1, MPI_2DOUBLE_COMPLEX, env.proc_id * n2 + i, 0, MPI_COMM_WORLD, &reqs1.at(i));
  //   } 
  //   for (int i = 0; i < n1 && t->isActive(); i++) {
  //     MPI_Isend(els2_send.data(), len2, MPI_2DOUBLE_COMPLEX, i * n2 + env.proc_id, 0, MPI_COMM_WORLD, &reqs1.at(i));
  //   }

  //   std::vector<MPI_Status> stats(2);
  //   if (env.proc_id < n1 * n2) {
  //     MPI_Recv(els1_recv.data(), len1, MPI_2DOUBLE_COMPLEX, env.proc_id % n2, 0, MPI_COMM_WORLD, &stats.at(1));
  //     MPI_Recv(els2_recv.data(), len2, MPI_2DOUBLE_COMPLEX, env.proc_id / n2, 0, MPI_COMM_WORLD, &stats.at(2));
  //   } 

  //   SDenseTensor loc1(this->env, dims1, els1_recv);
  //   SDenseTensor loc2(this->env, dims2, els2_recv);

  //   dims3.insert(dims3.begin(), ddims2.begin(), ddims2.end());
  //   dims3.insert(dims3.begin(), ddims1.begin(), ddims1.end());
  //   auto t3 = new DDenseTensor(this->env, dims3, els3, n_dist_idxs1 + n_dist_idxs2);

  //   _set_els(&loc1, &loc2, t3, ti1, ti2, ti3);

  //   return t3;
  // }

  void DDenseTensor::rep_all(std::size_t n) {
    std::vector<MPI_Request> send_reqs(n, MPI_REQUEST_NULL);
    for (int i = 1; active && (i < n); ++i) {
      MPI_Isend(loc_els.data(), getLocSize(), MPI_C_DOUBLE_COMPLEX, env.proc_id + i * getDistSize(), 0, MPI_COMM_WORLD, &send_reqs.at(i));
    }

    MPI_Request recv_req = MPI_REQUEST_NULL;
    if (!active && env.proc_id < n * getDistSize()) {
      loc_els.reserve(getLocSize());
      MPI_Irecv(loc_els.data(), getLocSize(), MPI_C_DOUBLE_COMPLEX, env.proc_id % getDistSize(), 0, MPI_COMM_WORLD, &recv_req);
      active = true;
    }

    dims.insert(dims.begin(), n);
    dist_dims.insert(dist_dims.begin(), n);
    ++n_dist_idxs;

    MPI_Waitall(n, send_reqs.data(), MPI_STATUS_IGNORE);
    MPI_Wait(&recv_req, MPI_STATUS_IGNORE);

    return;
  }

  void DDenseTensor::rep_each(std::size_t n) {
    std::vector<MPI_Request> send_reqs(n, MPI_REQUEST_NULL);
    for (int i = 0; active && (i < n); ++i) {
      // Rank 0 sending data to itself is a deadlock
      if (i == 0 && env.proc_id == 0) continue;
      MPI_Isend(loc_els.data(), getLocSize(), MPI_C_DOUBLE_COMPLEX, n * env.proc_id + i, 0, MPI_COMM_WORLD, &send_reqs.at(i));
    }

    MPI_Waitall(n, send_reqs.data(), MPI_STATUS_IGNORE);

    MPI_Request recv_req = MPI_REQUEST_NULL;
    if (env.proc_id != 0 && env.proc_id < n * getDistSize()) {
      loc_els.reserve(getLocSize());
      MPI_Irecv(loc_els.data(), getLocSize(), MPI_C_DOUBLE_COMPLEX, env.proc_id / n, 0, MPI_COMM_WORLD, &recv_req);
      active = true;
    }

    dims.insert(dims.begin() + getDistSize(), n);
    dist_dims.insert(dist_dims.begin() + getDistSize(), n);
    ++n_dist_idxs;

    MPI_Wait(&recv_req, MPI_STATUS_IGNORE);

    return;
  }
}
