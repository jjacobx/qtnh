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

    if (els.size() != getLocSize()) {
      throw std::invalid_argument("Invalid length of elements.");
    }

    if (env.proc_id >= getDistSize()) {
      active = false;
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

  void DDenseTensor::rep_all(std::size_t n) {
    return;
  }

  void DDenseTensor::rep_each(std::size_t n) {
    return;
  }
}
