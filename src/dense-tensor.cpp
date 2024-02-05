#include <cassert>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "dense-tensor.hpp"
#include "indexing.hpp"

namespace qtnh {
  std::optional<qtnh::tel> DenseTensor::getLocEl(const qtnh::tidx_tup& loc_idxs) const {
    if (!active) return {};

    std::size_t idx = 0;
    std::size_t base = 1;
    for (int i = loc_idxs.size() - 1; i >= 0; --i) {
        idx += loc_idxs.at(i) * base;
        base *= dims.at(i);
    }

    return loc_els.at(idx);
  }

  void DenseTensor::setLocEl(const qtnh::tidx_tup& loc_idxs, qtnh::tel el) {
    if (!active) return;

    std::size_t idx = 0;
    std::size_t base = 1;
    for (int i = loc_idxs.size() - 1; i >= 0; --i) {
        idx += loc_idxs.at(i) * base;
        base *= dims.at(i);
    }

    loc_els.at(idx) = el;

    return;
  }

  SDenseTensor::SDenseTensor(const QTNHEnv& env, const qtnh::tidx_tup& dims, std::vector<qtnh::tel> els)
    : DenseTensor(env, dims, els) {
    auto n = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<qtnh::tidx>());
    if (els.size() != n) {
      throw std::invalid_argument("Invalid length of elements.");
    }
  }

  std::optional<qtnh::tel> SDenseTensor::getGlobEl(const qtnh::tidx_tup& glob_idxs) const {
    return getLocEl(glob_idxs);
  }

  void SDenseTensor::setGlobEl(const qtnh::tidx_tup& glob_idxs, qtnh::tel el) {
    setLocEl(glob_idxs, el);
    return;
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

    qtnh::tidx_tup dims1 = this->getDims();
    qtnh::tidx_tup dims2 = t->getDims();

    std::vector<TIdxFlag> flags1(dims1.size(), TIdxFlag::open);
    std::vector<TIdxFlag> flags2(dims2.size(), TIdxFlag::open);

    for (auto w : wires) {
      flags1.at(w.first) = TIdxFlag::closed;
      flags2.at(w.second) = TIdxFlag::closed;
    }

    TIndexing ti1(dims1, flags1);
    TIndexing ti2(dims2, flags2);
    TIndexing ti3 = TIndexing::app(ti1.cut(TIdxFlag::closed), ti2.cut(TIdxFlag::closed));

    qtnh::tidx_tup dims3 = ti3.getDims();
    std::size_t n = std::accumulate(dims3.begin(), dims3.end(), 1, std::multiplies<qtnh::tidx>());
    std::vector<qtnh::tel> els3(n, 0.0);

    auto t3 = new SDenseTensor(this->env, dims3, els3);

    auto it = ti3.begin();
    for (auto idxs1 : ti1) {
      for (auto idxs2 : ti2) {
        qtnh::tel el3 = 0.0;

        while(true) {
          auto el1 = this->getLocEl(idxs1).value();
          auto el2 = t->getLocEl(idxs2).value();
          el3 += el1 * el2;

          if (ti1.isLast(idxs1, TIdxFlag::closed) && ti2.isLast(idxs2, TIdxFlag::closed)) {
            t3->setLocEl(*it, el3);
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

    return t3;
  }

  Tensor* SDenseTensor::contract(DDenseTensor* t, const std::vector<qtnh::wire>& wires) {
    std::cout << "Contracting SDense with DDense" << std::endl;
    return nullptr;
  }

  DDenseTensor SDenseTensor::distribute(tidx_tup_st nidx) {
    auto loc_dims = dims;
    loc_dims.erase(loc_dims.begin(), loc_dims.begin() + nidx);

    auto dist_dims = dims;
    dist_dims.erase(dist_dims.begin() + nidx, dist_dims.end());

    auto nloc = std::accumulate(loc_dims.begin(), loc_dims.end(), 1, std::multiplies<qtnh::tidx>());
    auto ndist = std::accumulate(dist_dims.begin(), dist_dims.end(), 1, std::multiplies<qtnh::tidx>());

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
    auto loc_dims = dims;
    loc_dims.erase(loc_dims.begin(), loc_dims.begin() + n_dist_idxs);
    
    auto dist_dims = dims;
    dist_dims.erase(dist_dims.begin() + n_dist_idxs, dist_dims.end());

    auto nloc = std::accumulate(loc_dims.begin(), loc_dims.end(), 1, std::multiplies<qtnh::tidx>());
    if (els.size() != nloc) {
      throw std::invalid_argument("Invalid length of elements.");
    }

    auto ndist = std::accumulate(dist_dims.begin(), dist_dims.end(), 1, std::multiplies<qtnh::tidx>());
    if (env.proc_id >= ndist) {
      active = false;
    }
  }

  std::optional<qtnh::tel> DDenseTensor::getGlobEl(const qtnh::tidx_tup& glob_idxs) const {
    auto loc_idxs = glob_idxs;
    loc_idxs.erase(loc_idxs.begin(), loc_idxs.begin() + n_dist_idxs);

    int rank = 0;
    qtnh::tidx base = 1;
    for (int i = 0; i < n_dist_idxs; ++i) {
      rank += glob_idxs.at(i) * base;
      base *= dims.at(i);
    }

    if (env.proc_id == rank) {
      return getLocEl(loc_idxs);
    } else {
      return {};
    }
  }

  void DDenseTensor::setGlobEl(const qtnh::tidx_tup& glob_idxs, qtnh::tel el) {
    auto loc_idxs = glob_idxs;
    loc_idxs.erase(loc_idxs.begin(), loc_idxs.begin() + n_dist_idxs);

    int rank = 0;
    qtnh::tidx base = 1;
    for (int i = 0; i < n_dist_idxs; ++i) {
      rank += glob_idxs.at(i) * base;
      base *= dims.at(i);
    }

    if (env.proc_id == rank) {
      setLocEl(loc_idxs, el);
    }

    return;
  }

  void DDenseTensor::swap(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) {
    throw "Unimplemented funciton!";
  }

  Tensor* DDenseTensor::contract_disp(Tensor* t, const std::vector<qtnh::wire>& wires) {
    return t->contract(this, wires);
  }

  Tensor* DDenseTensor::contract(SDenseTensor* t, const std::vector<qtnh::wire>& wires) {
    std::cout << "Contracting DDense with SDense" << std::endl;
    return nullptr;
  }

  Tensor* DDenseTensor::contract(DDenseTensor* t, const std::vector<qtnh::wire>& wires) {
    std::cout << "Contracting DDense with DDense" << std::endl;
    return nullptr;
  }
}

