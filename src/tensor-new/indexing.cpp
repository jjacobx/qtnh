#include <algorithm>
#include <numeric>

#include "core/utils.hpp"
#include "tensor-new/indexing.hpp"

namespace qtnh {
  TIndexing::TIndexing(qtnh::tidx_tup dims) 
    : TIndexing(dims, std::vector<TIFlag>(dims.size(), { "default", 0 })) {}

  TIndexing::TIndexing(qtnh::tidx_tup dims, std::vector<TIFlag> ifls)
    : dims_(dims), ifls_(ifls), maps_(_generate_maps(ifls_)) {}

  bool TIndexing::isValid(qtnh::tidx_tup idxs) const {
    if (idxs.size() != dims_.size()) {
      return false;
    }

    for (std::size_t i = 0; i < idxs.size(); ++i) {
      if (idxs.at(i) >= dims_.at(i)) {
        return false;
      }
    }

    return true;
  }

  bool TIndexing::isEqual(qtnh::tidx_tup idxs1, qtnh::tidx_tup idxs2, std::string ifl_label) const {
    for (std::size_t i = 0; i < idxs1.size(); ++i) {
      if ((ifls_.at(i).label == ifl_label) && (idxs1.at(i) != idxs2.at(i))) {
        return false;
      }
    }

    return true;
  }

  bool TIndexing::isLast(qtnh::tidx_tup idxs, std::string ifl_label) const {
    for (std::size_t i = 0; i < idxs.size(); ++i) {
      if ((ifls_.at(i).label == ifl_label) && (idxs.at(i) != dims_.at(i) - 1)) {
        return false;
      }
    }

    return true;
  }

  qtnh::tidx_tup TIndexing::next(qtnh::tidx_tup idxs, std::string ifl_label) const {
    for (std::size_t i = 0; i < idxs.size(); ++i) {
      auto k = maps_.at(i);

      if (ifls_.at(k).label != ifl_label) {
        continue;
      }

      if (idxs.at(k) < dims_.at(k) - 1) {
        idxs.at(k)++;
        return idxs;
      } else if (idxs.at(k) == dims_.at(k) - 1) {
        idxs.at(k) = 0;
        continue;
      } else {
        break;
      }
    }

    throw std::out_of_range("Tuple is outside of indexing.");
  }

  qtnh::tidx_tup TIndexing::prev(qtnh::tidx_tup idxs, std::string ifl_label) const {
    for (std::size_t i = 0; i < idxs.size(); ++i) {
      auto k = maps_.at(i);

      if (ifls_.at(k).label != ifl_label) {
        continue;
      }

      if (idxs.at(k) == 0) {
        idxs.at(k) = dims_.at(k) - 1;
        continue;
      } else if (idxs.at(k) < dims_.at(k)) {
        idxs.at(k)--;
        return idxs;
      } else {
        break;
      }
    }

    throw std::out_of_range("Tuple is outside of indexing.");
  }

  qtnh::tidx_tup TIndexing::reset(qtnh::tidx_tup idxs, std::string ifl_label) const {
    for (std::size_t i = 0; i < idxs.size(); i++) {
      if (ifls_.at(i).label == ifl_label) {
        idxs.at(i) = 0;
      }
    }

    return idxs;
  }

  TIndexing TIndexing::cut(std::string ifl_label) const {
    auto new_dims = dims_;
    auto new_ifls = ifls_;

    std::size_t rm_count = 0;
    for (std::size_t i = 0; i < ifls_.size(); i++) {
      if (ifls_.at(i).label == ifl_label) {
        new_dims.erase(new_dims.begin() + i - rm_count);
        new_ifls.erase(new_ifls.begin() + i - rm_count);
        rm_count++;
      }
    }

    TIndexing new_ti(new_dims, new_ifls);
    return new_ti;
  }

  TIndexing::TupIterator::TupIterator(qtnh::tidx_tup dims, std::vector<std::size_t> order, qtnh::tidx_tup start, bool is_end)
    : dims_(dims), order_(order), current_(start), is_end_(is_end) {}

  TIndexing::TupIterator TIndexing::TupIterator::begin() {
    auto new_start = current_;
    for (std::size_t i = 0; i < order_.size(); ++i) {
      auto k = order_.at(i);
      new_start.at(k) = 0;
    }

    return TupIterator(dims_, order_, new_start, false);
  }

  TIndexing::TupIterator TIndexing::TupIterator::end() {
    return TupIterator(dims_, order_, current_, true);
  }

  TIndexing::TupIterator& TIndexing::TupIterator::operator++() {
    for (std::size_t i = 0; i < order_.size(); ++i) {
      auto k = order_.at(i);

      if (current_.at(k) < dims_.at(k) - 1) {
        current_.at(k)++;
        return *this;
      } else if (current_.at(k) == dims_.at(k) - 1) {
        current_.at(k) = 0;
        continue;
      }
    }
    
    is_end_ = true;
    return *this;
  }

  TIndexing::TupIterator TIndexing::TupIterator::operator++(int) {
    TupIterator old = *this;
    operator++();
    return old;
  }

  qtnh::tidx_tup TIndexing::TupIterator::operator*() const {
    return current_;
  }

  TIndexing::NumIterator::NumIterator(qtnh::tidx_tup dims, std::vector<std::size_t> incrs, std::size_t zero, qtnh::tidx_tup current_idxs, bool is_end)
    : dims_(dims), incrs_(incrs), zero_(zero), current_idxs_(current_idxs), is_end_(is_end) {}

  TIndexing::NumIterator TIndexing::NumIterator::begin() {
    return NumIterator(dims_, incrs_, zero_, qtnh::tidx_tup(current_idxs_.size(), 0), false);
  }

  TIndexing::NumIterator TIndexing::NumIterator::end() {
    return NumIterator(dims_, incrs_, zero_, current_idxs_, true);
  }

  TIndexing::NumIterator& TIndexing::NumIterator::operator++() {
    for (std::size_t i = 0; i < current_idxs_.size(); ++i) {

      if (current_idxs_.at(i) < dims_.at(i) - 1) {
        current_idxs_.at(i)++;
        return *this;
      } else if (current_idxs_.at(i) == dims_.at(i) - 1) {
        current_idxs_.at(i) = 0;
        continue;
      }
    }
    
    is_end_ = true;
    return *this;
  }

  TIndexing::NumIterator TIndexing::NumIterator::operator++(int) {
    NumIterator old = *this;
    operator++();
    return old;
  }

  std::size_t TIndexing::NumIterator::operator*() const {
    auto result = zero_;
    for (std::size_t i = 0; i < current_idxs_.size(); ++i) {
      result += current_idxs_.at(i) * incrs_.at(i);
    }

    return result;
  }

  TIndexing::TupIterator TIndexing::tup(std::string ifl_label) const {
    return tup(ifl_label, qtnh::tidx_tup(dims_.size(), 0));
  }

  TIndexing::TupIterator TIndexing::tup(std::string ifl_label, qtnh::tidx_tup start) const {
    std::vector<std::size_t> order;
    for (std::size_t i = 0; i < maps_.size(); ++i) {
      auto k = maps_.at(i);
      if (ifls_.at(k).label == ifl_label) {
        order.push_back(maps_.at(i));
      }
    }

    return TIndexing::TupIterator(dims_, order, start, false);
  }

  TIndexing::NumIterator TIndexing::num(std::string ifl_label) const {
    return num(ifl_label, qtnh::tidx_tup(dims_.size(), 0));
  }

  TIndexing::NumIterator TIndexing::num(std::string ifl_label, qtnh::tidx_tup start) const {
    std::vector<std::size_t> incrs_all(dims_.size());
    std::size_t zero = 0;

    auto n = dims_.size();
    std::size_t base = 1;
    for (std::size_t i = 0; i < n; ++i) {
      incrs_all.at(n - i - 1) = base;

      if (ifls_.at(n - i - 1).label != ifl_label) {
        zero += start.at(n - i - 1) * base;
      }

      base *= dims_.at(n - i - 1);
    }

    qtnh::tidx_tup dims;
    std::vector<std::size_t> incrs;
    qtnh::tidx_tup current_idxs;

    for (std::size_t i = 0; i < maps_.size(); ++i) {
      auto k = maps_.at(i);

      if (ifls_.at(k).label == ifl_label) {
        dims.push_back(dims_.at(k));
        incrs.push_back(incrs_all.at(k));
        current_idxs.push_back(start.at(k));
      }
    }

    return TIndexing::NumIterator(dims, incrs, zero, current_idxs, false);
  }

  TIndexing TIndexing::app(const TIndexing& ti1, const TIndexing& ti2) {
    auto dims = ti1.dims_;
    auto ifls = ti1.ifls_;
    dims.insert(dims.end(), ti2.dims_.begin(), ti2.dims_.end());
    ifls.insert(ifls.end(), ti2.ifls_.begin(), ti2.ifls_.end());

    return TIndexing(dims, ifls);
  }

  std::vector<std::size_t> _generate_maps(std::vector<TIFlag> ifls) {
    std::vector<std::size_t> maps;
    std::iota(maps.begin(), maps.end(), 0);

    std::stable_sort(maps.begin(), maps.end(), [&](std::size_t i, std::size_t j){ return ifls.at(i).tag < ifls.at(j).tag; });
    std::reverse(maps.begin(), maps.end());

    return maps;
  }
}
