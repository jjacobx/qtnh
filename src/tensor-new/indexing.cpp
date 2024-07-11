#include <algorithm>
#include <numeric>

#include "tensor-new/indexing.hpp"

namespace qtnh {
  TIndexing::TIndexing(qtnh::tidx_tup dims) 
    : TIndexing(dims, std::vector<TIFlag>(dims.size(), { "default", 0 })) {}

  TIndexing::TIndexing(qtnh::tidx_tup dims, std::vector<TIFlag> ifls)
    : dims_(dims), ifls_(ifls), maps_(_generate_maps(ifls_)) {}

  bool TIndexing::isValid(qtnh::tidx_tup idxs) {
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

  bool TIndexing::isEqual(qtnh::tidx_tup idxs1, qtnh::tidx_tup idxs2, std::string ifl_label) {
    for (std::size_t i = 0; i < idxs1.size(); ++i) {
      if ((ifls_.at(i).label == ifl_label) && (idxs1.at(i) != idxs2.at(i))) {
        return false;
      }
    }

    return true;
  }

  bool TIndexing::isLast(qtnh::tidx_tup idxs, std::string ifl_label) {
    for (std::size_t i = 0; i < idxs.size(); ++i) {
      if ((ifls_.at(i).label == ifl_label) && (idxs.at(i) != dims_.at(i) - 1)) {
        return false;
      }
    }

    return true;
  }

  qtnh::tidx_tup TIndexing::next(qtnh::tidx_tup idxs, std::string ifl_label) {
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

  qtnh::tidx_tup TIndexing::prev(qtnh::tidx_tup idxs, std::string ifl_label) {
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

  qtnh::tidx_tup TIndexing::reset(qtnh::tidx_tup idxs, std::string ifl_label) {
    for (std::size_t i = 0; i < idxs.size(); i++) {
      if (ifls_.at(i).label == ifl_label) {
        idxs.at(i) = 0;
      }
    }

    return idxs;
  }

  TIndexing TIndexing::cut(std::string ifl_label) {
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

  std::vector<std::size_t> _generate_maps(std::vector<TIFlag> ifls) {
    std::vector<std::size_t> maps;
    std::iota(maps.begin(), maps.end(), 0);

    std::stable_sort(maps.begin(), maps.end(), [&](std::size_t i, std::size_t j){ return ifls.at(i).tag < ifls.at(j).tag; });
    std::reverse(maps.begin(), maps.end());

    return maps;
  }
}