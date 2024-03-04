#include "tensor/indexing.hpp"

namespace qtnh {
  std::ostream& ops::operator<<(std::ostream& out, const qtnh::tidx_tup& o) {
    out << "(";
    for (std::size_t i = 0; i < o.size(); i++) {
      out << o.at(i);
      if (i < o.size() - 1) {
        out << ", ";
      }
    }
    out << ")";

    return out;
  }

  TIndexing::TIndexing()  : TIndexing(qtnh::tidx_tup{ 1 }) {}

  TIndexing::TIndexing(const qtnh::tidx_tup& dims) : TIndexing(dims, qtnh::tifl_tup(dims.size(), { TIdxT::open, 0 })) {}

  TIndexing::TIndexing(const qtnh::tidx_tup& dims, std::size_t closed_idx) : TIndexing(dims) {
    if (closed_idx >= dims.size()) {
      throw std::out_of_range("Closed index must fit within tensor dimensions.");
    }

    ifls.at(closed_idx) = { TIdxT::closed, 0 };
  }

  TIndexing::TIndexing(const qtnh::tidx_tup& dims, const qtnh::tifl_tup& ifls) : dims(dims), ifls(ifls) {
    if (dims.size() != ifls.size()) {
      throw std::invalid_argument("Dimensions and flags must be of equal length.");
    }

    for (std::size_t i = 0; i < dims.size(); i++) {
      if (dims.at(i) == 0) {
        throw std::invalid_argument("All dimensions should be greater than 0.");
      }
    }
  }

  const qtnh::tidx_tup& TIndexing::getDims() const { return dims; }
  const tifl_tup& TIndexing::getIFls() const { return ifls; }

  bool TIndexing::isValid(const qtnh::tidx_tup& idxs) {
    if (idxs.size() != this->dims.size()) {
      return false;
    }

    for (std::size_t i = 0; i < idxs.size(); i++) {
      if (idxs.at(i) >= this->dims.at(i)) {
        return false;
      }
    }

    return true;
  }

  bool TIndexing::isEqual(const qtnh::tidx_tup& idxs1, const qtnh::tidx_tup& idxs2, TIdxT type, qtnh::tidx_tup_st tag) {
    for (std::size_t i = 0; i < idxs1.size(); i++) {
      if ((ifls.at(i) == qtnh::tifl{ type, tag }) && (idxs1.at(i) != idxs2.at(i))) {
        return false;
      }
    }

    return true;
  }

  bool TIndexing::isLast(const qtnh::tidx_tup& idxs, TIdxT type, qtnh::tidx_tup_st tag) {
    for (std::size_t i = 0; i < idxs.size(); i++) {
      if ((ifls.at(i) == qtnh::tifl{ type, tag }) && (idxs.at(i) != this->dims.at(i) - 1)) {
        return false;
      }
    }

    return true;
  }

  qtnh::tidx_tup& TIndexing::next(qtnh::tidx_tup& idxs, TIdxT type, qtnh::tidx_tup_st tag) {
    for (std::size_t i = idxs.size(); i > 0; i--) {
      if (ifls.at(i - 1) != qtnh::tifl{ type, tag }) {
        continue;
      }

      if (idxs.at(i - 1) < this->dims.at(i - 1) - 1) {
        idxs.at(i - 1)++;
        return idxs;
      } else if (idxs.at(i - 1) == this->dims.at(i - 1) - 1) {
        idxs.at(i - 1) = 0;
        continue;
      } else {
        break;
      }
    }

    throw std::out_of_range("Tuple is outside of indexing.");
  }

  qtnh::tidx_tup& TIndexing::prev(qtnh::tidx_tup& idxs, TIdxT type, qtnh::tidx_tup_st tag) {
    for (std::size_t i = idxs.size(); i > 0; i--) {
      if (ifls.at(i - 1) != qtnh::tifl{ type, tag }) {
        continue;
      }

      if (idxs.at(i - 1) == 0) {
        idxs.at(i - 1) = this->dims.at(i - 1) - 1;
        continue;
      } else if (idxs.at(i - 1) < this->dims.at(i - 1)) {
        idxs.at(i - 1)--;
        return idxs;
      } else {
        break;
      }
    }

    throw std::out_of_range("Tuple is outside of indexing.");
  }

  qtnh::tidx_tup& TIndexing::reset(qtnh::tidx_tup& idxs, TIdxT type, qtnh::tidx_tup_st tag) {
    for (std::size_t i = 0; i < idxs.size(); i++) {
      if (ifls.at(i) == qtnh::tifl{ type, tag }) {
        idxs.at(i) = 0;
      }
    }

    return idxs;
  }

  TIndexing TIndexing::cut(TIdxT type) {
    qtnh::tidx_tup new_dims = dims;
    tifl_tup new_ifls = ifls;

    std::size_t rm_count = 0;
    for (std::size_t i = 0; i < ifls.size(); i++) {
      if (ifls.at(i).first == type) {
        new_dims.erase(new_dims.begin() + i - rm_count);
        new_ifls.erase(new_ifls.begin() + i - rm_count);
        rm_count++;
      }
    }

    TIndexing result(new_dims, new_ifls);
    return result;
  }

  TIndexing TIndexing::cut(qtnh::tifl ifl) {
    qtnh::tidx_tup new_dims = dims;
    tifl_tup new_ifls = ifls;

    std::size_t rm_count = 0;
    for (std::size_t i = 0; i < ifls.size(); i++) {
      if (ifls.at(i) == ifl) {
        new_dims.erase(new_dims.begin() + i - rm_count);
        new_ifls.erase(new_ifls.begin() + i - rm_count);
        rm_count++;
      }
    }

    TIndexing result(new_dims, new_ifls);
    return result;
  }

  bool TIndexing::operator==(const TIndexing& rhs) {
    return (dims == rhs.getDims()) && (ifls == rhs.getIFls());
  }

  bool TIndexing::operator!=(const TIndexing& rhs) {
    return (dims != rhs.getDims()) || (ifls != rhs.getIFls());
  }

  TIndexing::iterator::iterator(const qtnh::tidx_tup& dims, const qtnh::tifl_tup& ifls, const qtnh::tidx_tup& current, TIdxT type, qtnh::tidx_tup_st tag) 
    : dims(dims), ifls(ifls), current(current), active_ifl({ type, tag }) {};

  const qtnh::tifl& TIndexing::iterator::getActiveIFl() const { return active_ifl; }

  TIndexing::iterator& TIndexing::iterator::operator++() {
    TIndexing ti(dims, ifls);
    for (std::size_t i = 0; i < this->current.size(); i++) {
      if ((ifls.at(i) == active_ifl) && (current.at(i) != dims.at(i) - 1)) {
        ti.next(current, active_ifl.first, active_ifl.second);
        return *this;
      }
    }
    
    active_ifl = { TIdxT::oob, 0 };
    return *this;
  }

  bool TIndexing::iterator::operator!=(const iterator& rhs) {
    // Return difference if iterator flags are different
    if (active_ifl != rhs.getActiveIFl()) {
      return true;
    } else if (active_ifl == qtnh::tifl{ TIdxT::oob, 0 }) {
      return false;
    }

    for (std::size_t i = 0; i < ifls.size(); i++) {
      if ((ifls.at(i) == active_ifl) && (current.at(i) != (*rhs).at(i))) {
        return true;
      }
    }

    return false;
  }

  qtnh::tidx_tup TIndexing::iterator::operator*() const {
    return this->current;
  }

  TIndexing::iterator TIndexing::begin(TIdxT type, qtnh::tidx_tup_st tag) {
    qtnh::tidx_tup tup(dims.size(), 0);
    iterator it(dims, ifls, tup, type, tag);
    return it;
  }

  TIndexing::iterator TIndexing::end() {
    qtnh::tidx_tup tup(this->dims.size(), 0);

    iterator it(dims, ifls, tup, TIdxT::oob, 0);
    return it;
  }

  TIndexing TIndexing::app(const TIndexing& ti1, const TIndexing& ti2) {
    qtnh::tidx_tup dims = ti1.getDims();
    qtnh::tifl_tup flags = ti1.getIFls();
    dims.insert(dims.end(), ti2.getDims().begin(), ti2.getDims().end());
    flags.insert(flags.end(), ti2.getIFls().begin(), ti2.getIFls().end());

    TIndexing result(dims, flags);
    return result;
  }
}
