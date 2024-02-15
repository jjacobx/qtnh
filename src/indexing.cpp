#include "indexing.hpp"

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

  TIndexing::TIndexing()  : TIndexing(qtnh::tidx_tup{1}) {}

  TIndexing::TIndexing(const qtnh::tidx_tup& dims) : TIndexing(dims, tidx_flags(dims.size(), TIdxFlag::open)) {}

  TIndexing::TIndexing(const qtnh::tidx_tup& dims, std::size_t closed_idx) : TIndexing(dims) {
    if (closed_idx >= dims.size()) {
      throw std::out_of_range("Closed index must fit within tensor dimensions.");
    }

    this->flags.at(closed_idx) = TIdxFlag::closed;
  }

  TIndexing::TIndexing(const qtnh::tidx_tup& dims, const tidx_flags& flags) : dims(dims), flags(flags) {
    if (dims.size() != flags.size()) {
      throw std::invalid_argument("Dimensions and flags must be of equal length.");
    }

    for (std::size_t i = 0; i < dims.size(); i++) {
      if (dims.at(i) == 0) {
        throw std::invalid_argument("All dimensions should be greater than 0.");
      }
    }
  }

  const qtnh::tidx_tup& TIndexing::getDims() const { return dims; }
  const tidx_flags& TIndexing::getFlags() const { return flags; }

  bool TIndexing::isValid(const qtnh::tidx_tup& tup) {
    if (tup.size() != this->dims.size()) {
      return false;
    }

    for (std::size_t i = 0; i < tup.size(); i++) {
      if (tup.at(i) >= this->dims.at(i)) {
        return false;
      }
    }

    return true;
  }

  bool TIndexing::isEqual(const qtnh::tidx_tup& tup1, const qtnh::tidx_tup& tup2, TIdxFlag fl) {
    for (std::size_t i = 0; i < tup1.size(); i++) {
      if ((this->flags.at(i) == fl) && (tup1.at(i) != tup2.at(i))) {
        return false;
      }
    }

    return true;
  }

  bool TIndexing::isLast(const qtnh::tidx_tup& tup, TIdxFlag fl) {
    for (std::size_t i = 0; i < tup.size(); i++) {
      if ((this->flags.at(i) == fl) && (tup.at(i) != this->dims.at(i) - 1)) {
        return false;
      }
    }

    return true;
  }

  qtnh::tidx_tup& TIndexing::next(qtnh::tidx_tup& tup, TIdxFlag fl) {
    for (std::size_t i = tup.size(); i > 0; i--) {
      if (this->flags.at(i - 1) != fl) {
        continue;
      }

      if (tup.at(i - 1) < this->dims.at(i - 1) - 1) {
        tup.at(i - 1)++;
        return tup;
      } else if (tup.at(i - 1) == this->dims.at(i - 1) - 1) {
        tup.at(i - 1) = 0;
        continue;
      } else {
        break;
      }
    }

    throw std::out_of_range("Tuple is outside of indexing.");
  }

  qtnh::tidx_tup& TIndexing::prev(qtnh::tidx_tup& tup, TIdxFlag fl) {
    for (std::size_t i = tup.size(); i > 0; i--) {
      if (this->flags.at(i - 1) != fl) {
        continue;
      }

      if (tup.at(i - 1) == 0) {
        tup.at(i - 1) = this->dims.at(i - 1) - 1;
        continue;
      } else if (tup.at(i - 1) < this->dims.at(i - 1)) {
        tup.at(i - 1)--;
        return tup;
      } else {
        break;
      }
    }

    throw std::out_of_range("Tuple is outside of indexing.");
  }

  qtnh::tidx_tup& TIndexing::reset(qtnh::tidx_tup& tup, TIdxFlag fl) {
    for (std::size_t i = 0; i < tup.size(); i++) {
      if (this->flags.at(i) == fl) {
        tup.at(i) = 0;
      }
    }

    return tup;
  }

  TIndexing TIndexing::cut(TIdxFlag fl) {
    qtnh::tidx_tup new_dims = this->dims;
    tidx_flags new_flags = this->flags;

    std::size_t rm_count = 0;
    for (std::size_t i = 0; i < flags.size(); i++) {
      if (flags.at(i) == fl) {
        new_dims.erase(new_dims.begin() + i - rm_count);
        new_flags.erase(new_flags.begin() + i - rm_count);
        rm_count++;
      }
    }

    TIndexing result(new_dims, new_flags);
    return result;
  }

  bool TIndexing::operator==(const TIndexing& rhs) {
    return (this->dims == rhs.getDims()) && (this->flags == rhs.getFlags());
  }

  bool TIndexing::operator!=(const TIndexing& rhs) {
    return (this->dims != rhs.getDims()) || (this->flags != rhs.getFlags());
  }

  TIndexing::iterator::iterator(const qtnh::tidx_tup& dims, const tidx_flags& flags, const qtnh::tidx_tup& current, TIdxFlag active_flag) 
    : dims(dims), flags(flags), current(current), active_flag(active_flag) {};

  const TIdxFlag& TIndexing::iterator::getActiveFlag() const { return this->active_flag; }

  TIndexing::iterator& TIndexing::iterator::operator++() {
    TIndexing ti(this->dims, this->flags);
    for (std::size_t i = 0; i < this->current.size(); i++) {
      if ((this->flags.at(i) == this->active_flag) && (this->current.at(i) != this->dims.at(i) - 1)) {
        ti.next(this->current, this->active_flag);
        return *this;
      }
    }
    
    this->active_flag = TIdxFlag::oob;
    return *this;
  }

  bool TIndexing::iterator::operator!=(const iterator& rhs) {
    // Return difference if iterator flags are different
    if (this->active_flag != rhs.getActiveFlag()) {
      return true;
    } else if (this->active_flag == TIdxFlag::oob) {
      return false;
    }

    for (std::size_t i = 0; i < flags.size(); i++) {
      if ((flags.at(i) == this->active_flag) && (this->current.at(i) != (*rhs).at(i))) {
        return true;
      }
    }

    return false;
  }

  qtnh::tidx_tup TIndexing::iterator::operator*() const {
    return this->current;
  }

  TIndexing::iterator TIndexing::begin(TIdxFlag fl) {
    qtnh::tidx_tup tup(this->dims.size(), 0);
    iterator it(this->dims, this->flags, tup, fl);
    return it;
  }

  TIndexing::iterator TIndexing::end() {
    qtnh::tidx_tup tup(this->dims.size(), 0);

    iterator it(this->dims, this->flags, tup, TIdxFlag::oob);
    return it;
  }

  TIndexing TIndexing::app(const TIndexing& ti1, const TIndexing& ti2) {
    qtnh::tidx_tup dims = ti1.getDims();
    tidx_flags flags = ti1.getFlags();
    dims.insert(dims.end(), ti2.getDims().begin(), ti2.getDims().end());
    flags.insert(flags.end(), ti2.getFlags().begin(), ti2.getFlags().end());

    TIndexing result(dims, flags);
    return result;
  }
}
