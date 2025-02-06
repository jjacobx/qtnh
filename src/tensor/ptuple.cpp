#include "tensor/ptuple.hpp"

namespace qtnh {
  PTuple::PTuple(std::size_t len) : tup_(len) {
    std::iota(tup_.begin(), tup_.end(), 0);
  }

  PTuple PTuple::operator*(const PTuple& ptup) {
    auto& tup1 = tup_;
    auto& tup2 = ptup.tup_;
    auto tup3 = ptup.tup_;

    for (auto i = 0UL; i < tup_.size(); ++i) {
      tup3.at(i) = tup2.at(tup1.at(i));
    }

    PTuple ptup_res(tup_.size());
    ptup_res.tup_ = tup3;

    return ptup_res;
  }

  PTuple PTuple::inv() {
    auto tupi = tup_;

    for (auto i = 0UL; i < tupi.size(); ++i) {
      auto j = tupi.at(i);
      while (tupi.at(j) != i) {
        std::swap(i, tupi.at(j));
        std::swap(i, j);
      }
    }

    PTuple ptup_res(tupi.size());
    ptup_res.tup_ = tupi;

    return ptup_res;
  }

  PTuple::shifter::shifter(std::vector<qtnh::tidx_tup_st>& tup, std::size_t pos)
    : shifter(tup, pos, pos + 1) {}

  PTuple::shifter::shifter(std::vector<qtnh::tidx_tup_st>& tup, std::size_t from, std::size_t to)
    : tup_(tup), from_(from), to_(to) {}
  
  void PTuple::shifter::operator>>(int n) {
    std::vector<qtnh::tidx_tup_st> sub_tup(tup_.begin() + from_, tup_.begin() + to_);
    tup_.erase(tup_.begin() + from_, tup_.begin() + to_);
    tup_.insert(tup_.begin() + from_ + n, sub_tup.begin(), sub_tup.end());
  }

  void PTuple::shifter::operator<<(int n) {
    operator>>(-n);
  }

  PTuple::shifter PTuple::at(std::size_t pos) {
    return shifter(tup_, pos);
  }

  PTuple::shifter PTuple::at(std::size_t from, std::size_t to) {
    return shifter(tup_, from, to);
  }
}
