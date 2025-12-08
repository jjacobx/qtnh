#include <numeric>

#include "util/ptuple.hpp"

namespace qtnh {
  PTuple::PTuple(std::size_t len) : tup_(len) {
    std::iota(tup_.begin(), tup_.end(), 0);
  }

  PTuple::PTuple(tup_t tup) : tup_(tup) {}

  tup_t _prod(tup_t tup1, tup_t tup2) {
    auto tup3 = tup_t(tup1.size());
    for (auto i = 0UL; i < tup1.size(); ++i) {
      tup3.at(i) = tup2.at(tup1.at(i));
    }

    return tup3;
  }

  tup_t _inv(tup_t tup) {
    auto tupi = tup_t(tup.size(), X);
    for (auto i = 0UL; i < tup.size(); ++i) {
      auto j = tup.at(i);
      while (tupi.at(j) == X) {
        tupi.at(j) = i;
        i = j;
        j = tup.at(j);
      }
    }

    return tupi;
  }

  PTuple::shifter::shifter(tup_t* tup, std::size_t pos)
  : shifter(tup, pos, pos + 1)
  {}

  PTuple::shifter::shifter(tup_t* tup, std::size_t from, std::size_t to)
  : tup_(tup)
  , from_(from)
  , to_(to)
  {}
  
  void PTuple::shifter::operator>>(int n) {
    std::vector<qtnh::tidx_tup_st> sub_tup((*tup_).begin() + from_, (*tup_).begin() + to_);
    (*tup_).erase((*tup_).begin() + from_, (*tup_).begin() + to_);
    (*tup_).insert((*tup_).begin() + from_ + n, sub_tup.begin(), sub_tup.end());
  }

  void PTuple::shifter::operator<<(int n) {
    operator>>(-n);
  }

  PTuple::shifter PTuple::at(std::size_t pos) {
    return shifter(&tup_, pos);
  }

  PTuple::shifter PTuple::at(std::size_t from, std::size_t to) {
    return shifter(&tup_, from, to);
  }

  PTupleTar PTupleTar::toTar() const {
    return PTupleTar(tup());
  }

  PTupleSrc PTupleTar::toSrc() const {
    return PTupleSrc(_inv(tup()));
  }

  PTupleTar PTupleTar::operator*(const PTupleTar& ptup) const {
    return PTupleTar(_prod(ptup.tup(), tup()));
  }

  PTupleTar PTupleTar::inv() const {
    return PTupleTar(_inv(tup()));
  }

  PTupleTar PTupleSrc::toTar() const {
    return PTupleTar(_inv(tup()));
  }

  PTupleSrc PTupleSrc::toSrc() const {
    return PTupleSrc(tup());
  }

  PTupleSrc PTupleSrc::operator*(const PTupleSrc& ptup) const {
    return PTupleSrc(_prod(tup(), ptup.tup()));
  }

  PTupleSrc PTupleSrc::inv() const {
    return PTupleSrc(_inv(tup()));
  }

  IndexGroup::IndexGroup(std::vector<std::string> labels, std::vector<tup_t> groups)
  : labels_(labels)
  , groups_()
  {
    for (auto i = 0UL; i < labels.size(); ++i) {
      groups_.insert({labels.at(i), groups.at(i)});
    }
  }

  PTupleSrc IndexGroup::ptup() const {
    tup_t tup;
    for (auto i = 0UL; i < labels_.size(); ++i) {
      auto& group = groups_.at(labels_.at(i));
      tup.insert(tup.end(), group.begin(), group.end());
    }
    
    return PTupleSrc(tup);
  }

  void IndexGroup::reorder(std::vector<std::string> labels) {
    labels_ = labels;
  }

  tup_t& IndexGroup::at(std::string k) {
    return groups_.at(k);
  }

  qtnh::tidx_tup_st& IndexGroup::at(std::string k, std::size_t i) {
    return groups_.at(k).at(i);
  }
}
