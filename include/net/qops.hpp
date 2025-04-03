#ifndef __NET_OPS__
#define __NET_OPS__

#include "net/mps.hpp"

namespace qtnh {
  namespace qops {
    MPO swap(const QTNHEnv& env, std::size_t n);
    MPO cmpo(const QTNHEnv& env, const MPO& mpo, std::size_t n);
    std::vector<qtnh::wire> rotate_swaps(std::size_t n, int d);
  }
}

#endif