#ifndef __NET_OPS__
#define __NET_OPS__

#include "net/mps.hpp"

namespace qtnh {
  namespace qops {
    tptr_symm x(const QTNHEnv& env);
    tptr_symm h(const QTNHEnv& env);
    MPO ca(const QTNHEnv& env, std::size_t n, std::vector<tel> els);

    MPO swap(const QTNHEnv& env, std::size_t n);
    MPO cmpo(const QTNHEnv& env, const MPO& mpo, std::size_t n);
    std::vector<qtnh::wire> rotate_swaps(std::size_t n, int d);

    MPO cmp(const QTNHEnv& env, std::size_t n);
    MPO icmp(const QTNHEnv& env, std::size_t n);
  }
}

#endif