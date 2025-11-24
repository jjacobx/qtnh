#ifndef QTNH_NET_QOPS_HPP_INCLUDE
#define QTNH_NET_QOPS_HPP_INCLUDE

#include "net/mps.hpp"

namespace qtnh {
  namespace qops {
    tptr_symm x(const QTNHEnv& env);
    tptr_symm h(const QTNHEnv& env);
    tptr_symm sqrt_x(const QTNHEnv& env);
    tptr_symm sqrt_y(const QTNHEnv& env);
    tptr_symm sqrt_w(const QTNHEnv& env);
    tptr_symm fsim_tp(const QTNHEnv& env, double phi = M_PI / 6);
    MPO ca(const QTNHEnv& env, std::size_t n, std::vector<tel> els);

    MPO swap(const QTNHEnv& env, std::size_t n);
    MPO fsim(const QTNHEnv& env, std::size_t n, double phi = M_PI / 6);
    MPO cmpo(const QTNHEnv& env, const MPO& mpo, std::size_t n);
    std::vector<qtnh::wire> rotate_swaps(std::size_t n, int d);

    MPO cmp(const QTNHEnv& env, std::size_t n, double mul = 1.0);
    MPO cmp_rev(const QTNHEnv& env, std::size_t n, double mul = -1.0);
  }
}

#endif