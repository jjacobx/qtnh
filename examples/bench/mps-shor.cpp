#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include "qtnh.hpp"

using namespace qtnh;
using namespace std::chrono;
using namespace std::complex_literals;

// m-bit phase estimation of modular exponentiation of 2^q mod 2^n - 1. 
void order_finding(MPS& mps, std::size_t m, std::size_t n, std::size_t q) {
  auto& env = mps.site(0).bc().env();

  mps.apply(qops::x(env), { m + n - 1 });
  for (auto i = 0UL; i < m; ++i) {
    mps.apply(qops::h(env), { m - i - 1 });

    auto swap_tars = qops::rotate_swaps(n, q * (1 << i));
    for (auto [a, b] : swap_tars) {
      auto swap_mpo = qops::swap(env, b - a + 1);
      auto cswap_mpo = qops::cmpo(env, swap_mpo, i + b + 2);
      cswap_mpo.rightCanonicalise();

      mps.leftCanonicalise(m + n - 1);
      mps.rightCanonicalise(m - i - 1);
      mps.apply(cswap_mpo, m - i - 1);
    }
  }

  // IQFT. 
  for (auto i = 0UL; i < m; ++i) {
    if (i > 0) {
      MPO mpo = qops::cmp(env, i + 1, -1.0);
      mpo.rightCanonicalise();

      mps.leftCanonicalise(m - i - 1);
      mps.rightCanonicalise(m - i - 1);
      mps.apply(mpo, m - i - 1);
    }

    mps.apply(qops::h(env), { m - i - 1 });
  }
}

int main(int argc, char* argv[]) {
  constexpr auto SITE_DIM = 2UL;
  
  auto M = 2UL;
  auto N = 3UL;
  auto Q = 1UL;
  auto CHI_DIS = 2UL;
  auto CHI_LOC = 8UL;

  if (argc > 2) {
    M = static_cast<unsigned int>(strtol(argv[1], nullptr, 0));
    N = static_cast<unsigned int>(strtol(argv[2], nullptr, 0));
  }
  if (argc > 3) {
    Q = static_cast<unsigned int>(strtol(argv[3], nullptr, 0));
  }
  if (argc > 4) {
    CHI_DIS = static_cast<unsigned int>(strtol(argv[4], nullptr, 0));
    CHI_LOC = static_cast<unsigned int>(strtol(argv[5], nullptr, 0));
  }

  QTNHEnv env;
  MPS mps(env, M + N, SITE_DIM, { CHI_DIS, CHI_LOC });

  utils::barrier();
  auto start = high_resolution_clock::now();

  order_finding(mps, M, N, Q);

  utils::barrier();
  auto stop = high_resolution_clock::now();

  MPS zero_amp(env, M + N, SITE_DIM, { 1, 1 });
  auto norm = mps.norm();
  auto amp0 = mps.overlap(zero_amp);
  auto delta = duration_cast<milliseconds>(stop - start);
  
  if (utils::is_root()) {
    std::cout << "|T| = " << norm << "\n";
    std::cout << "T[0] = " << amp0 << "\n";
    std::cout << "Bond dims = " << mps.bondDims() << "\n";
    std::cout << "Time taken: " << delta.count() << " ms\n";
  }

  auto R = N / std::gcd(N, Q);

  constexpr auto N_SAMP = 1000UL;

  auto samples = mps.sample(0, M, N_SAMP);
  auto flipped_samples = utils::flip_map(samples);

  for (auto& [n, s] : flipped_samples) {
    // auto res = 0UL;
    // auto n_bits = s.size();
    // for (auto i = 0UL ; i < n_bits; ++i) {
    //   res += std::size_t(std::pow(2, i)) * s.at(n_bits - i - 1);
    // }

    // auto up_reg = res / (1 << N);
    // auto dn_reg = res % (1 << N);
    // if (utils::is_root()) {
    //   std::cout << "|" << up_reg << ">|" << dn_reg << ">: " << n << "\n";
    // }

    auto frac = 0.0;
    auto n_bits = s.size();
    for (auto i = 0UL ; i < n_bits; ++i) {
      frac += std::pow(2.0, -double(i) - 1) * s.at(i);
    }

    auto S = std::size_t(std::round(frac * R));
    auto diff = std::abs(frac - double(S) / R);
    auto abs = diff * R;

    if (utils::is_root()) {
      std::cout << std::fixed << std::setprecision(4) << 
        S << "/" << R << " ± " << diff << " (" << 
        abs << "): \t" << n << "\n";
    }
  }

  // mps.rightCanonicalise(M - 1);
  // const auto& bc = mps.site(0).bc();
  // std::vector<tel> els(CHI_LOC, 0);
  // if (utils::is_root()) els.at(0) = 1.0;
  // tptr tp_res = DenseTensor::make(bc.env(), { CHI_DIS }, { CHI_LOC }, std::move(els));

  // for (auto i = 0UL; i < M; ++i) {
  //   tptr tp_tmp = mps.site(i).copy();
  //   ConParams params({{ 0, 0 }, { i + 1, 3 }});
  //   pcon con(std::move(tp_res), std::move(tp_tmp), params);
  //   tp_res = con.contract();
  // }

  // tp_res = Tensor::fold(std::move(tp_res), { 0, M + 1 }, utils::binops::add_sq, 0.0);
  // tp_res->print_serial("Psi_M");

  if (M + N <= 5) {
    auto tp = std::move(mps).toDense();
    tp->print_serial("Psi");
  }
}