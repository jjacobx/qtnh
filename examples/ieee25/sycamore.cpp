#include <chrono>
#include <iostream>
#include <random>
#include "qtnh.hpp"

using namespace qtnh;
using namespace std::chrono;
using namespace std::complex_literals;

constexpr auto SITE_DIM = 2UL;
constexpr auto N_SITES = 53UL;

const std::vector<std::size_t> init_sites_a {
  1, 4, 6, 9, 11, 13, 15, 17, 19, 21, 24, 26, 28, 30, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51
};

const std::vector<std::size_t> init_sites_b {
  2, 5, 7, 10, 12, 16, 18, 20, 22, 25, 27, 29, 31, 34, 36, 38, 42, 44, 48
};

const std::vector<std::size_t> init_sites_c {
  1, 4, 6, 9, 11, 13, 16, 18, 20, 22, 25, 27, 29, 31, 34, 36, 38, 40, 43, 45, 47, 49, 51
};

const std::vector<std::size_t> init_sites_d {
  2, 5, 7, 10, 12, 14, 17, 19, 21, 23, 26, 28, 30, 32, 35, 37, 39, 42, 44, 48
};

const std::vector<std::size_t> mpo_init_sites_c {
  0, 3, 8, 17, 26, 35, 43, 49, 1, 6, 13, 21, 30, 39, 4, 11, 19, 28, 37, 45, 9, 23, 32
};

const std::vector<std::size_t> mpo_n_sites_c {
  3, 5, 7, 9, 9, 8, 6, 4, 5, 7, 8, 9, 9, 8, 7, 8, 9, 9, 8, 6, 8, 9, 9
};

const std::vector<std::size_t> mpo_init_sites_d {
  2, 7, 14, 22, 31, 5, 12, 20, 29, 38, 10, 18, 27, 36, 44, 16, 25, 34, 42, 48
};

const std::vector<std::size_t> mpo_n_sites_d {
  5, 7, 8, 9, 9, 7, 8, 9, 9, 8, 8, 9, 9, 8, 6, 9, 9, 8, 6, 4
};

const std::vector<std::vector<std::size_t>> pattern_swap {
  init_sites_a, 
  init_sites_b, 
  init_sites_c, 
  init_sites_d, 
  init_sites_c, 
  init_sites_d, 
  init_sites_a, 
  init_sites_b
};

const std::vector<std::vector<std::size_t>> pattern_mpo {
  init_sites_a, 
  init_sites_b, 
  mpo_init_sites_c, 
  mpo_init_sites_d, 
  mpo_init_sites_c, 
  mpo_init_sites_d, 
  init_sites_a, 
  init_sites_b
};

PTupleTar ptup_ab_cd({
               16, 
            9, 17, 25, 
        4, 10, 18, 26, 34, 
     1, 5, 11, 19, 27, 35, 
  0, 2, 6, 12, 20, 28, 36, 42, 47, 
     3, 7, 13, 21, 29, 37, 43, 48, 51, 
        8, 14, 22, 30, 38, 44, 49, 52, 
           15, 23, 31, 39, 45, 50, 
               24, 32, 40, 46, 
                   33, 41
});

auto rand_gate(std::size_t i) {
  switch(i) {
    case 0:
      return qops::sqrt_x;
    case 1:
      return qops::sqrt_y;
    case 2:
      return qops::sqrt_w;
    default:
      throw std::invalid_argument("Only three random gates available.");
  }
}

void permute(BCMPS& mps, PTupleTar ptup, bool update_dims) {
  auto tup = ptup.tup();
  auto start = high_resolution_clock::now();

  auto swapped_count = 0UL;
  auto print_frequency = 20UL;

  for (auto i = 0UL; i < tup.size() - 1; ++i) {
    if (tup.at(i) > tup.at(i + 1)) {
      ++swapped_count;
      
      mps.leftCanonicalise(i + 1);
      mps.rightCanonicalise(i);

      mps.swap(i, update_dims);
      std::swap(tup.at(i), tup.at(i + 1));
      
      if (swapped_count % 20 == 0 && utils::is_root()) {
        auto stop = high_resolution_clock::now();
        auto delta = duration_cast<milliseconds>(stop - start);

        std::cout << "Swapped " << print_frequency << "(" << swapped_count << 
          ") sites (" << delta.count() << " ms)" << std::endl;

        start = high_resolution_clock::now();
      }

      i = 0UL;
    }
  }
}

void rcs_swap(const QTNHEnv& env, BCMPS& mps, std::size_t d, bool update_dims = true) {
  std::mt19937 gen(2025);
  std::uniform_int_distribution<std::size_t> dist02(0, 2);

  for (auto i = 0UL; i < d; ++i) {
    if (utils::is_root()) {
      std::cout << "\nIteration " << i + 1 << "/" << d << std::endl;
    }

    // mps.leftCanonicalise(NSITES - 1);

    if (i % 8 == 2) {
      if (utils::is_root()) {
        std::cout << "Permuting AB -> CD..." << std::endl;
      }

      auto start = high_resolution_clock::now();

      permute(mps, ptup_ab_cd, update_dims);

      auto stop = high_resolution_clock::now();
      auto delta = duration_cast<milliseconds>(stop - start);

      auto norm = mps.norm();
      if (utils::is_root()) {
        std::cout << "Done (" << delta.count() << " ms)" << std::endl;
        std::cout << "norm = " << norm << "\n";
      }
    } else if (i % 8 == 6) {
      if (utils::is_root()) {
        std::cout << "Permuting CD -> AB..." << std::endl;
      }

      auto start = high_resolution_clock::now();

      permute(mps, ptup_ab_cd.inv(), update_dims);

      auto stop = high_resolution_clock::now();
      auto delta = duration_cast<milliseconds>(stop - start);

      auto norm = mps.norm();
      if (utils::is_root()) {
        std::cout << "Done (" << delta.count() << " ms)" << std::endl;
        std::cout << "norm = " << norm << "\n";
      }
    }

    if (utils::is_root()) {
      std::cout << "Applying random gates..." << std::endl;
    }

    auto start = high_resolution_clock::now();
    
    for (auto j = 0UL; j < N_SITES; ++j) {
      auto num = dist02(gen);
      mps.apply(rand_gate(num)(env), { j });
    }

    auto stop = high_resolution_clock::now();
    auto delta = duration_cast<milliseconds>(stop - start);

    auto targets = pattern_swap.at(i % 8);

    if (utils::is_root()) {
      std::cout << "Done (" << delta.count() << " ms)" << std::endl;
      std::cout << "Applying entangling gates..." << std::endl;
      std::cout << "Targets: " << targets << std::endl;
    }

    auto counter = 1UL;
    for (auto& j : targets) {
      auto start = high_resolution_clock::now();

      mps.leftCanonicalise(j + 1);
      mps.rightCanonicalise(j);

      auto checkpoint = high_resolution_clock::now();

      mps.apply(qops::fsim_tp(env), { j, j + 1 }, update_dims);

      auto stop = high_resolution_clock::now();
      auto delta12 = duration_cast<milliseconds>(checkpoint - start);
      auto delta23 = duration_cast<milliseconds>(stop - checkpoint);
      auto delta13 = duration_cast<milliseconds>(stop - start);

      if (utils::is_root()) {
        std::cout << counter++ << "/" << targets.size() << " done (" << 
          delta13.count() << " ms = " << delta12.count() << " + " << 
          delta23.count() << " ms)" << std::endl;
      }
    }

    auto norm = mps.norm();
    if (utils::is_root()) {
      std::cout << "bonds = " << mps.bondDims() << "\n";
      std::cout << "norm = " << norm << "\n";
    }
  }
}

void rcs_mpo(const QTNHEnv& env, BCMPS& mps, std::size_t d, bool update_dims = true) {
  std::mt19937 gen(2025);
  std::uniform_int_distribution<std::size_t> dist02(0, 2);

  for (auto i = 0UL; i < d; ++i) {
    if (utils::is_root()) {
      std::cout << "\nIteration " << i + 1 << "/" << d << std::endl;
      std::cout << "Applying random gates..." << std::endl;
    }

    auto start = high_resolution_clock::now();
    
    for (auto j = 0UL; j < N_SITES; ++j) {
      auto num = dist02(gen);
      mps.apply(rand_gate(num)(env), { j });
    }

    auto stop = high_resolution_clock::now();
    auto delta = duration_cast<milliseconds>(stop - start);

    auto targets = pattern_mpo.at(i % 8);

    if (utils::is_root()) {
      std::cout << "Done (" << delta.count() << " ms)" << std::endl;
      std::cout << "Applying entangling gates..." << std::endl;
      std::cout << "Targets: " << targets << std::endl;
    }

    auto counter = 1UL;
    for (auto& j : targets) {
      auto start = high_resolution_clock::now();
      auto checkpoint = start;

      if (i % 8 < 2 || i % 8 > 5) {
        mps.leftCanonicalise(j + 1);
        mps.rightCanonicalise(j);

        checkpoint = high_resolution_clock::now();

        mps.apply(qops::fsim_tp(env), { j, j + 1 }, update_dims);
      } else {
        auto mpo_lenght = (i % 2 == 0) ? 
          mpo_n_sites_c.at(counter - 1) : 
          mpo_n_sites_d.at(counter - 1);

        MPO mpo = qops::fsim(env, mpo_lenght);
        mpo.rightCanonicalise();

        mps.leftCanonicalise(j + mpo_lenght - 1);
        mps.rightCanonicalise(j);

        checkpoint = high_resolution_clock::now();

        mps.apply(mpo, j, update_dims);
      }

      auto stop = high_resolution_clock::now();
      auto delta12 = duration_cast<milliseconds>(checkpoint - start);
      auto delta23 = duration_cast<milliseconds>(stop - checkpoint);
      auto delta13 = duration_cast<milliseconds>(stop - start);

      if (utils::is_root()) {
        std::cout << counter << "/" << targets.size() << " done (" << 
          delta13.count() << " ms = " << delta12.count() << " + " << 
          delta23.count() << " ms)" << std::endl;
      }

      ++counter;
    }

    auto norm = mps.norm();
    if (utils::is_root()) {
      std::cout << "bonds = " << mps.bondDims() << "\n";
      std::cout << "norm = " << norm << "\n";
    }
  }
}

int main(int argc, char* argv[]) {
  QTNHEnv env;
  
  auto DEPTH = 4UL;
  auto CHI_CYC = 4UL;
  auto CHI_DIS = 2UL;
  auto CHI_BLK = 2UL;

  enum class LongRangeMethod { SWAP, MPO };
  auto LONG_RANGE_METHOD = LongRangeMethod::SWAP;

  enum class DecMethod { SVD, QPD };
  auto DEC_METHOD = DecMethod::SVD;

  if (argc > 1) {
    DEPTH = static_cast<unsigned int>(strtol(argv[1], nullptr, 0));
  }
  if (argc > 4) {
    CHI_CYC = static_cast<unsigned int>(strtol(argv[2], nullptr, 0));
    CHI_DIS = static_cast<unsigned int>(strtol(argv[3], nullptr, 0));
    CHI_BLK = static_cast<unsigned int>(strtol(argv[4], nullptr, 0));
  }
  if (argc > 5) {
    std::string method(argv[5]);
    if (method == "SWAP") {
      LONG_RANGE_METHOD = LongRangeMethod::SWAP;
      if (utils::is_root()) std::cout << "Using SWAP long-range interaction method.\n";
    } else if (method == "MPO") {
      LONG_RANGE_METHOD = LongRangeMethod::MPO;
      if (utils::is_root()) std::cout << "Using MPO long-range interaction method.\n";
    } else if (utils::is_root()) {
      std::cout << "Unknown long-range interaction method, defaulting to SWAP.\n";
    }
  }
  if (argc > 6) {
    std::string method(argv[6]);
    if (method == "SVD") {
      DEC_METHOD = DecMethod::SVD;
      if (utils::is_root()) std::cout << "Using SVD decomposition method.\n";
    } else if (method == "QPD") {
      DEC_METHOD = DecMethod::QPD;
      if (utils::is_root()) std::cout << "Using QPD decomposition method.\n";
    } else if (utils::is_root()) {
      std::cout << "Unknown decomposition method, defaulting to SVD.\n";
    }
  }

  if (utils::is_root()) {
    std::cout << "Sycamore RCS with d = " << DEPTH << "\n";
    std::cout << "CHI = " << CHI_CYC * CHI_DIS * CHI_BLK << "\n";
  }
  
  // MPS mps(env, NROW * NCOL, SITE_DIM, { CHI_DIS, CHI_LOC });
  BCMPS mps(env, N_SITES, SITE_DIM, { CHI_CYC, CHI_DIS, CHI_BLK });

  utils::barrier();
  auto start = high_resolution_clock::now();

  auto update_dims = true;
  switch (DEC_METHOD) {
    case DecMethod::SVD:
      update_dims = true;
      break;
    case DecMethod::QPD:
      update_dims = false;
      break;
    default:
      utils::throw_unimplemented();
      break;
  }

  switch (LONG_RANGE_METHOD) {
    case LongRangeMethod::SWAP:
      rcs_swap(env, mps, DEPTH, update_dims);
      break;
    case LongRangeMethod::MPO:
      rcs_mpo(env, mps, DEPTH, update_dims);
      break;
    default:
      utils::throw_unimplemented();
      break;
  }

  utils::barrier();
  auto stop = high_resolution_clock::now();

  BCMPS zero_amp(env, N_SITES, SITE_DIM, { 1, 1, 1 });
  auto norm = mps.norm();
  auto bonds = mps.bondDims();

  mps.renormalise();
  auto amp0 = mps.overlap(zero_amp);
  auto delta = duration_cast<milliseconds>(stop - start);
  
  if (utils::is_root()) {
    std::cout << "|T| = " << norm << "\n";
    std::cout << "T[0] = " << amp0 << "\n";
    std::cout << "Max chis = " << bonds << "\n";
    std::cout << "Time taken: " << delta.count() << " ms\n";
  }
}
