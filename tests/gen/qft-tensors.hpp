#ifndef QFT_TENSORS_HPP 
#define QFT_TENSORS_HPP

#include "validation-primitives.hpp"

using namespace std::complex_literals;

namespace gen {
  tensor_info plus_state {{ 2 }, { std::pow(2, -.5), std::pow(2, -.5) }};
  tensor_info hadamard {{ 2, 2 }, { std::pow(2, -.5),  std::pow(2, -.5), std::pow(2, -.5), -std::pow(2, -.5) }};

  tensor_info cphase(double p) {
    return {{ 2, 2, 2, 2 }, { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, std::exp(1i * p)}};
  }
}

#endif