#ifndef RANDOM_TENSORS_HPP
#define RANDOM_TENSORS_HPP

#include "contraction-validation.hpp"

using namespace std::complex_literals;

namespace gen {
  const contraction_validation v1 {
    tensor_info {{ 2, 2 }, {  0.20-0.80i, -0.90-0.40i, -0.50+0.70i,  0.50+0.00i }}, 
    tensor_info {{ 2, 2 }, { -0.80-0.40i, -0.60-0.20i, -0.20+0.40i, -0.10-0.30i }}, 
    tensor_info {{ 2, 2 }, { -0.04+0.24i,  0.54+0.32i,  0.26+0.58i,  0.29-0.43i }}, 

    std::vector<qtnh::wire>{{0, 1}}
  };

  const std::vector<contraction_validation> cvs { v1 };
}

#endif