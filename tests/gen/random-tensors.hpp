#include "contraction-validation.hpp"

namespace randt {
  const contraction_validation v1 {
    tensor_info {{ 2, 2 }, { 1.0, 2.0, 3.0, 4.0 }}, 
    tensor_info {{ 2, 2 }, { 1.0, 2.0, 3.0, 4.0 }}, 
    tensor_info {{ 2, 2 }, { 7.0, 15.0, 10.0, 22.0 }}, 

    std::vector<qtnh::wire> {{ 0, 1 }}
  };
}
