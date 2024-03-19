#include "core/typedefs.hpp"

struct tensor_info {
  qtnh::tidx_tup dims;
  std::vector<qtnh::tel> els;
};

struct contraction_validation {
  tensor_info t1_info;
  tensor_info t2_info;
  tensor_info t3_info;

  std::vector<qtnh::wire> wires;
};
