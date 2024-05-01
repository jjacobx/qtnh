#ifndef CONTRACTION_VALIDATION_HPP
#define CONTRACTION_VALIDATION_HPP

#include "core/typedefs.hpp"

// validation-primitives.hpp

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

struct bond_info {
  std::size_t t1_idx;
  std::size_t t2_idx;
  std::vector<qtnh::wire> wires;
};

struct tn_validation {
  std::vector<tensor_info> t_infos;
  std::vector<bond_info> b_infos;

  // tensor_info result_info;
};

#endif