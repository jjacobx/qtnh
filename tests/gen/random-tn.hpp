#ifndef RANDOM_TN_HPP
#define RANDOM_TN_HPP

#include "contraction-validation.hpp"

using namespace std::complex_literals;

namespace gen {
  const tn_validation v1 {
    std::vector<tensor_info> {
      tensor_info {{ 2 }, { -0.60+0.40i,  0.10+0.50i }}, 
      tensor_info {{ 2, 2 }, { -0.30+0.60i, -0.60+0.30i, -0.40-0.50i, -0.30+0.60i }}
    }, 

    std::vector<bond_info> {
      bond_info { 0, 1, {}}
    }
  };

  const tn_validation v2 {
    std::vector<tensor_info> {
      tensor_info {{ 2, 2 }, {  0.50-0.80i,  0.60-1.00i,  0.00-0.20i, -0.70-0.70i }}, 
      tensor_info {{ 2, 2, 2, 2 }, {  0.10+0.00i, -0.10-0.30i,  0.80-0.60i,  0.90+0.50i,  0.00-0.20i, -0.50+0.20i,  0.90+0.30i, -1.00+0.60i,  0.90-0.40i, -0.80-0.70i,  0.70+0.90i,  0.10-0.40i, -0.60-0.50i,  0.30+0.10i,  0.00-0.70i,  0.70-0.20i }}
    }, 

    std::vector<bond_info> {
      bond_info { 0, 1, {{ 0, 2 }}}
    }
  };

  const tn_validation v3 {
    std::vector<tensor_info> {
      tensor_info {{ 2, 2, 2 }, {  0.90-0.80i, -0.20+0.00i,  0.30-0.90i, -1.00-0.10i,  0.40+0.50i, -0.70-0.40i,  0.10-0.70i,  0.40+0.60i }}, 
      tensor_info {{ 2, 2, 2 }, { -0.20+0.10i,  0.20-0.70i,  0.70+0.80i, -0.10-0.10i, -0.60+0.30i,  0.90-0.20i,  0.80+0.20i,  0.90-0.40i }}
    }, 

    std::vector<bond_info> {
      bond_info { 0, 1, {{ 0, 2 }, { 1, 0 }, }}
    }
  };

  const tn_validation v4 {
    std::vector<tensor_info> {
      tensor_info {{ 2, 2, 2, 2 }, {  0.90-0.40i, -0.60-0.50i,  0.70+0.80i,  0.70+0.10i, -0.80+0.90i,  0.20-0.70i,  0.80-0.40i, -0.50-0.40i,  0.80+0.40i,  0.10-0.40i,  0.80+0.50i,  0.50+0.90i, -0.90+0.30i,  0.60+0.30i,  0.60+0.80i,  0.60+0.00i }}, 
      tensor_info {{ 2, 2, 2, 2 }, {  0.30-0.50i, -0.50-0.10i,  0.30-0.80i, -0.80+0.50i,  0.60+0.30i, -0.10-0.10i, -0.40+0.70i, -0.70+0.80i,  0.30-0.10i,  0.10-0.30i, -0.30-0.20i, -1.00-0.70i, -1.00+0.90i,  0.90+0.30i,  0.80-0.40i,  0.70+0.50i }}
    }, 

    std::vector<bond_info> {
      bond_info { 0, 1, {{ 1, 0 }, { 2, 3 }, { 0, 2 }, }}
    }
  };

  const tn_validation v5 {
    std::vector<tensor_info> {
      tensor_info {{ 2 }, {  0.10+0.80i,  0.10-0.20i }}, 
      tensor_info {{ 2, 2, 2, 2 }, {  0.50-0.30i, -0.30+0.10i, -0.90-1.00i, -0.40+0.50i, -0.80+0.70i,  0.40+0.60i,  0.90-0.90i, -0.60-1.00i, -0.20-0.40i, -0.80+0.70i,  0.10-1.00i,  0.70-0.60i,  0.40-0.90i, -0.90+0.00i,  0.20+0.20i, -0.10-0.80i }}
    }, 

    std::vector<bond_info> {
      bond_info { 0, 1, {}}
    }
  };

}

#endif