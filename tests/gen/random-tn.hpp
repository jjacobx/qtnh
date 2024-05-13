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
    }, 

    tensor_info {{ 2, 2, 2 }, { -0.06-0.48i,  0.24-0.42i,  0.44+0.14i, -0.06-0.48i, -0.33-0.09i, -0.21-0.27i,  0.21-0.25i, -0.33-0.09i }}
  };

  const tn_validation v2 {
    std::vector<tensor_info> {
      tensor_info {{ 2, 2 }, {  0.50-0.80i,  0.60-1.00i,  0.00-0.20i, -0.70-0.70i }}, 
      tensor_info {{ 2, 2, 2, 2 }, {  0.10+0.00i, -0.10-0.30i,  0.80-0.60i,  0.90+0.50i,  0.00-0.20i, -0.50+0.20i,  0.90+0.30i, -1.00+0.60i,  0.90-0.40i, -0.80-0.70i,  0.70+0.90i,  0.10-0.40i, -0.60-0.50i,  0.30+0.10i,  0.00-0.70i,  0.70-0.20i }}
    }, 

    std::vector<bond_info> {
      bond_info { 0, 1, {{ 0, 2 }}}
    }, 

    tensor_info {{ 2, 2, 2, 2 }, { -0.07-0.24i, -0.19-0.25i, -0.10-0.28i,  0.03+0.70i,  0.31-1.06i, -1.04+0.27i, -0.84+0.23i,  0.19-0.33i, -0.92-0.24i, -0.64-1.06i, -0.62-0.96i,  1.02+0.90i,  0.28-2.26i, -1.53+0.59i, -1.35+0.79i, -0.35-0.59i }}
  };

  const tn_validation v3 {
    std::vector<tensor_info> {
      tensor_info {{ 2, 2, 2 }, {  0.90-0.80i, -0.20+0.00i,  0.30-0.90i, -1.00-0.10i,  0.40+0.50i, -0.70-0.40i,  0.10-0.70i,  0.40+0.60i }}, 
      tensor_info {{ 2, 2, 2 }, { -0.20+0.10i,  0.20-0.70i,  0.70+0.80i, -0.10-0.10i, -0.60+0.30i,  0.90-0.20i,  0.80+0.20i,  0.90-0.40i }}
    }, 

    std::vector<bond_info> {
      bond_info { 0, 1, {{ 0, 2 }, { 1, 0 }}}
    }, 

    tensor_info {{ 2, 2 }, {  0.37+0.05i,  1.51-1.26i,  0.73+0.61i, -0.29+0.05i }}
  };

  const tn_validation v4 {
    std::vector<tensor_info> {
      tensor_info {{ 2, 2, 2, 2 }, {  0.90-0.40i, -0.60-0.50i,  0.70+0.80i,  0.70+0.10i, -0.80+0.90i,  0.20-0.70i,  0.80-0.40i, -0.50-0.40i,  0.80+0.40i,  0.10-0.40i,  0.80+0.50i,  0.50+0.90i, -0.90+0.30i,  0.60+0.30i,  0.60+0.80i,  0.60+0.00i }}, 
      tensor_info {{ 2, 2, 2, 2 }, {  0.30-0.50i, -0.50-0.10i,  0.30-0.80i, -0.80+0.50i,  0.60+0.30i, -0.10-0.10i, -0.40+0.70i, -0.70+0.80i,  0.30-0.10i,  0.10-0.30i, -0.30-0.20i, -1.00-0.70i, -1.00+0.90i,  0.90+0.30i,  0.80-0.40i,  0.70+0.50i }}
    }, 

    std::vector<bond_info> {
      bond_info { 0, 1, {{ 1, 0 }, { 2, 3 }, { 0, 2 }}}
    }, 

    tensor_info {{ 2, 2 }, { -0.43-2.62i, -0.64+0.29i, -2.81-1.39i,  0.02+0.11i }}
  };

  const tn_validation v5 {
    std::vector<tensor_info> {
      tensor_info {{ 2 }, {  0.10+0.80i,  0.10-0.20i }}, 
      tensor_info {{ 2, 2, 2, 2 }, {  0.50-0.30i, -0.30+0.10i, -0.90-1.00i, -0.40+0.50i, -0.80+0.70i,  0.40+0.60i,  0.90-0.90i, -0.60-1.00i, -0.20-0.40i, -0.80+0.70i,  0.10-1.00i,  0.70-0.60i,  0.40-0.90i, -0.90+0.00i,  0.20+0.20i, -0.10-0.80i }}
    }, 

    std::vector<bond_info> {
      bond_info { 0, 1, {}}
    }, 

    tensor_info {{ 2, 2, 2, 2, 2 }, {  0.29+0.37i, -0.11-0.23i,  0.71-0.82i, -0.44-0.27i, -0.64-0.57i, -0.44+0.38i,  0.81+0.63i,  0.74-0.58i,  0.30-0.20i, -0.64-0.57i,  0.81-0.02i,  0.55+0.50i,  0.76+0.23i, -0.09-0.72i, -0.14+0.18i,  0.63-0.16i, -0.01-0.13i, -0.01+0.07i, -0.29+0.08i,  0.06+0.13i,  0.06+0.23i,  0.16-0.02i, -0.09-0.27i, -0.26+0.02i, -0.10+0.00i,  0.06+0.23i, -0.19-0.12i, -0.05-0.20i, -0.14-0.17i, -0.09+0.18i,  0.06-0.02i, -0.17-0.06i }}
  };

  const tn_validation v6 {
    std::vector<tensor_info> {
      tensor_info {{ 2, 2 }, {  0.00+0.70i, -0.20-0.70i,  0.90-0.60i, -0.70-0.80i }}, 
      tensor_info {{ 2, 2, 2, 2 }, { -0.90+0.40i, -0.80-0.60i,  0.30-0.80i, -0.20-0.10i, -1.00+0.90i, -0.30-0.10i, -0.90-0.10i, -0.10-0.70i, -0.60-0.50i, -0.80+0.10i, -0.70-0.40i, -1.00-0.90i,  0.10+0.90i,  0.50-0.80i, -0.70+0.10i, -0.20+0.80i }}, 
      tensor_info {{ 2, 2, 2 }, {  0.30-0.20i, -0.80+0.70i,  0.00+0.60i, -0.80+0.70i, -1.00+0.30i,  0.80-1.00i,  0.40-0.70i, -0.10+0.30i }}
    }, 

    std::vector<bond_info> {
      bond_info { 0, 1, {{ 1, 2 }}}, 
      bond_info { 1, 2, {{ 0, 2 }}}
    }, 

    tensor_info {{ 2, 2, 2, 2, 2 }, { -0.68+0.00i,  0.14-0.51i,  1.42+0.32i, -0.90+0.41i,  0.21-0.80i,  0.42-0.37i, -0.35+1.27i, -0.17-0.62i, -0.21-0.64i, -0.01-1.04i,  0.74+0.75i, -0.36+0.16i, -1.28+0.63i, -1.16+0.31i,  1.66-1.02i, -0.37+0.59i, -0.17-0.42i, -0.72-1.92i,  1.27-0.38i,  0.13+1.21i, -1.13-1.82i, -0.96-2.69i,  2.42+1.80i, -0.82+0.38i, -1.33+0.47i, -3.15-0.04i,  1.35-2.35i,  1.12+0.98i,  0.28+2.02i, -0.00+1.17i, -0.20-2.79i,  0.48+1.20i }}
  };

  const tn_validation v7 {
    std::vector<tensor_info> {
      tensor_info {{ 2, 2, 2 }, { -0.20+0.80i, -0.40-0.80i,  0.10-0.90i, -0.10-0.30i, -0.10-0.90i,  0.50+0.10i, -0.80+0.20i, -0.10+0.90i }}, 
      tensor_info {{ 2, 2, 2 }, { -0.10-0.90i,  0.90+0.80i,  0.40-0.10i, -0.20+0.60i, -0.10+0.70i,  0.20-0.90i, -0.50+0.40i, -0.40+0.60i }}, 
      tensor_info {{ 2, 2, 2 }, { -0.90+0.50i, -0.80-0.80i,  0.60+0.50i, -0.10-0.80i, -0.30+0.00i, -0.40+0.90i, -0.80+0.50i,  0.70-0.20i }}
    }, 

    std::vector<bond_info> {
      bond_info { 0, 1, {{ 2, 0 }, { 1, 2 }}}, 
      bond_info { 1, 2, {{ 1, 1 }}}
    }, 

    tensor_info {{ 2, 2, 2 }, { -0.96+2.81i, -1.55-1.94i, -2.01+0.17i,  1.03+2.36i,  0.79-1.93i,  0.25+1.36i,  1.58+0.37i, -0.74-1.66i }}
  };

  const tn_validation v8 {
    std::vector<tensor_info> {
      tensor_info {{ 2, 2, 2, 2 }, { -0.80+0.70i, -0.50+0.60i, -0.80-0.60i, -0.80-0.50i, -0.70+0.20i, -0.40+0.50i,  0.10-0.90i, -0.90-0.80i, -0.40+0.30i,  0.00+0.20i, -0.70+0.00i, -1.00+0.90i,  0.30-0.70i, -1.00+0.80i,  0.30-0.20i,  0.90+0.00i }}, 
      tensor_info {{ 2, 2, 2, 2 }, { -0.10+0.40i,  0.30-0.40i, -0.90-0.80i, -0.60-0.70i, -0.60+0.10i,  0.90+0.00i,  0.00+0.10i, -0.30+0.50i,  0.30+0.30i,  0.50-0.40i, -0.10-0.20i, -0.10+0.60i, -0.80+0.10i, -0.30+0.50i,  0.90+0.10i, -0.90-1.00i }}, 
      tensor_info {{ 2, 2, 2, 2 }, { -0.30-0.40i, -0.60+0.40i, -0.80+0.00i,  0.60+0.00i, -1.00+0.90i,  0.50-0.70i,  0.30+0.30i,  0.30-0.10i, -0.70-0.30i, -0.30-0.50i, -0.80-0.50i,  0.40-0.90i,  0.30+0.60i,  0.40+0.80i,  0.90+0.70i, -0.60+0.00i }}
    }, 

    std::vector<bond_info> {
      bond_info { 0, 1, {{ 1, 2 }, { 3, 1 }, { 0, 3 }}}, 
      bond_info { 1, 2, {{ 0, 0 }}}
    }, 

    tensor_info {{ 2, 2, 2, 2 }, { -1.05-0.81i,  0.14-0.64i, -0.83-0.90i,  1.28-1.39i,  0.49+1.69i, -0.39+1.21i,  1.00+1.56i, -0.95-0.62i,  1.74+1.89i, -0.43-0.50i,  1.46+0.74i, -3.38+1.78i,  0.50-4.99i,  1.67+0.11i, -1.03-2.74i,  0.64+1.99i }}
  };

  const tn_validation v9 {
    std::vector<tensor_info> {
      tensor_info {{ 2, 2 }, {  0.20+0.30i,  0.80+0.90i, -0.40-0.10i, -0.90-0.20i }}, 
      tensor_info {{ 2, 2, 2 }, { -0.90-0.50i,  0.30-0.80i, -0.90-0.10i,  0.30-0.30i,  0.70+0.60i, -0.60+0.80i, -0.80-0.20i,  0.30-0.20i }}, 
      tensor_info {{ 2, 2, 2, 2 }, {  0.30+0.60i, -0.60-0.60i, -0.50+0.80i, -0.50-0.50i,  0.40+0.30i, -0.30+0.70i,  0.80+0.00i,  0.10-0.30i,  0.90-0.80i, -0.20-1.00i, -0.50+0.60i,  0.60+0.00i, -0.60+0.50i,  0.70-1.00i, -0.20+0.00i,  0.10+0.30i }}
    }, 

    std::vector<bond_info> {
      bond_info { 0, 1, {{ 1, 0 }}}, 
      bond_info { 1, 2, {{ 1, 1 }}}
    }, 

    tensor_info {{ 2, 2, 2, 2, 2 }, { -0.34-0.44i,  1.45-0.51i, -1.08-1.31i, -0.04-0.30i,  1.53+1.07i, -0.86-0.35i, -0.32-0.14i,  0.28+0.14i, -0.10-0.30i,  0.29+0.88i,  0.88-0.62i,  0.56+0.28i, -1.20+0.95i,  0.75+0.42i,  0.32-0.58i, -0.53+0.20i,  0.45+0.26i, -0.75+0.93i,  1.24+0.41i,  0.15+0.03i, -1.35+0.04i,  0.84-0.42i,  0.13-0.02i, -0.16+0.12i,  0.09+0.15i, -0.49-0.50i, -0.37+0.72i, -0.39+0.06i,  0.37-1.04i, -0.52+0.17i,  0.03+0.41i,  0.19-0.30i }}
  };

  const tn_validation v10 {
    std::vector<tensor_info> {
      tensor_info {{ 2, 2, 2, 2 }, { -0.90-0.60i, -0.50-0.80i, -0.10+0.70i,  0.20-0.60i,  0.20-0.90i,  0.30-0.60i,  0.60-0.20i,  0.80-0.10i,  0.20+0.00i,  0.70-0.80i, -0.10-0.90i, -0.40-0.60i,  0.10+0.20i,  0.90-0.80i, -0.90+0.90i, -0.40+0.70i }}, 
      tensor_info {{ 2, 2, 2 }, {  0.90+0.90i,  0.40+0.40i,  0.20+0.60i, -1.00-0.40i,  0.60+0.40i, -0.60+0.50i,  0.00-0.80i,  0.90-0.20i }}, 
      tensor_info {{ 2, 2, 2 }, {  0.40-0.60i, -0.70-0.10i, -0.40+0.40i,  0.20-0.10i, -0.60-0.80i,  0.60+0.80i,  0.30+0.50i, -0.50-0.90i }}
    }, 

    std::vector<bond_info> {
      bond_info { 0, 1, {{ 1, 1 }, { 3, 0 }}}, 
      bond_info { 1, 2, {{ 2, 0 }}}
    }, 

    tensor_info {{ 2, 2, 2, 2 }, { -1.68-0.89i,  0.09+1.69i,  1.12+0.91i, -0.48-0.53i,  0.28-0.05i, -0.27+0.21i, -0.14+0.01i,  0.38-0.09i, -0.91-0.78i,  0.41+0.84i,  0.51+0.60i, -0.58-0.65i, -1.85-2.40i,  0.66+2.44i,  0.97+1.74i, -1.05-2.02i }}
  };

  const std::vector<tn_validation> tn2_vals { v1, v2, v3, v4, v5 };
  const std::vector<tn_validation> tn3_vals { v6, v7, v8, v9, v10 };
}

#endif