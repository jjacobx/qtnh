#include "core/typedefs.hpp"

namespace qtnh {
  namespace lalg {
    extern "C" void sl_init_(int*, int*, int*);
    extern "C" void blacs_gridinfo_(int*, int*, int*, int*, int*);
    extern "C" void blacs_gridexit_(int*);

    // Call this to make processes independent. 
    extern "C" void blacs_get_(int*, int*, int*);
    extern "C" void blacs_gridmap_(int*, int*, int*, int*, int*, int*);
  }
}
