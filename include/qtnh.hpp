#ifndef __QTNH__
#define __QTNH__

/// QTNH project namespace. 
namespace qtnh {
  /// Namespace for helper functions. 
  namespace utils {}
  /// Namespace for linear algebra routines. 
  namespace lalg {}
}

#include "blas/matrix.hpp"
// #include "blas/routines.hpp" - skip imported Fortran routines
#include "blas/wrappers.hpp"

#include "net/mps.hpp"
#include "net/network.hpp"
#include "net/qops.hpp"

#include "ten/con/base.hpp"
#include "ten/con/pair.hpp"
#include "ten/con/pair-defs.hpp"
#include "ten/con/self.hpp"
#include "ten/con/self-defs.hpp"

#include "ten/dec/base.hpp"

#include "ten/type/dense.hpp"
#include "ten/type/diag.hpp"
#include "ten/type/symm.hpp"
#include "ten/type/tensor.hpp"

#include "util/env.hpp"
#include "util/indexing.hpp"
#include "util/ops.hpp"
#include "util/ptuple.hpp"
#include "util/typedefs.hpp"
#include "util/utils.hpp"
#include "util/vector.hpp"

#endif