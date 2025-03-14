#ifndef QTNH_HPP
#define QTNH_HPP

/// QTNH project namespace. 
namespace qtnh {
  /// Namespace for tensor operators. It is recommended to use it directly to allow
  /// e.g. tensor printing. 
  namespace ops {}
  /// Namespace for helper functions. 
  namespace utils {}
  /// Namespace for linear algebra routines. 
  namespace lalg {}
}

#include "env.hpp"

#include "blas/matrix.hpp"
// #include "blas/routines.hpp" - skip imported Fortran routines
#include "blas/wrappers.hpp"

#include "net/network.hpp"

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

#include "ten/util/indexing.hpp"
#include "ten/util/ptuple.hpp"
#include "ten/util/typedefs.hpp"
#include "ten/util/utils.hpp"
#include "ten/util/vector.hpp"

#endif