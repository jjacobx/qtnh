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

#include "con/base.hpp"
#include "con/pair.hpp"
#include "con/pair-defs.hpp"
#include "con/self.hpp"
#include "con/self-defs.hpp"

#include "core/env.hpp"
#include "core/typedefs.hpp"
#include "core/utils.hpp"

#include "lalg/matrix.hpp"
#include "lalg/routines.hpp"

#include "tensor/dense.hpp"
#include "tensor/diag.hpp"
#include "tensor/indexing.hpp"
#include "tensor/network.hpp"
#include "tensor/ptuple.hpp"
#include "tensor/symm.hpp"
#include "tensor/tensor.hpp"

#endif