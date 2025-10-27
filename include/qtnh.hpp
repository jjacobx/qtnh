#ifndef QTNH_QTNH_HPP_INCLUDED
#define QTNH_QTNH_HPP_INCLUDED

// Definitions below are for documentation only.
/// QTNH project namespace.
namespace qtnh {
  /// Namespace for helper functions.
  namespace utils {}
  /// Namespace for linear algebra routines.
  namespace lalg {}
} // namespace qtnh

// Export all usable header files.
#include "blas/matrix.hpp"       // IWYU pragma: export
#include "blas/wrappers.hpp"     // IWYU pragma: export
#include "net/bcmps.hpp"         // IWYU pragma: export
#include "net/mps.hpp"           // IWYU pragma: export
#include "net/network.hpp"       // IWYU pragma: export
#include "net/qops.hpp"          // IWYU pragma: export
#include "ten/con/base.hpp"      // IWYU pragma: export
#include "ten/con/pair-defs.hpp" // IWYU pragma: export
#include "ten/con/pair.hpp"      // IWYU pragma: export
#include "ten/con/self-defs.hpp" // IWYU pragma: export
#include "ten/con/self.hpp"      // IWYU pragma: export
#include "ten/dec/base.hpp"      // IWYU pragma: export
#include "ten/type/dense.hpp"    // IWYU pragma: export
#include "ten/type/diag.hpp"     // IWYU pragma: export
#include "ten/type/symm.hpp"     // IWYU pragma: export
#include "ten/type/tensor.hpp"   // IWYU pragma: export
#include "util/env.hpp"          // IWYU pragma: export
#include "util/indexing.hpp"     // IWYU pragma: export
#include "util/ops.hpp"          // IWYU pragma: export
#include "util/ptuple.hpp"       // IWYU pragma: export
#include "util/typedefs.hpp"     // IWYU pragma: export
#include "util/utils.hpp"        // IWYU pragma: export
#include "util/vector.hpp"       // IWYU pragma: export

#endif