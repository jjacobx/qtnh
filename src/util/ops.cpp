#include <iomanip>

#include "util/ops.hpp"

namespace qtnh {
  template<>
  std::ostream& operator<<(std::ostream& out, const std::vector<qtnh::tel>& v) {
    for (std::size_t i = 0; i < v.size(); ++i) {
      auto el = v.at(i);
      auto real = el.real() < ZERO_TOL ? 0.0 : el.real();
      auto imag = el.imag() < ZERO_TOL ? 0.0 : el.imag();

      out << tel(real, imag);
      if (i + 1 < v.size()) out << ", ";
    }

    return out;
  }
  
  std::ostream& operator<<(std::ostream& out, const TensorNetwork::Bond& o) {
    out << "(" << o.tensor_ids.first << ", " << o.tensor_ids.second << "); ";
    out << "{";
    for (std::size_t i = 0; i < o.wires.size(); ++i) {
      out << "(" << o.wires.at(i).first << ", " << o.wires.at(i).second << ")";
      if (i < o.wires.size() - 1) out << ", ";
    }

    out << "}";
    return out;
  }

  std::ostream& operator<<(std::ostream& out, const Tensor& o) {
    if (!o.bc().isActive()) {
      out << "Inactive";
      return out;
    }

    out << std::setprecision(2);

    TIndexing ti(o.totDims());
    for (auto idxs : ti.tup()) {
      if (o.has(idxs)) {
        out << o.at(idxs);
        if ((utils::idxs_to_i(idxs, o.totDims()) + 1) % o.locSize() != 0) {
          out << ", ";
        }
      }
    }

    return out;
  }

  std::ostream& operator<<(std::ostream& out, const Broadcaster& o) {
    out << "Bcaster [base: " << o.base() << ", ";
    out << "params: { " << o.params().str << ", " << o.params().cyc << ", " << o.params().off;
    out << " }]";

    return out;
  }

  bool operator==(const BcParams& p1, const BcParams& p2) {
    return (p1.str == p2.str) && (p1.cyc == p2.cyc) && (p1.off == p2.off);
  }

  std::ostream& operator<<(std::ostream& out, const qtnh::tidx_tup& o) {
    out << "(";
    for (std::size_t i = 0; i < o.size(); i++) {
      out << o.at(i);
      if (i < o.size() - 1) {
        out << ", ";
      }
    }
    out << ")";

    return out;
  }

  std::ostream& operator<<(std::ostream& out, const TIFlag& o) {
    out << "(" << o.tag << ", " << o.label << ")";
    return out; 
  }
}
