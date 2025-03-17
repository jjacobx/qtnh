#include <iostream>

#include "net/network.hpp"
#include "ten/type/tensor.hpp"
#include "util/typedefs.hpp"

namespace qtnh {
  template<typename T, typename U>
  std::ostream& operator<<(std::ostream& out, const std::pair<T, U>& p) {
    out << "(" << p.first << ", " << p.second << ")";
    return out;
  }

  template<typename T>
  std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
    for (std::size_t i = 0; i < v.size(); ++i) {
      out << v.at(i);
      if (i + 1 < v.size()) out << ", ";
    }

    return out;
  }

  template <typename T, std::size_t N>
  std::ostream& operator<<(std::ostream& out, const std::array<T, N>& o) {
    for (auto i = 0UL; i < N - 1; ++i) {
      out << o.at(i) << ", ";
    }

    out << o.at(N - 1);
    return out;
  }

  /// Print bond information via std::cout. 
  std::ostream& operator<<(std::ostream&, const TensorNetwork::Bond&);

  /// Print tensor elements via std::cout. 
  std::ostream& operator<<(std::ostream&, const Tensor&);
  std::ostream& operator<<(std::ostream&, const Broadcaster&);

  /// Print tensor index tuple via std::cout. 
  std::ostream& operator<<(std::ostream&, const qtnh::tidx_tup&);
  std::ostream& operator<<(std::ostream&, const TIFlag&);

  bool operator==(const BcParams& p1,const BcParams& p2);
}
