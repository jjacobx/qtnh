#include <complex>

// ! Unused. 
// Might be useful to implement these.
namespace qtnh {
  struct CustomComplex {
    double real_ = 0.0;
    double imag_ = 0.0;

    double real() { return real_; }
    double imag() { return imag_; }
  };
} // namespace qtnh

namespace std {
  inline qtnh::CustomComplex conj(qtnh::CustomComplex c) {
    return qtnh::CustomComplex{c.real(), -c.imag()};
  }

  inline double abs(qtnh::CustomComplex c) {
    return c.real() * c.real() + c.imag() * c.imag();
  }
} // namespace std
