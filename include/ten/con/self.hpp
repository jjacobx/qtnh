#ifndef QTNH_TEN_CON_SELF_HPP_INCLUDE
#define QTNH_TEN_CON_SELF_HPP_INCLUDE

#include "ten/con/base.hpp"

namespace qtnh {
  template<typename T>
  class SelfContractor : ContractorBase<T> {
    public:
      SelfContractor() = delete;
      ~SelfContractor() = default;

      SelfContractor(std::unique_ptr<T> tp, ConParams params)
        : ContractorBase<T>(std::move(tp), nullptr, params) {}
      
      SelfContractor(const SelfContractor&) = delete;
      SelfContractor(SelfContractor&&) = default;
      SelfContractor& operator=(const SelfContractor&) = delete;
      SelfContractor& operator=(SelfContractor&&) = default;

      const ConParams& params() const noexcept { return this->params_; }

      qtnh::tptr contract();
  };
}

#endif