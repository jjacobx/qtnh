#ifndef __CON_PAIR__
#define __CON_PAIR__

#include "con/base.hpp"

namespace qtnh {
  template<typename T1, typename T2>
  class PairContractor : private ContractorBase<T1, T2> {
    public:
      using ContractorBase<T1, T2>::ContractorBase;
      ~PairContractor() = default;

      PairContractor(const PairContractor&) = delete;
      PairContractor(PairContractor&&) = default;
      PairContractor& operator=(const PairContractor&) = delete;
      PairContractor& operator=(PairContractor&&) = default;

      const ConParams& params() const noexcept { return this->params_; }

      qtnh::tptr contract();
  };

  template<typename T1>
  class PairContractor<T1, IdenTensor> : private ContractorBase<T1, IdenTensor> {
    public:
      using ContractorBase<T1, IdenTensor>::ContractorBase;
      ~PairContractor() = default;

      PairContractor(const PairContractor&) = delete;
      PairContractor(PairContractor&&) = default;
      PairContractor& operator=(const PairContractor&) = delete;
      PairContractor& operator=(PairContractor&&) = default;

      const ConParams& params() const noexcept { return this->params_; }

      qtnh::tptr contract();
  };

  template<typename T2>
  class PairContractor<IdenTensor, T2> : private ContractorBase<IdenTensor, T2> {
    public:
      using ContractorBase<IdenTensor, T2>::ContractorBase;
      ~PairContractor() = default;

      PairContractor(const PairContractor&) = delete;
      PairContractor(PairContractor&&) = default;
      PairContractor& operator=(const PairContractor&) = delete;
      PairContractor& operator=(PairContractor&&) = default;

      const ConParams& params() const noexcept { return this->params_; }

      qtnh::tptr contract();
  };
}

#endif