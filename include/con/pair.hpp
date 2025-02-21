#ifndef __CON_PAIR__
#define __CON_PAIR__

#include "con/base.hpp"

namespace qtnh {
  template<typename T1, typename T2>
  using keep_if_base = std::enable_if_t<std::is_abstract_v<T1> || std::is_abstract_v<T2>>;

  template<typename T>
  using rm_if_iden = std::enable_if_t<!std::is_same_v<IdenTensor, T>>;

  template<typename T1, typename T2, typename Enable>
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
      qtnh::tptr contract_scalapack();
  };

  template<typename T1, typename T2>
  class PairContractor<T1, T2, keep_if_base<T1, T2>> : private ContractorBase<T1, T2> {
    public:
      using ContractorBase<T1, T2>::ContractorBase;
      ~PairContractor() = default;

      PairContractor(const PairContractor&) = delete;
      PairContractor(PairContractor&&) = default;
      PairContractor& operator=(const PairContractor&) = delete;
      PairContractor& operator=(PairContractor&&) = default;

      const ConParams& params() const noexcept { return this->params_; }

      qtnh::tptr contract();
      qtnh::tptr contract_scalapack();
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
  class PairContractor<IdenTensor, T2, rm_if_iden<T2>> : private ContractorBase<IdenTensor, T2> {
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

  template<typename T1>
  class PairContractor<T1, SwapTensor, rm_if_iden<T1>> : private ContractorBase<T1, SwapTensor> {
    public:
      using ContractorBase<T1, SwapTensor>::ContractorBase;
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