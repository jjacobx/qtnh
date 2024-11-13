#ifndef __CONTRACT_GENERAL_TEMP
#define __CONTRACT_GENERAL_TEMP

#include "tensor/tensor.hpp"
#include "tensor/dense.hpp"
#include "tensor/symm.hpp"
#include "tensor/diag.hpp"

namespace qtnh {
  template<
    typename T1, 
    typename T2 = void, 
    typename = std::enable_if_t<std::is_base_of_v<Tensor, T1>>, 
    typename = std::enable_if_t<std::is_base_of_v<Tensor, T2> || std::is_void_v<T2>>
  >
  class ContractorBase {
    public:
      ContractorBase() = delete;
      ContractorBase(std::unique_ptr<T1> tp1, std::unique_ptr<T2> tp2, ConParams params)
        : tp1_(std::move(tp1)), tp2_(std::move(tp2)), params_(params) {}

      ContractorBase(const ContractorBase&) = delete;
      ContractorBase(ContractorBase&&) = default;
      ContractorBase& operator=(const ContractorBase&) = delete;
      ContractorBase& operator=(ContractorBase&&) = default;

      ~ContractorBase() = default;

    protected:
      std::unique_ptr<T1> tp1_;
      std::unique_ptr<T2> tp2_;
      ConParams params_;
  };

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
      using ContractorBase<T1, IdenTensor>::ContractorBase;
      ~PairContractor() = default;

      PairContractor(const PairContractor&) = delete;
      PairContractor(PairContractor&&) = default;
      PairContractor& operator=(const PairContractor&) = delete;
      PairContractor& operator=(PairContractor&&) = default;

      const ConParams& params() const noexcept { return this->params_; }

      qtnh::tptr contract();
  };

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

  
  template<> qtnh::tptr PairContractor<DenseTensor, DenseTensor>::contract();
  // template<> qtnh::tptr Contractor<DenseTensor, SymmTensor>::contract();
  // template<> qtnh::tptr Contractor<DenseTensor, DiagTensor>::contract();

  // template<> qtnh::tptr Contractor<SymmTensor, DenseTensor>::contract();
  // template<> qtnh::tptr Contractor<SymmTensor, SymmTensor>::contract();
  // template<> qtnh::tptr Contractor<SymmTensor, DiagTensor>::contract();

  // template<> qtnh::tptr Contractor<DiagTensor, DenseTensor>::contract();
  // template<> qtnh::tptr Contractor<DiagTensor, SymmTensor>::contract();
  // template<> qtnh::tptr Contractor<DiagTensor, DiagTensor>::contract();

  template<> qtnh::tptr SelfContractor<DenseTensor>::contract();
  // template<> qtnh::tptr Contractor<SymmTensor>::contract();
  // template<> qtnh::tptr Contractor<DiagTensor>::contract();

  template<typename T1, typename T2>
  qtnh::tptr PairContractor<T1, T2>::contract() {
    PairContractor<DenseTensor, DenseTensor> dcon (
      Tensor::convert<DenseTensor>(std::move(this->tp1_)), 
      Tensor::convert<DenseTensor>(std::move(this->tp2_)), 
      this->params_
    );

    auto tp_res = dcon.contract();
    this->params_ = dcon.params();

    return tp_res;
  }

  template<typename T>
  qtnh::tptr SelfContractor<T>::contract() {
    SelfContractor<DenseTensor> dcon (
      Tensor::toDense(Tensor::cast<DenseTensor>(std::move(this->tp1_))), 
      this->params_
    );

    auto tp_res = dcon.contract();
    this->params_ = dcon.params();

    return tp_res;
  }
}



#endif