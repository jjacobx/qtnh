#ifndef __CONTRACT_GENERAL_TEMP
#define __CONTRACT_GENERAL_TEMP

#include "tensor/tensor.hpp"
#include "tensor/dense.hpp"
#include "tensor/symm.hpp"
#include "tensor/diag.hpp"

namespace qtnh {
  template<class T1, class T2 = void>
  class Contractor {
    public:
      Contractor() = delete;
      Contractor(const Contractor&) = delete;

      Contractor(std::unique_ptr<T1> tp1, std::unique_ptr<T2> tp2, ConParams params)
        : tp1_(std::move(tp1)), tp2_(std::move(tp2)), params_(params) {}

      Contractor(Contractor&&) = default;
      ~Contractor() = default;

      Contractor& operator=(const Contractor&) = delete;
      Contractor& operator=(Contractor&&) = default;

      const ConParams& params() const noexcept { return params_; }

      virtual qtnh::tptr contract() {
        Contractor<DenseTensor, DenseTensor> dcon (
          Tensor::convert<DenseTensor>(std::move(tp1_)), 
          Tensor::convert<DenseTensor>(std::move(tp2_)), 
          params_
        );

        auto tp_res = dcon.contract();
        params_ = dcon.params();

        return tp_res;
      }

      protected:
        std::unique_ptr<T1> tp1_;
        std::unique_ptr<T2> tp2_;
        ConParams params_;
  };

  // template<class T1>
  // class Contractor<T1, IdenTensor> : public Contractor<T1, IdenTensor> {
  //   using Contractor<T1, IdenTensor>::Contractor;

  //   qtnh::tptr contract() override {
  //     return tp1_;
  //   }
  // };

  // template<class T2>
  // class Contractor<IdenTensor, T2> : public Contractor<IdenTensor, T2> {
  //   using Contractor<IdenTensor, T2>::Contractor;

  //   qtnh::tptr contract() override {
  //     return tp2_;
  //   }
  // };

  template<class T>
  class Contractor<T> {
    public:
      Contractor() = delete;
      Contractor(const Contractor&) = delete;

      Contractor(std::unique_ptr<T> tp, ConParams params)
        : tp_(std::move(tp)), params_(params) {}

      Contractor(Contractor&&) = default;
      ~Contractor() = default;

      Contractor& operator=(const Contractor&) = delete;
      Contractor& operator=(Contractor&&) = default;

      const ConParams& params() const noexcept { return params_; }

      qtnh::tptr contract() {
        Contractor<DenseTensor> dcon (
          Tensor::toDense(Tensor::cast<DenseTensor>(std::move(tp_))), 
          params_
        );

        auto tp_res = dcon.contract();
        params_ = dcon.params();

        return tp_res;
      }

      protected:
        std::unique_ptr<T> tp_;
        ConParams params_;
  };

  
  template<> qtnh::tptr Contractor<DenseTensor, DenseTensor>::contract();
  // template<> qtnh::tptr Contractor<DenseTensor, SymmTensor>::contract();
  // template<> qtnh::tptr Contractor<DenseTensor, DiagTensor>::contract();

  // template<> qtnh::tptr Contractor<SymmTensor, DenseTensor>::contract();
  // template<> qtnh::tptr Contractor<SymmTensor, SymmTensor>::contract();
  // template<> qtnh::tptr Contractor<SymmTensor, DiagTensor>::contract();

  // template<> qtnh::tptr Contractor<DiagTensor, DenseTensor>::contract();
  // template<> qtnh::tptr Contractor<DiagTensor, SymmTensor>::contract();
  // template<> qtnh::tptr Contractor<DiagTensor, DiagTensor>::contract();

  template<> qtnh::tptr Contractor<DenseTensor>::contract();
  // template<> qtnh::tptr Contractor<SymmTensor>::contract();
  // template<> qtnh::tptr Contractor<DiagTensor>::contract();
}

#endif