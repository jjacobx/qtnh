#ifndef __CON_BASE__
#define __CON_BASE__

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
}

#endif