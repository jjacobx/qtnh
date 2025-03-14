#ifndef __TEN_CON_BASE__
#define __TEN_CON_BASE__

#include "ten/type/dense.hpp"
#include "ten/type/diag.hpp"
#include "ten/type/tensor.hpp"
#include "ten/type/symm.hpp"

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
        : tp1_(std::move(tp1)), tp2_(std::move(tp2)), params_(params) 
        {
          Tensor* op1 = tp1_.get();
          Tensor* op2 = tp2_.get();;
          if (op2 == nullptr) {
            op2 = tp1_.get();
          }

          // Validate contraction dimensions
          for (auto& w : params_.wires) {
            if (op1->totDims().at(w.first) != op2->totDims().at(w.second)) {
              throw std::invalid_argument("Incompatible contraction dimensions.");
            }
            if ((w.first < op1->disDims().size()) != (w.second < op2->disDims().size())) {
              throw std::invalid_argument("Cannot contract distributed and local dimensions.");
            }
          }
        }

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