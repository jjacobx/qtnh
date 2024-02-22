#ifndef _TENSOR__SPECIAL_HPP
#define _TENSOR__SPECIAL_HPP

#include "../core/typedefs.hpp"
#include "tensor/base.hpp"
#include "tensor/dense.hpp"

namespace qtnh {
    class SwapTensor : public SharedTensor {
    private:
      virtual Tensor* contract_disp(Tensor* t, const std::vector<qtnh::wire>& wires) override;
      virtual Tensor* contract(SDenseTensor* t, const std::vector<qtnh::wire>& wires) override;
      virtual Tensor* contract(DDenseTensor* t, const std::vector<qtnh::wire>& wires) override;

    public:
      SwapTensor() = delete;
      SwapTensor(const SwapTensor&) = delete;
      SwapTensor(const QTNHEnv& env, std::size_t n1, std::size_t n2);
      ~SwapTensor() = default;

    virtual qtnh::tel operator[](const qtnh::tidx_tup& idxs) const override;

    virtual void swap(qtnh::tidx_tup_st, qtnh::tidx_tup_st) override;
  };

  class IdentityTensor : public SharedTensor {
    private:
      virtual Tensor* contract_disp(Tensor* t, const std::vector<qtnh::wire>& wires) override;
      virtual Tensor* contract(SDenseTensor* t, const std::vector<qtnh::wire>& wires) override;
      virtual Tensor* contract(DDenseTensor* t, const std::vector<qtnh::wire>& wires) override;
      
    public:
      IdentityTensor() = delete;
      IdentityTensor(const IdentityTensor&) = delete;
      IdentityTensor(const QTNHEnv& env, const qtnh::tidx_tup& in_dims);
      ~IdentityTensor() = default;

      virtual qtnh::tel operator[](const qtnh::tidx_tup& idxs) const override;

      virtual void swap(qtnh::tidx_tup_st, qtnh::tidx_tup_st) override;
  };

  class DistributeTensor : public IdentityTensor {
    private:
      virtual Tensor* contract_disp(Tensor* t, const std::vector<qtnh::wire>& wires) override;
      virtual Tensor* contract(SDenseTensor* t, const std::vector<qtnh::wire>& wires) override;
      virtual Tensor* contract(DDenseTensor* t, const std::vector<qtnh::wire>& wires) override;
      
    public:
      DistributeTensor() = delete;
      DistributeTensor(const DistributeTensor&) = delete;
      DistributeTensor(const QTNHEnv& env, const qtnh::tidx_tup& in_dims);
      ~DistributeTensor() = default;
  };
}

#endif