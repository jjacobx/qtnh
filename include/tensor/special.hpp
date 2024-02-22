#ifndef _TENSOR__SPECIAL_HPP
#define _TENSOR__SPECIAL_HPP

#include "core/typedefs.hpp"
#include "tensor/base.hpp"
#include "tensor/dense.hpp"

namespace qtnh {
  class SwapTensor : public SharedTensor {
    private:
      virtual Tensor* contract_disp(Tensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(SDenseTensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(DDenseTensor*, const std::vector<qtnh::wire>&) override;

    public:
      SwapTensor() = delete;
      SwapTensor(const SwapTensor&) = delete;
      SwapTensor(const QTNHEnv&, std::size_t, std::size_t);
      ~SwapTensor() = default;

    virtual qtnh::tel operator[](const qtnh::tidx_tup&) const override;

    virtual void swap(qtnh::tidx_tup_st, qtnh::tidx_tup_st) override;
  };

  class IdentityTensor : public SharedTensor {
    private:
      virtual Tensor* contract_disp(Tensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(SDenseTensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(DDenseTensor*, const std::vector<qtnh::wire>&) override;
      
    public:
      IdentityTensor() = delete;
      IdentityTensor(const IdentityTensor&) = delete;
      IdentityTensor(const QTNHEnv&, const qtnh::tidx_tup&);
      ~IdentityTensor() = default;

      virtual qtnh::tel operator[](const qtnh::tidx_tup&) const override;

      virtual void swap(qtnh::tidx_tup_st, qtnh::tidx_tup_st) override;
  };

  class ConvertTensor : public IdentityTensor {
    friend class SDenseTensor;
    friend class DDenseTensor;

    private:
      virtual Tensor* contract_disp(Tensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(SDenseTensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(DDenseTensor*, const std::vector<qtnh::wire>&) override;
      
    public:
      ConvertTensor() = delete;
      ConvertTensor(const ConvertTensor&) = delete;
      ConvertTensor(const QTNHEnv&, const qtnh::tidx_tup&);
      ~ConvertTensor() = default;
  };
}

#endif