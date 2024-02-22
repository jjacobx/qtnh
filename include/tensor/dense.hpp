#ifndef _TENSOR__DENSE_HPP
#define _TENSOR__DENSE_HPP

#include "core/typedefs.hpp"
#include "tensor/base.hpp"

namespace qtnh {
  class DenseTensor : public WritableTensor {
    protected:
      std::vector<qtnh::tel> loc_els;
    
    public:
      DenseTensor() = delete;
      DenseTensor(const DenseTensor&) = delete;
      DenseTensor(std::vector<qtnh::tel>);
      ~DenseTensor() = default;

      virtual qtnh::tel operator[](const qtnh::tidx_tup&) const override;
      virtual qtnh::tel& operator[](const qtnh::tidx_tup&) override;
  };

  class SDenseTensor : public DenseTensor, public SharedTensor {
    private:
      virtual Tensor* contract_disp(Tensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(ConvertTensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(SDenseTensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(DDenseTensor*, const std::vector<qtnh::wire>&) override;
      
    public: 
      SDenseTensor() = delete;
      SDenseTensor(const SDenseTensor&) = delete;
      SDenseTensor(const QTNHEnv&, qtnh::tidx_tup, std::vector<qtnh::tel>);
      SDenseTensor(const QTNHEnv&, qtnh::tidx_tup, std::vector<qtnh::tel>, bool);
      ~SDenseTensor() = default;

      virtual void setEl(const qtnh::tidx_tup&, qtnh::tel) override;
      virtual void setLocEl(const qtnh::tidx_tup&, qtnh::tel) override;

      virtual void swap(qtnh::tidx_tup_st, qtnh::tidx_tup_st) override;
      DDenseTensor* distribute(tidx_tup_st);
  };

  class DDenseTensor : public DenseTensor {
    private:
      virtual Tensor* contract_disp(Tensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(ConvertTensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(SDenseTensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(DDenseTensor*, const std::vector<qtnh::wire>&) override;

    public:
      DDenseTensor() = delete;
      DDenseTensor(const SDenseTensor&) = delete;
      DDenseTensor(const QTNHEnv&, qtnh::tidx_tup, std::vector<qtnh::tel>, qtnh::tidx_tup_st);
      ~DDenseTensor() = default;

      virtual std::optional<qtnh::tel> getEl(const qtnh::tidx_tup&) const override;
      virtual std::optional<qtnh::tel> getLocEl(const qtnh::tidx_tup& idxs) const override;
      virtual void setEl(const qtnh::tidx_tup&, qtnh::tel) override;
      virtual void setLocEl(const qtnh::tidx_tup&, qtnh::tel) override;

      virtual void swap(qtnh::tidx_tup_st, qtnh::tidx_tup_st) override;

      void scatter(tidx_tup_st);
      void gather(tidx_tup_st);
      SDenseTensor* share();

      void rep_all(std::size_t);
      void rep_each(std::size_t);
  };


}

#endif