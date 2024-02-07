#ifndef DENSE_TENSOR_HPP
#define DENSE_TENSOR_HPP

#include "tensor-new.hpp"

namespace qtnh {

  class DenseTensor : public Tensor {
    protected:
      std::vector<qtnh::tel> loc_els;
    
    public:
      DenseTensor() = delete;
      DenseTensor(const DenseTensor&) = delete;
      DenseTensor(const QTNHEnv& env, const qtnh::tidx_tup& dims, std::vector<qtnh::tel> els)
        : Tensor(env, dims), loc_els(els) {};
      ~DenseTensor() = default;

      virtual std::optional<qtnh::tel> getLocEl(const qtnh::tidx_tup&) const override;
      virtual qtnh::tel operator[](const qtnh::tidx_tup&) const override;

      virtual void setEl(const qtnh::tidx_tup&, qtnh::tel) = 0;
      virtual void setLocEl(const qtnh::tidx_tup&, qtnh::tel) = 0;
      virtual qtnh::tel& operator[](const qtnh::tidx_tup&) = 0;

      // const std::vector<qtnh::tel>& getLocEls() const { return loc_els; }
  };

  class SDenseTensor : public DenseTensor {
    private:
      virtual Tensor* contract_disp(Tensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(SDenseTensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(DDenseTensor*, const std::vector<qtnh::wire>&) override;
      
    public: 
      SDenseTensor() = delete;
      SDenseTensor(const SDenseTensor&) = delete;
      SDenseTensor(const QTNHEnv&, const qtnh::tidx_tup&, std::vector<qtnh::tel>);
      ~SDenseTensor() = default;

      virtual std::optional<qtnh::tel> getEl(const qtnh::tidx_tup&) const override;
      virtual void setEl(const qtnh::tidx_tup&, qtnh::tel) override;
      virtual void setLocEl(const qtnh::tidx_tup&, qtnh::tel) override;
      virtual qtnh::tel& operator[](const qtnh::tidx_tup&) override;

      virtual void swap(qtnh::tidx_tup_st, qtnh::tidx_tup_st) override;
      DDenseTensor distribute(tidx_tup_st);
  };

  class DDenseTensor : public DenseTensor {
    private:
      qtnh::tidx_tup_st n_dist_idxs;

      virtual Tensor* contract_disp(Tensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(SDenseTensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(DDenseTensor*, const std::vector<qtnh::wire>&) override;

    public:
      DDenseTensor() = delete;
      DDenseTensor(const SDenseTensor&) = delete;
      DDenseTensor(const QTNHEnv&, const qtnh::tidx_tup&, std::vector<qtnh::tel>, qtnh::tidx_tup_st);
      ~DDenseTensor() = default;

      // qtnh::tidx_tup_st getDistIdxs() const { return n_dist_idxs; }

      virtual std::optional<qtnh::tel> getEl(const qtnh::tidx_tup&) const override;
      virtual void setEl(const qtnh::tidx_tup&, qtnh::tel) override;
      virtual void setLocEl(const qtnh::tidx_tup&, qtnh::tel) override;
      virtual qtnh::tel& operator[](const qtnh::tidx_tup&) override;

      virtual void swap(qtnh::tidx_tup_st, qtnh::tidx_tup_st) override;
      // SDenseTensor share();

      void scatter(tidx_tup_st);
      // void gather(tidx_tup_st nidx);

      void rep_all(std::size_t);
      void rep_each(std::size_t);
  };
}

#endif