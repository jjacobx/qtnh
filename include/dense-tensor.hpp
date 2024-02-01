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

      virtual std::optional<qtnh::tel> getLocEl(const qtnh::tidx_tup& loc_idxs) const override;
      void setLocEl(const qtnh::tidx_tup& loc_idxs, qtnh::tel el);

      virtual void setGlobEl(const qtnh::tidx_tup& glob_idxs, qtnh::tel el) = 0;

      // qtnh::tel& operator[](const qtnh::tidx_tup& idxs);
  };

  class SDenseTensor;
  class DDenseTensor;

  class SDenseTensor : public DenseTensor {
    public:
      SDenseTensor() = delete;
      SDenseTensor(const SDenseTensor&) = delete;
      SDenseTensor(const QTNHEnv& env, const qtnh::tidx_tup& dims, std::vector<qtnh::tel> els);
      ~SDenseTensor() = default;

      virtual std::optional<qtnh::tel> getGlobEl(const qtnh::tidx_tup& glob_idxs) const override;
      virtual void setGlobEl(const qtnh::tidx_tup& glob_idxs, qtnh::tel el) override;
      virtual void swap(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) override;
      virtual Tensor& contract(const Tensor& t, const std::vector<qtnh::wire>& wires) const override;

      DDenseTensor distribute(tidx_tup_st nidx);
  };

  class DDenseTensor : public DenseTensor {
    private:
      qtnh::tidx_tup_st n_dist_idxs;
      
    public:
      DDenseTensor() = delete;
      DDenseTensor(const SDenseTensor&) = delete;
      DDenseTensor(const QTNHEnv& env, const qtnh::tidx_tup& dims, std::vector<qtnh::tel> els, qtnh::tidx_tup_st n_dist_idxs);
      ~DDenseTensor() = default;

      virtual std::optional<qtnh::tel> getGlobEl(const qtnh::tidx_tup& glob_idxs) const override;
      virtual void setGlobEl(const qtnh::tidx_tup& glob_idxs, qtnh::tel el) override;
      virtual void swap(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) override;
      virtual Tensor& contract(const Tensor& t, const std::vector<qtnh::wire>& wires) const override;

      // SDenseTensor share();
      // void scatter(tidx_tup_st nidx);
      // void gather(tidx_tup_st nidx);
  };
}
