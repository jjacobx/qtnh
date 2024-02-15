#ifndef DENSE_TENSOR_HPP
#define DENSE_TENSOR_HPP

#include "tensor.hpp"

namespace qtnh {

  namespace ops {
    std::ostream& operator<<(std::ostream&, const Tensor&);
  }

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
      SDenseTensor(const QTNHEnv&, const qtnh::tidx_tup&, std::vector<qtnh::tel>, bool);
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
      virtual Tensor* contract_disp(Tensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(SDenseTensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(DDenseTensor*, const std::vector<qtnh::wire>&) override;

    public:
      DDenseTensor() = delete;
      DDenseTensor(const SDenseTensor&) = delete;
      DDenseTensor(const QTNHEnv&, const qtnh::tidx_tup&, std::vector<qtnh::tel>, qtnh::tidx_tup_st);
      ~DDenseTensor() = default;

      virtual std::optional<qtnh::tel> getEl(const qtnh::tidx_tup&) const override;
      virtual void setEl(const qtnh::tidx_tup&, qtnh::tel) override;
      virtual void setLocEl(const qtnh::tidx_tup&, qtnh::tel) override;
      virtual qtnh::tel& operator[](const qtnh::tidx_tup&) override;

      virtual void swap(qtnh::tidx_tup_st, qtnh::tidx_tup_st) override;

      void scatter(tidx_tup_st);
      void gather(tidx_tup_st);
      SDenseTensor share();

      void rep_all(std::size_t);
      void rep_each(std::size_t);
  };

  class SwapTensor : public Tensor {
    private:
      virtual Tensor* contract_disp(Tensor* t, const std::vector<qtnh::wire>& wires) override {
        t->swap(wires.at(0).first, wires.at(0).second); return t; }
      virtual Tensor* contract(SDenseTensor* t, const std::vector<qtnh::wire>& wires) override {
        t->swap(wires.at(0).first, wires.at(0).second); return t; }
      virtual Tensor* contract(DDenseTensor* t, const std::vector<qtnh::wire>& wires) override {
        t->swap(wires.at(0).first, wires.at(0).second); return t; }

    public:
      SwapTensor() = delete;
      SwapTensor(const SwapTensor&) = delete;
      SwapTensor(const QTNHEnv& env, std::size_t n1, std::size_t n2)
        : Tensor(env, { n1, n2, n2, n1 }) {}
      ~SwapTensor() = default;

    virtual std::optional<qtnh::tel> getEl(const qtnh::tidx_tup& idxs) const override {
      return getLocEl(idxs); }
    virtual std::optional<qtnh::tel> getLocEl(const qtnh::tidx_tup& idxs) const override {
      return (*this)[idxs]; }
    virtual qtnh::tel operator[](const qtnh::tidx_tup& idxs) const override {
      return (idxs.at(0) == idxs.at(3)) && (idxs.at(1) == idxs.at(2)); }

    virtual void swap(qtnh::tidx_tup_st, qtnh::tidx_tup_st) override {
      utils::throw_unimplemented(); return; }
  };

  
}

#endif