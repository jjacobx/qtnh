#ifndef _TENSOR__BASE_HPP
#define _TENSOR__BASE_HPP

#include <optional>

#include "../core/env.hpp"
#include "../core/typedefs.hpp"

namespace qtnh {
  class SDenseTensor;
  class DDenseTensor;

  class Tensor {
    friend class SDenseTensor;
    friend class DDenseTensor;

    private:
      inline static qtnh::uint counter = 0;
      const qtnh::uint id;

    protected:
      const QTNHEnv& env;
      bool active;

      qtnh::tidx_tup dims;
      qtnh::tidx_tup loc_dims;
      qtnh::tidx_tup dist_dims;

      virtual Tensor* contract_disp(Tensor*, const std::vector<qtnh::wire>&);
      virtual Tensor* contract(SDenseTensor*, const std::vector<qtnh::wire>&);
      virtual Tensor* contract(DDenseTensor*, const std::vector<qtnh::wire>&);

    public:
      Tensor() = delete;
      Tensor(const Tensor&) = delete;
      Tensor(const QTNHEnv& env, const qtnh::tidx_tup& dims);
      ~Tensor() = default;

      qtnh::uint getID() const { return id; }
      bool isActive() const { return active; }
      const qtnh::tidx_tup& getDims() const { return dims; }
      const qtnh::tidx_tup& getLocDims() const { return loc_dims; }
      const qtnh::tidx_tup& getDistDims() const { return dist_dims; }

      std::size_t getSize() const;
      std::size_t getLocSize() const;
      std::size_t getDistSize() const;

      virtual std::optional<qtnh::tel> getEl(const qtnh::tidx_tup& idxs) const;
      virtual std::optional<qtnh::tel> getLocEl(const qtnh::tidx_tup& idxs) const;
      virtual qtnh::tel operator[](const qtnh::tidx_tup&) const = 0;

      virtual void swap(qtnh::tidx_tup_st, qtnh::tidx_tup_st) = 0;

      static Tensor* contract(Tensor* t1, Tensor* t2, const std::vector<qtnh::wire>& wires);
  };

  namespace ops {
    std::ostream& operator<<(std::ostream&, const Tensor&);
  }
}

#endif