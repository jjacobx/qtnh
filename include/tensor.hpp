#ifndef TENSOR_NEW_HPP
#define TENSOR_NEW_HPP

#include <map>
#include <optional>

#include "env.hpp"
#include "typedefs.hpp"
#include "utils.hpp"

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

      virtual Tensor* contract_disp(Tensor*, const std::vector<qtnh::wire>&) {
        utils::throw_unimplemented(); return nullptr; }
      virtual Tensor* contract(SDenseTensor*, const std::vector<qtnh::wire>&) {
        utils::throw_unimplemented(); return nullptr; }
      virtual Tensor* contract(DDenseTensor*, const std::vector<qtnh::wire>&) {
        utils::throw_unimplemented(); return nullptr; }

    public:
      Tensor() = delete;
      Tensor(const Tensor&) = delete;
      Tensor(const QTNHEnv& env, const qtnh::tidx_tup& dims)
        : id(++counter), env(env), active(true), dims(dims), loc_dims(dims), dist_dims(qtnh::tidx_tup()) {};
      ~Tensor() = default;

      qtnh::uint getID() const { return id; }
      bool isActive() const { return active; };

      const qtnh::tidx_tup& getDims() const { return dims; }
      const qtnh::tidx_tup& getLocDims() const { return loc_dims; }
      const qtnh::tidx_tup& getDistDims() const { return dist_dims; }

      std::size_t getSize() const { return utils::dims_to_size(getDims()); }
      std::size_t getLocSize() const { return utils::dims_to_size(getLocDims()); }
      std::size_t getDistSize() const { return utils::dims_to_size(getDistDims()); }

      virtual std::optional<qtnh::tel> getEl(const qtnh::tidx_tup&) const = 0;
      virtual std::optional<qtnh::tel> getLocEl(const qtnh::tidx_tup&) const = 0;
      virtual qtnh::tel operator[](const qtnh::tidx_tup& loc_idxs) const = 0;

      virtual void swap(qtnh::tidx_tup_st, qtnh::tidx_tup_st) = 0;

      static Tensor* contract(Tensor* t1, Tensor* t2, const std::vector<qtnh::wire>& wires) { 
        return t2->contract_disp(t1, wires); }
  };
}

#endif