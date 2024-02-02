#include <map>
#include <optional>

//#include "coords.hpp"
#include "env.hpp"
#include "typedefs.hpp"

namespace qtnh {

  class Tensor {
    protected:
      inline static unsigned int counter = 0;

      unsigned int id;
      const QTNHEnv& env;
      const qtnh::tidx_tup& dims;

      bool active;

    public:
      Tensor() = delete;
      Tensor(const Tensor&) = delete;
      Tensor(const QTNHEnv& env, const qtnh::tidx_tup& dims)
        : id(++counter), env(env), dims(dims), active(true) {};
      ~Tensor() = default;

      unsigned int getID() const { return id; }
      const qtnh::tidx_tup& getLocDims() const { return dims; }
      const qtnh::tidx_tup& getDims() const { return dims; }

      virtual std::optional<qtnh::tel> getLocEl(const qtnh::tidx_tup& loc_idxs) const = 0;
      virtual std::optional<qtnh::tel> getGlobEl(const qtnh::tidx_tup& glob_idxs) const = 0;
      // virtual qtnh::tel operator[](const tidx_tup& loc_idxs) const = 0;

      virtual void swap(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) = 0;
      virtual Tensor& contract(const Tensor& t, const std::vector<qtnh::wire>& wires) const = 0;
      // void distribute(std::vector<qtnh::tidx_tup_st> idx_locs, std::vector<int> proc_ids);
  };
}
