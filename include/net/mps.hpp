#ifndef __NET_MPS__
#define __NET_MPS__

#include "net/network.hpp"

namespace qtnh {
  enum class MPS_NORM {
    left, 
    right, 
    none
  };

  class MPS {
    public:
      using chi_pair = std::pair<std::size_t, std::size_t>;

      MPS() = delete;
      MPS(const QTNHEnv& env, std::size_t n_sites, qtnh::tidx site_dim, chi_pair chis);
      MPS(const QTNHEnv& env, std::size_t n_sites, qtnh::tidx_tup site_dims, chi_pair chis);
      MPS(qtnh::tptr tp, chi_pair chis, MPS_NORM norm = MPS_NORM::left);
      ~MPS() = default;

      const Tensor& at(std::size_t k) const { return *site_tensors_.at(k); }
      MPS_NORM norm(std::size_t k) const { return site_norms_.at(k); }

      qtnh::tidx_tup siteDims() const { return site_dims_; }
      std::size_t nSites() const { return site_tensors_.size(); }

      constexpr std::size_t disChi() const { return dis_chi_; }
      constexpr std::size_t locChi() const { return loc_chi_; }
      constexpr std::size_t totChi() const { return dis_chi_ * loc_chi_; }

      void apply(std::unique_ptr<SymmTensorBase> tp, 
                 std::vector<std::size_t> sites);
      
      qtnh::tel self_overlap();
      qtnh::tel overlap(MPS& mps);

      std::unique_ptr<DenseTensor> toDense() &&;

    private:
      std::vector<qtnh::tptr> site_tensors_;
      std::vector<MPS_NORM> site_norms_;
      qtnh::tidx_tup site_dims_;

      std::size_t dis_chi_;
      std::size_t loc_chi_;
  };
}

#endif