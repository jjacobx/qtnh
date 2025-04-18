#ifndef __NET_MPS__
#define __NET_MPS__

#include <map>
#include "net/network.hpp"

namespace qtnh {
  using tptr_symm = std::unique_ptr<SymmTensorBase>;

  enum class SITE_CANON {
    left, 
    right, 
    none
  };

  class MPO;

  class MPS {
    public:
      using chi_pair = std::pair<std::size_t, std::size_t>;
      using sample_t = std::vector<std::size_t>;

      MPS() = delete;
      MPS(const QTNHEnv& env, std::size_t n_sites, qtnh::tidx site_dim, chi_pair chis);
      MPS(const QTNHEnv& env, std::size_t n_sites, qtnh::tidx_tup site_dims, chi_pair chis);
      MPS(qtnh::tptr tp, chi_pair chis, SITE_CANON norm = SITE_CANON::left);
      ~MPS() = default;

      const Tensor& site(std::size_t k) const { return *site_tensors_.at(k); }
      const std::vector<SITE_CANON>& siteCanons(std::size_t k) const { return site_canons_; }
      const qtnh::tidx_tup& siteDims() const { return site_dims_; }
      const qtnh::tidx_tup& bondDims() const { return bond_dims_; }
      std::size_t nSites() const { return site_tensors_.size(); }

      constexpr std::size_t disChi() const { return dis_chi_; }
      constexpr std::size_t locChi() const { return loc_chi_; }
      constexpr std::size_t totChi() const { return dis_chi_ * loc_chi_; }

      void apply(tptr_symm tp, std::vector<std::size_t> sites);
      void apply(const MPO& mpo, std::size_t from);

      // TODO: Remove. 
      void apply_old(const MPO& mpo, std::size_t from);
      
      qtnh::tel overlap(MPS& mps);
      qtnh::tel norm();
      
      void renormalise();
      void leftCanonicalise(std::size_t to);
      void rightCanonicalise(std::size_t to);

      std::map<sample_t, std::size_t> sample(std::size_t from, std::size_t to, std::size_t n);
      std::unique_ptr<DenseTensor> toDense() &&;

      void print() const;

    private:
      std::vector<qtnh::tptr> site_tensors_;
      std::vector<SITE_CANON> site_canons_;
      qtnh::tidx_tup site_dims_;
      qtnh::tidx_tup bond_dims_;

      std::size_t dis_chi_;
      std::size_t loc_chi_;
  };

  class MPO {
    public:
      MPO() = delete;
      MPO(std::vector<qtnh::tptr>&& site_ops);
      ~MPO() = default;

      const Tensor& at(std::size_t k) const { return *site_ops_.at(k); }
      qtnh::tptr extract(std::size_t k) { return std::move(site_ops_.at(k)); }
      std::size_t nSites() const { return site_ops_.size(); }

      constexpr std::size_t pDim() const { return pdim_; }

      void rightCanonicalise();

      void print() const;

    private:
      std::vector<qtnh::tptr> site_ops_;
      std::size_t pdim_;
  };
}

#endif