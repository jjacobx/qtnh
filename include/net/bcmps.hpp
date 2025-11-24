#ifndef QTNH_NET_BCMPS_HPP_INCLUDE
#define QTNH_NET_BCMPS_HPP_INCLUDE

#include "net/mps.hpp"
#include "util/ptuple.hpp"

namespace qtnh {
  class BCMPS {
    public:
      using chi_triple = std::array<std::size_t, 3>;
      using sample_t = std::vector<std::size_t>;

      BCMPS() = delete;
      BCMPS(const QTNHEnv& env, std::size_t n_sites, qtnh::tidx site_dim, chi_triple chis);
      BCMPS(const QTNHEnv& env, std::size_t n_sites, qtnh::tidx_tup site_dims, chi_triple chis);

      BCMPS(qtnh::tptr tp, chi_triple chis, SITE_CANON norm = SITE_CANON::left);
      BCMPS(std::vector<qtnh::tptr>&& sites);
      ~BCMPS() = default;

      static BCMPS rand(const QTNHEnv& env, std::size_t n_sites, qtnh::tidx site_dim, chi_triple chis, std::size_t bond_dim);
      BCMPS copy();

      const Tensor& site(std::size_t k) const { return *site_tensors_.at(k); }
      const std::vector<SITE_CANON>& siteCanons() const { return site_canons_; }
      const qtnh::tidx_tup& siteDims() const { return site_dims_; }
      const qtnh::tidx_tup& bondDims() const { return bond_dims_; }
      std::size_t nSites() const { return site_tensors_.size(); }

      constexpr std::size_t cycChi() const { return cyc_chi_; }
      constexpr std::size_t disChi() const { return dis_chi_; }
      constexpr std::size_t blkChi() const { return blk_chi_; }
      constexpr std::size_t locChi() const { return cyc_chi_ * blk_chi_; }
      constexpr std::size_t totChi() const { return cyc_chi_ * dis_chi_ * blk_chi_; }

      void apply(tptr_symm tp, std::vector<std::size_t> sites, bool update_dims = true);
      void apply(const MPO& mpo, std::size_t from, bool update_dims = true);

      // TODO: Remove. 
      void apply_old(const MPO& mpo, std::size_t from);
      
      qtnh::tel overlap(BCMPS& mps);
      qtnh::tel norm();

      void renormalise();
      void leftCanonicalise(std::size_t to);
      void rightCanonicalise(std::size_t to);

      void swap(std::size_t n);
      void permute(PTupleTar ptup);

      std::map<sample_t, std::size_t> sample(std::size_t from, std::size_t to, std::size_t n);
      std::unique_ptr<DenseTensor> toDense() &&;

      void print() const;

    private:
      std::vector<qtnh::tptr> site_tensors_;
      std::vector<SITE_CANON> site_canons_;
      qtnh::tidx_tup site_dims_;
      qtnh::tidx_tup bond_dims_;

      std::size_t cyc_chi_;
      std::size_t dis_chi_;
      std::size_t blk_chi_;
  };
}

#endif