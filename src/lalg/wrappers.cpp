#include <iostream>

#include "core/utils.hpp"
#include "lalg/routines.hpp"
#include "lalg/wrappers.hpp"

namespace qtnh {
  namespace lalg {
    std::tuple<BlockCyclicMatrix, cvec, BlockCyclicMatrix> PZGESVD(BlockCyclicMatrix&& m) {
      auto dims_m = m.totDims();
      auto chi = std::min(dims_m.first, dims_m.second);
  
      std::vector<double> sd(static_cast<std::size_t>(chi));
      BlockCyclicMatrix u(m.grid(), dims_m.first, chi, m.blkDims().first, m.blkDims().second);
      BlockCyclicMatrix v(m.grid(), chi, dims_m.second, m.blkDims().first, m.blkDims().second);

      auto one = 1;
      
      char job_u = 'V', job_vt = 'V';
      int lwork = -1, lrwork = -1;
      std::vector<qtnh::tel> work(1);
      std::vector<double> rwork(1);
      auto info = -1;

      if (m.grid().active()) {
        // Won't be modified, so must const cast. 
        // Needs two steps because descriptor is constexpr. 
        auto desc_m = m.descSVD(); auto desc_mp = const_cast<int*>(desc_m.data());
        auto desc_u = u.descSVD(); auto desc_up = const_cast<int*>(desc_u.data());
        auto desc_v = v.descSVD(); auto desc_vp = const_cast<int*>(desc_v.data());
        
        // Query size of the work array. 
        pzgesvd_(&job_u, &job_vt, &dims_m.first, &dims_m.second, 
                 m.data(), &one, &one, desc_mp, sd.data(), 
                 u.data(), &one, &one, desc_up, 
                 v.data(), &one, &one, desc_vp, 
                 work.data(), &lwork, rwork.data(), &info);

        lwork = static_cast<int>(work.at(0).real());
        lrwork = static_cast<int>(rwork.at(0));
        work.resize(lwork);
        rwork.resize(lrwork);
        
        #ifdef DEBUG
          if (m.grid().procIdxs() == mtup { 0, 0 }) {
            std::cout << "LWORK = " << lwork << "\n";
            std::cout << "LRWORK = " << lrwork << "\n";
          }
       #endif

       pzgesvd_(&job_u, &job_vt, &dims_m.first, &dims_m.second, 
                m.data(), &one, &one, desc_mp, sd.data(), 
                u.data(), &one, &one, desc_up, 
                v.data(), &one, &one, desc_vp, 
                work.data(), &lwork, rwork.data(), &info);
      }

      cvec sc(static_cast<std::size_t>(chi));
      for (auto i = 0UL; i < sc.size(); ++i) {
        sc.at(i) = qtnh::tel(sd.at(i));
      }

      // Moves to prevent copying large object. 
      // This is likely not suitable for NRVO. 
      return { std::move(u), std::move(sc), std::move(v) };
    }

    BlockCyclicMatrix PZGEMM(BlockCyclicMatrix&& a, BlockCyclicMatrix&& b, bool use_bt) {
      auto dims_a = a.totDims();
      auto dims_b = b.totDims();

      if (use_bt) {
        dims_b = { dims_b.second, dims_b.first };
      }

      if (dims_a.second != dims_b.first) {
        throw std::invalid_argument("Incompatible matrices");
      }

      BlockCyclicMatrix c(a.grid(), dims_a.first, dims_b.second, 
                          a.blkDims().first, b.blkDims().second);

      auto one = 1;
      
      // Second matrix transposed to allow non-square process grids. 
      char trans_a = 'N', trans_b = use_bt ? 'T' : 'N';
      qtnh::tel alpha = 1, beta = 0;
      
      if (a.grid().active()) {
        // Won't be modified, so must const cast. 
        // Needs two steps because descriptor is constexpr. 
        auto desc_a = a.descSVD(); auto desc_ap = const_cast<int*>(desc_a.data());
        auto desc_b = b.descSVD(); auto desc_bp = const_cast<int*>(desc_b.data());
        auto desc_c = c.descSVD(); auto desc_cp = const_cast<int*>(desc_c.data());

        pzgemm_(&trans_a, &trans_b, &dims_a.first, &dims_b.second, &dims_a.second, &alpha, 
                a.data(), &one, &one, desc_ap, 
                b.data(), &one, &one, desc_bp, &beta, 
                c.data(), &one, &one, desc_cp);
      }

      return c;
    }
  }
}
