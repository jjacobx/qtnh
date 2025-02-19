#include <iostream>

#include "core/utils.hpp"
#include "lalg/routines.hpp"

namespace qtnh {
  namespace lalg {
    std::tuple<BlockCyclicMatrix, cvec, BlockCyclicMatrix> PZGESVD(BlockCyclicMatrix&& m) {
      auto dims_m = m.totDims();
      auto chi = std::min(dims_m.first, dims_m.second);
  
      std::vector<double> sd(static_cast<std::size_t>(chi));
      BlockCyclicMatrix u(m.grid(), dims_m.first, chi, m.nBlock());
      BlockCyclicMatrix v(m.grid(), chi, dims_m.second, m.nBlock());

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
  }
}
