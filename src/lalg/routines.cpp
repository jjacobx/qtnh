#include <iostream>

#include "core/utils.hpp"
#include "lalg/routines.hpp"

namespace qtnh {
  namespace lalg {
    std::tuple<BlockCyclicMatrix, cvec, BlockCyclicMatrix> PZGESVD(BlockCyclicMatrix&& m) {
      auto dims_m = m.totDims();
      auto chi = std::min(dims_m.first, dims_m.second);
  
      std::vector<double> s(static_cast<std::size_t>(chi));
      BlockCyclicMatrix m_u(m.grid(), dims_m.first, chi, m.nBlock());
      BlockCyclicMatrix m_v(m.grid(), chi, dims_m.second, m.nBlock());

      auto one = 1;
      
      char jobu = 'V', jobvt = 'V';
      int lwork = -1, lrwork = -1;
      std::vector<qtnh::tel> work(1);
      std::vector<double> rwork(1);
      auto info = -1;

      if (m.grid().active()) {
        // Won't be modified, so must const cast. 
        // Needs two steps because descriptor is constexpr. 
        auto m_desc = m.descriptor(); auto m_desc_p = const_cast<int*>(m_desc.data());
        auto u_desc = m_u.descriptor(); auto u_desc_p = const_cast<int*>(u_desc.data());
        auto v_desc = m_v.descriptor(); auto v_desc_p = const_cast<int*>(v_desc.data());
        
        // Query size of the work array. 
        pzgesvd_(&jobu, &jobvt, &dims_m.first, &dims_m.second, m.data(), &one, &one, m_desc_p, 
                 s.data(), m_u.data(), &one, &one, u_desc_p, m_v.data(), &one, &one, v_desc_p, 
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

        pzgesvd_(&jobu, &jobvt, &dims_m.first, &dims_m.second, m.data(), &one, &one, m_desc_p, 
                 s.data(), m_u.data(), &one, &one, u_desc_p, m_v.data(), &one, &one, v_desc_p, 
                 work.data(), &lwork, rwork.data(), &info);
      }

      cvec s_c(static_cast<std::size_t>(chi));
      for (auto i = 0UL; i < s_c.size(); ++i) {
        s_c.at(i) = qtnh::tel(s.at(i));
      }

      // Moves to prevent copying large object. 
      // This is likely not suitable for NRVO. 
      return { std::move(m_u), std::move(s_c), std::move(m_v) };
    }
  }
}
