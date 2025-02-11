#include "lalg/routines.hpp"

namespace qtnh {
  namespace lalg {
    std::tuple<BlockMatrix, cvec, BlockMatrix> PZGESVD(BlockMatrix&& m) {
      auto dims_m = m.totDims();
      auto chi = std::min(dims_m.first, dims_m.second);
  
      std::vector<double> s(static_cast<std::size_t>(chi));
      BlockMatrix m_u(m.grid(), dims_m.first, chi);
      BlockMatrix m_v(m.grid(), chi, dims_m.second);

      auto one = 1;
      
      // ! Work params might break for larger matrices. 
      char jobu = 'V', jobvt = 'V';
      tel work2[10000];
      int lwork2 = 10000;
      double rwork2[10000];
      auto info2 = -1;

      if (m.grid().active()) {
        // Won't be modified, so must const cast. 
        // Needs two steps because descriptor is constexpr. 
        auto m_desc = m.descriptor(); auto m_desc_p = const_cast<int*>(m_desc.data());
        auto u_desc = m_u.descriptor(); auto u_desc_p = const_cast<int*>(u_desc.data());
        auto v_desc = m_v.descriptor(); auto v_desc_p = const_cast<int*>(v_desc.data());

        pzgesvd_(&jobu, &jobvt, &dims_m.first, &dims_m.second, m.data(), &one, &one, m_desc_p, s.data(), 
                 m_u.data(), &one, &one, u_desc_p, m_v.data(), &one, &one, v_desc_p, 
                 work2, &lwork2, rwork2, &info2);
      }

      cvec s_c(static_cast<std::size_t>(chi));
      for (auto i = 0UL; i < s_c.size(); ++i) {
        s_c.at(i) = qtnh::tel(s.at(i));
      }

      // Moves to prevent copying large object. 
      // This is likely not suitable for NRVO. 
      return { std::move(m_u), std::move(s_c), std::move(m_v) };
    }

    std::tuple<BlockCyclicMatrix, cvec, BlockCyclicMatrix> PZGESVD(BlockCyclicMatrix&& m) {
      auto dims_m = m.totDims();
      auto chi = std::min(dims_m.first, dims_m.second);
  
      std::vector<double> s(static_cast<std::size_t>(chi));
      BlockCyclicMatrix m_u(m.grid(), dims_m.first, chi, m.nBlock());
      BlockCyclicMatrix m_v(m.grid(), chi, dims_m.second, m.nBlock());

      auto one = 1;
      
      // ! Work params might break for larger matrices. 
      char jobu = 'V', jobvt = 'V';
      tel work2[10000];
      int lwork2 = 10000;
      double rwork2[10000];
      auto info2 = -1;

      if (m.grid().active()) {
        // Won't be modified, so must const cast. 
        // Needs two steps because descriptor is constexpr. 
        auto m_desc = m.descriptor(); auto m_desc_p = const_cast<int*>(m_desc.data());
        auto u_desc = m_u.descriptor(); auto u_desc_p = const_cast<int*>(u_desc.data());
        auto v_desc = m_v.descriptor(); auto v_desc_p = const_cast<int*>(v_desc.data());

        pzgesvd_(&jobu, &jobvt, &dims_m.first, &dims_m.second, m.data(), &one, &one, m_desc_p, s.data(), 
                 m_u.data(), &one, &one, u_desc_p, m_v.data(), &one, &one, v_desc_p, 
                 work2, &lwork2, rwork2, &info2);
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
