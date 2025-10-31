#ifdef DEBUG
#include <iostream>
#endif

#include "blas/routines.hpp"
#include "blas/wrappers.hpp"

namespace qtnh {
  namespace lalg {
    std::tuple<BlockCyclicMatrix, cvec, BlockCyclicMatrix> PZGESVD(BlockCyclicMatrix&& m) {
      auto dims_m = m.totDims();
      auto chi = std::min(dims_m.first, dims_m.second);
  
      std::vector<double> sd(static_cast<std::size_t>(chi));
      BlockCyclicMatrix u(m.grid(), { dims_m.first, chi }, { m.blkDims().first, m.blkDims().second });
      BlockCyclicMatrix v(m.grid(), { chi, dims_m.second }, { m.blkDims().first, m.blkDims().second });

      auto one = 1;
      
      char job_u = 'V', job_vt = 'V';
      int lwork = -1, lrwork = -1;
      std::vector<qtnh::tel> work(1);
      std::vector<double> rwork(1);
      auto info = -1;

      if (m.grid().active()) {
        // Won't be modified, so must const cast. 
        // Needs two steps because descriptor is constexpr. 
        auto desc_m = m.desc9(); auto desc_mp = const_cast<int*>(desc_m.data());
        auto desc_u = u.desc9(); auto desc_up = const_cast<int*>(desc_u.data());
        auto desc_v = v.desc9(); auto desc_vp = const_cast<int*>(desc_v.data());
        
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
        sc.at(i) = qtnh::tel { sd.at(i), 0.0 };
      }

      // Moves to prevent copying large object. 
      // This is likely not suitable for NRVO. 
      return { std::move(u), std::move(sc), std::move(v) };
    }

    std::tuple<BlockCyclicMatrix, BlockCyclicMatrix> PZGEQRD(BlockCyclicMatrix&& m) {
      auto dims_m = m.totDims();
      auto chi = std::min(dims_m.first, dims_m.second);

      cvec tau(static_cast<std::size_t>(chi));
      
      auto one = 1;
      int lwork = -1;
      cvec work(1);
      auto info = -1;

      // ! Might have to initialise with zeros. 
      cvec r_els(m.locSize(), 0.0);
      BlockCyclicMatrix r(m.grid(), m.blkDims(), m.disDims(), m.cycDims(), std::move(r_els));

      if (m.grid().active()) {
        auto desc_m = m.desc9(); auto desc_mp = const_cast<int*>(desc_m.data());
        pzgeqrf_(&dims_m.first, &dims_m.second, 
                 m.data(), &one, &one, desc_mp, 
                 tau.data(), work.data(), &lwork, &info);
                
        lwork = static_cast<int>(work.at(0).real());
        work.resize(lwork);
        
        #ifdef DEBUG
          if (m.grid().procIdxs() == mtup { 0, 0 }) {
            std::cout << "LWORK = " << lwork << "\n";
          }
        #endif

        pzgeqrf_(&dims_m.first, &dims_m.second, 
                 m.data(), &one, &one, desc_mp, 
                 tau.data(), work.data(), &lwork, &info);
        
        char up = 'U';
        auto desc_r = r.desc9(); auto desc_rp = const_cast<int*>(desc_r.data());
        pzlacpy_(&up, &dims_m.first, &dims_m.second, 
                 m.data(), &one, &one, desc_mp, 
                 r.data(), &one, &one, desc_rp);
        
        pzungqr_(&dims_m.first, &dims_m.second, &chi, 
                 m.data(), &one, &one, desc_mp, 
                 tau.data(), work.data(), &lwork, &info);
        
        lwork = static_cast<int>(work.at(0).real());
        work.resize(lwork);

        #ifdef DEBUG
          if (m.grid().procIdxs() == mtup { 0, 0 }) {
            std::cout << "LWORK = " << lwork << "\n";
          }
        #endif

        pzungqr_(&dims_m.first, &dims_m.second, &chi, 
                 m.data(), &one, &one, desc_mp, 
                 tau.data(), work.data(), &lwork, &info);
      }

      cvec q_els = m.extractEls();
      BlockCyclicMatrix q(m.grid(), m.blkDims(), m.disDims(), m.cycDims(), std::move(q_els));

      return { std::move(q), std::move(r) };
    }

    std::tuple<BlockCyclicMatrix, BlockCyclicMatrix> PZGELQD(BlockCyclicMatrix&& m) {
      auto dims_m = m.totDims();
      auto chi = std::min(dims_m.first, dims_m.second);

      cvec tau(static_cast<std::size_t>(chi));
      
      auto one = 1;
      int lwork = -1;
      cvec work(1);
      auto info = -1;

      // ! Might have to initialise with zeros. 
      cvec l_els(m.locSize(), 0.0);
      BlockCyclicMatrix l(m.grid(), m.blkDims(), m.disDims(), m.cycDims(), std::move(l_els));

      if (m.grid().active()) {
        auto desc_m = m.desc9(); auto desc_mp = const_cast<int*>(desc_m.data());
        pzgelqf_(&dims_m.first, &dims_m.second, 
                 m.data(), &one, &one, desc_mp, 
                 tau.data(), work.data(), &lwork, &info);
                
        lwork = static_cast<int>(work.at(0).real());
        work.resize(lwork);
        
        #ifdef DEBUG
          if (m.grid().procIdxs() == mtup { 0, 0 }) {
            std::cout << "LWORK = " << lwork << "\n";
          }
        #endif

        pzgelqf_(&dims_m.first, &dims_m.second, 
                 m.data(), &one, &one, desc_mp, 
                 tau.data(), work.data(), &lwork, &info);
        
        char lo = 'L';
        auto desc_l = l.desc9(); auto desc_lp = const_cast<int*>(desc_l.data());
        pzlacpy_(&lo, &dims_m.first, &dims_m.second, 
                 m.data(), &one, &one, desc_mp, 
                 l.data(), &one, &one, desc_lp);
        
        pzunglq_(&dims_m.first, &dims_m.second, &chi, 
                 m.data(), &one, &one, desc_mp, 
                 tau.data(), work.data(), &lwork, &info);
        
        lwork = static_cast<int>(work.at(0).real());
        work.resize(lwork);

        #ifdef DEBUG
          if (m.grid().procIdxs() == mtup { 0, 0 }) {
            std::cout << "LWORK = " << lwork << "\n";
          }
        #endif

        pzunglq_(&dims_m.first, &dims_m.second, &chi, 
                 m.data(), &one, &one, desc_mp, 
                 tau.data(), work.data(), &lwork, &info);
      }

      cvec q_els = m.extractEls();
      BlockCyclicMatrix q(m.grid(), m.blkDims(), m.disDims(), m.cycDims(), std::move(q_els));

      return { std::move(l), std::move(q) };
    }

    BlockCyclicMatrix PZGEMM(BlockCyclicMatrix&& a, BlockCyclicMatrix&& b, bool use_at, bool use_bt) {
      auto blk_dims_a = a.blkDims(), blk_dims_b = b.blkDims();
      auto dis_dims_a = a.disDims(), dis_dims_b = b.disDims();
      auto cyc_dims_a = a.cycDims(), cyc_dims_b = b.cycDims();
      auto tot_dims_a = a.totDims(), tot_dims_b = b.totDims();
      auto& grid = a.grid();

      if (use_at) {
        std::swap(blk_dims_a.first, blk_dims_a.second);
        std::swap(dis_dims_a.first, dis_dims_a.second);
        std::swap(cyc_dims_a.first, cyc_dims_a.second);
        std::swap(tot_dims_a.first, tot_dims_a.second);
      }
      if (use_bt) {
        std::swap(blk_dims_b.first, blk_dims_b.second);
        std::swap(dis_dims_b.first, dis_dims_b.second);
        std::swap(cyc_dims_b.first, cyc_dims_b.second);
        std::swap(tot_dims_b.first, tot_dims_b.second);
      }
      
      // Basic checks for distributed matrix multiplication. 
      if (tot_dims_a.second != tot_dims_b.first) {
        throw std::invalid_argument("Incompatible matrix dimensions of A and B");
      }
      if (std::addressof(a.grid()) != std::addressof(b.grid())) {
        throw std::invalid_argument("Process grids of A and B are different");
      }

      BlockCyclicMatrix c(
        grid, 
        mtup { blk_dims_a.first, blk_dims_b.second }, 
        mtup { dis_dims_a.first, dis_dims_b.second }, 
        mtup { cyc_dims_a.first, cyc_dims_b.second }
      );

      auto one = 1;
      
      // Second matrix transposed to allow non-square process grids. 
      char trans_a = use_at ? 'T' : 'N'; 
      char trans_b = use_bt ? 'T' : 'N';
      qtnh::tel alpha = 1, beta = 0;
      auto m = tot_dims_a.first, n = tot_dims_b.second, k = tot_dims_a.second;
      
      if (grid.active()) {
        // Won't be modified, so must const cast. 
        // Needs two steps because descriptor is constexpr. 
        auto desc_a = a.desc9(); auto desc_ap = const_cast<int*>(desc_a.data());
        auto desc_b = b.desc9(); auto desc_bp = const_cast<int*>(desc_b.data());
        auto desc_c = c.desc9(); auto desc_cp = const_cast<int*>(desc_c.data());

        pzgemm_(&trans_a, &trans_b, &m, &n, &k, &alpha, 
                a.data(), &one, &one, desc_ap, 
                b.data(), &one, &one, desc_bp, &beta, 
                c.data(), &one, &one, desc_cp);
      }

      return c;
    }
  }
}
