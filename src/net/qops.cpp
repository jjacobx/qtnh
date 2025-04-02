#include "net/qops.hpp"

// #include <iostream>
// #include "util/ops.hpp"

namespace qtnh {
  namespace qops {
    MPO swap(const QTNHEnv& env, std::size_t n) {
      std::vector<tel> t1_op { 
        1, 0, 0, 0, 
        0, 1, 0, 0, 
        0, 0, 1, 0, 
        0, 0, 0, 1
      };
      std::vector<tel> id_op { 
        1, 0, 0, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, 
        0, 0, 0, 0,  1, 0, 0, 1,  0, 0, 0, 0,  0, 0, 0, 0, 
        0, 0, 0, 0,  0, 0, 0, 0,  1, 0, 0, 1,  0, 0, 0, 0, 
        0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  1, 0, 0, 1
      };
      std::vector<tel> t2_op { 
        1, 0, 0, 0, 
        0, 0, 1, 0, 
        0, 1, 0, 0, 
        0, 0, 0, 1
      };

      std::vector<tptr> ops(n);

      ops.at(0) = DenseTensor::make(env, {}, { 4, 2, 2 }, std::move(t1_op));
      ops.at(0) = Tensor::permute(std::move(ops.at(0)), { 2, 1, 0 });

      for (auto i = 1UL; i < n - 1; ++i) {
        ops.at(i) = DenseTensor::make(env, {}, { 4, 4, 2, 2 }, std::vector<tel>(id_op));
        ops.at(i) = Tensor::permute(std::move(ops.at(i)), { 2, 3, 1, 0 });
      }
    
      ops.at(n - 1) = DenseTensor::make(env, {}, { 4, 2, 2 }, std::move(t2_op));
      ops.at(n - 1) = Tensor::permute(std::move(ops.at(n - 1)), { 2, 1, 0 });
    
      return MPO(std::move(ops));
    }

    MPO cmpo(const QTNHEnv& env, const MPO& mpo, std::size_t n) {
      std::vector<tel> c_op { 
        1, 0, 0, 0, 
        0, 0, 0, 1
      };
      std::vector<tel> id_op { 
        1, 0, 0, 1,  0, 0, 0, 0, 
        0, 0, 0, 0,  1, 0, 0, 1
      };

      std::vector<tptr> ops(n);

      ops.at(0) = DenseTensor::make(env, {}, { 2, 2, 2 }, std::move(c_op));
      ops.at(0) = Tensor::permute(std::move(ops.at(0)), { 2, 1, 0 });
    
      for (auto i = 1UL; i < n - mpo.nSites(); ++i) {
        ops.at(i) = DenseTensor::make(env, {}, { 2, 2, 2, 2 }, std::vector<tel>(id_op));
        ops.at(i) = Tensor::permute(std::move(ops.at(i)), { 2, 3, 1, 0 });
      }

      for (auto i = 0UL; i < mpo.nSites() - 1; ++i) {
        auto op = mpo.at(i).copy();
        auto nrow = 1UL, ncol = 1UL;

        if (i == 0UL) {
          op = Tensor::permute(std::move(op), { 2, 1, 0 });
          ncol = op->totDims().at(0);
        } else {
          op = Tensor::permute(std::move(op), { 3, 2, 0, 1 });
          nrow = op->totDims().at(0);
          ncol = op->totDims().at(1);
        }
        
        auto els = op->cast<DenseTensor>()->extractEls();
        
        // TODO: Implement for non-root processes. 
        if (utils::is_root()) {
          for (auto j = 0UL; j < nrow; ++j) {
            els.insert(els.begin() + ncol * (nrow - j - 1), 4, 0);
          }
          els.insert(els.begin(), 4 * ncol, 0);
          els.insert(els.begin(), { 1, 0, 0, 1 });
        }

        auto k = n - mpo.nSites() + i;
        ops.at(k) = DenseTensor::make(env, {}, { nrow + 1, ncol + 1, 2, 2 }, std::move(els));
        ops.at(k) = Tensor::permute(std::move(ops.at(k)), { 2, 3, 1, 0 });
      }

      auto op = mpo.at(mpo.nSites() - 1).copy();
      op = Tensor::permute(std::move(op), { 2, 1, 0 });

      auto nrow = op->totDims().at(0);
      auto els = op->cast<DenseTensor>()->extractEls();
      els.insert(els.begin(), { 1, 0, 0, 1 });

      ops.at(n - 1) = DenseTensor::make(env, {}, { nrow + 1, 2, 2 }, std::move(els));
      ops.at(n - 1) = Tensor::permute(std::move(ops.at(n - 1)), { 2, 1, 0 });

      return MPO(std::move(ops));
    }
  }
}
