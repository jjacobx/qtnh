#include "net/qops.hpp"

#ifdef DEBUG
#include <iostream>
#endif

namespace qtnh {
  namespace qops {
    tptr_symm x(const QTNHEnv& env) {
      std::vector<tel> els = {
        0, 1, 
        1, 0
      };
    
      return SymmTensor::make(env, {}, { 2, 2 }, std::move(els));
    }

    tptr_symm h(const QTNHEnv& env) {
      std::vector<tel> els = {
        1 / std::sqrt(2),  1 / std::sqrt(2), 
        1 / std::sqrt(2), -1 / std::sqrt(2)
      };
    
      return SymmTensor::make(env, {}, { 2, 2 }, std::move(els));
    }

    tptr_symm sqrt_x(const QTNHEnv& env) {
      std::vector<tel> els = {
        tel { 1, 0 } / std::sqrt(2), -tel { 0, 1 } / std::sqrt(2), 
       -tel { 0, 1 } / std::sqrt(2),  tel { 1, 0 } / std::sqrt(2)
     };
    
      return SymmTensor::make(env, {}, { 2, 2 }, std::move(els));
    }

    tptr_symm sqrt_y(const QTNHEnv& env) {
      std::vector<tel> els = {
        1 / std::sqrt(2), -1 / std::sqrt(2), 
        1 / std::sqrt(2),  1 / std::sqrt(2)
      };
    
      return SymmTensor::make(env, {}, { 2, 2 }, std::move(els));
    }

    tptr_symm sqrt_w(const QTNHEnv& env) {
      std::vector<tel> els = {
        tel { 1, 0 } / std::sqrt(2), -tel { 1, 1 } / 2.0, 
        tel { 1, -1 } / 2.0,          tel { 1, 0 } / std::sqrt(2)
      };
    
      return SymmTensor::make(env, {}, { 2, 2 }, std::move(els));
    }

    tptr_symm fsim_tp(const QTNHEnv& env, double phi) {
      std::vector<tel> els {
        1, 0,             0,             0, 
        0, 0,             tel { 0, -1 }, 0, 
        0, tel { 0, -1 }, 0,             0, 
        0, 0,             0,             std::exp(tel { 0, -phi })
      };

      return SymmTensor::make(env, {}, { 2, 2, 2, 2 }, std::move(els));
    }

    MPO ca(const QTNHEnv& env, std::size_t n, std::vector<tel> els) {
      std::vector<tel> c_op { 1, 0, 0, 0, 0, 0, 0, 1 };
      std::vector<tel> i_op { 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1 };
      std::vector<tel> t_op { 1, 0, 0, 1, els.at(0), els.at(1), els.at(2), els.at(3) };
    
      std::vector<tptr> ops(n);
    
      ops.at(0) = DenseTensor::make(env, {}, { 2, 2, 2 }, std::move(c_op));
      ops.at(0) = Tensor::permute(std::move(ops.at(0)), { 2, 1, 0 });
    
      for (auto i = 1UL; i < n - 1; ++i) {
        ops.at(i) = DenseTensor::make(env, {}, { 2, 2, 2, 2 }, std::vector<tel>(i_op));
        ops.at(i) = Tensor::permute(std::move(ops.at(i)), { 2, 3, 1, 0 });
      }
    
      ops.at(n - 1) = DenseTensor::make(env, {}, { 2, 2, 2 }, std::move(t_op));
      ops.at(n - 1) = Tensor::permute(std::move(ops.at(n - 1)), { 2, 1, 0 });
    
      return MPO(std::move(ops));
    }

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

    MPO fsim(const QTNHEnv& env, std::size_t n, double phi) {
      std::vector<tel> t1_op {
        1, 0,             0,             0, 
        0, tel { 0, -1 }, 0,             0, 
        0, 0,             tel { 0, -1 }, 0, 
        0, 0,             0,             std::exp(tel { 0, -phi })
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
            els.insert(els.begin() + 4 * ncol * (nrow - j - 1), 4, 0);
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

    std::vector<qtnh::wire> naive_rotate(std::size_t n, std::size_t d) {
      std::vector<qtnh::wire> targets(0);

      auto s = n - d;
      for (auto i = 0UL; i < s / 2; ++i) {
        targets.push_back({ i, s - i - 1 });
      }
      for (auto i = 0UL; i < d / 2; ++i) {
        targets.push_back({ s + i, n - i - 1 });
      }
      for (auto i = 0UL; i < n / 2; ++i) {
        targets.push_back({ i, n - i - 1 });
      }

      return targets;
    }

    std::vector<qtnh::wire> rotate_swaps(std::size_t n, int d) {
      std::vector<qtnh::wire> targets(0);
      std::vector<bool> rotated(n, false);
      
      for (auto l = 0UL; l < n; ++l) {
        auto i = l;
        while(!rotated.at(i)) {
          auto k = (int(i) + d) % int(n);
          auto j = k >= 0 ? std::size_t(k) : std::size_t(n + k);

          rotated.at(i) = true;
          if (rotated.at(j)) break;

          targets.push_back({ std::min(i, j), std::max(i, j) });
          i = j;
        }
      }

      return targets;
    }

    tel urot(std::size_t k, double mul) {
      return std::exp(mul * tel { 0, 2 } * M_PI / std::pow(2, k));
    }

    MPO cmp(const QTNHEnv& env, std::size_t n, double mul) {
      std::vector<tptr> ops(n);
      std::vector<tel> op;
    
      op = {
        1, 0, 0, 0, 
        0, 0, 0, 1
      };
      ops.at(0) = DenseTensor::make(env, {}, { 2, 2, 2 }, std::move(op));
      ops.at(0) = Tensor::permute(std::move(ops.at(0)), { 2, 1, 0 });
    
      for (auto i = 1UL; i + 1 < n; ++i) {
        op = {
          1, 0, 0, 1,  0, 0, 0, 0, 
          0, 0, 0, 0,  1, 0, 0, urot(i + 1, mul)
        };
        ops.at(i) = DenseTensor::make(env, {}, { 2, 2, 2, 2 }, std::move(op));
        ops.at(i) = Tensor::permute(std::move(ops.at(i)), { 2, 3, 1, 0 });
      }
      
      op = {
        1, 0, 0, 1, 
        1, 0, 0, urot(n, mul)
      };
      ops.at(n - 1) = DenseTensor::make(env, {}, { 2, 2, 2 }, std::move(op));
      ops.at(n - 1) = Tensor::permute(std::move(ops.at(n - 1)), { 2, 1, 0 });
    
      return MPO(std::move(ops));
    }

    MPO cmp_rev(const QTNHEnv& env, std::size_t n, double mul) {
      std::vector<tptr> ops(n);
      std::vector<tel> op;

      op = {
        1, 0, 0, 1, 
        1, 0, 0, urot(n, mul)
      };
      ops.at(0) = DenseTensor::make(env, {}, { 2, 2, 2 }, std::move(op));
      ops.at(0) = Tensor::permute(std::move(ops.at(0)), { 2, 1, 0 });
    
      for (auto i = 1UL; i + 1 < n; ++i) {
        op = {
          1, 0, 0, 1,  0, 0, 0, 0, 
          0, 0, 0, 0,  1, 0, 0, urot(n - i, mul)
        };
        ops.at(i) = DenseTensor::make(env, {}, { 2, 2, 2, 2 }, std::move(op));
        ops.at(i) = Tensor::permute(std::move(ops.at(i)), { 2, 3, 1, 0 });
      }
      
      op = {
        1, 0, 0, 0, 
        0, 0, 0, 1
      };
      ops.at(n - 1) = DenseTensor::make(env, {}, { 2, 2, 2 }, std::move(op));
      ops.at(n - 1) = Tensor::permute(std::move(ops.at(n - 1)), { 2, 1, 0 });
    
      return MPO(std::move(ops));
    }
  }
}
