#include "core/utils.hpp"
#include "tensor/special.hpp"

namespace qtnh {
  Tensor* SwapTensor::contract_disp(Tensor* tp, const std::vector<qtnh::wire>& ws) {
    tp->swap(ws.at(0).first, ws.at(0).second); 
    return tp; 
  }

  Tensor* SwapTensor::contract(SDenseTensor* tp, const std::vector<qtnh::wire>& ws) {
    tp->swap(ws.at(0).first, ws.at(0).second); 
    return tp; 
  }

  Tensor* SwapTensor::contract(DDenseTensor* tp, const std::vector<qtnh::wire>& ws) {
    tp->swap(ws.at(0).first, ws.at(0).second); 
    return tp; 
  }

  SwapTensor::SwapTensor(const QTNHEnv& env, std::size_t n1, std::size_t n2)
    : Tensor(env), SharedTensor({ n1, n2, n2, n1 }) {}
  
  qtnh::tel SwapTensor::operator[](const qtnh::tidx_tup& idxs) const {
    return (idxs.at(0) == idxs.at(3)) && (idxs.at(1) == idxs.at(2)); 
  }

  void SwapTensor::swap(qtnh::tidx_tup_st, qtnh::tidx_tup_st) {
      utils::throw_unimplemented(); 
      return;
  }


  Tensor* IdentityTensor::contract_disp(Tensor* tp, const std::vector<qtnh::wire>& ws) { 
    return tp; 
  }

  Tensor* IdentityTensor::contract(SDenseTensor* tp, const std::vector<qtnh::wire>& ws) { 
    return tp; 
  }

  Tensor* IdentityTensor::contract(DDenseTensor* tp, const std::vector<qtnh::wire>& ws) { 
    return tp; 
  }

  IdentityTensor::IdentityTensor(const QTNHEnv& env, const qtnh::tidx_tup& in_dims)
    : Tensor(env), SharedTensor(utils::concat_dims(in_dims, in_dims)) {};

  qtnh::tel IdentityTensor::operator[](const qtnh::tidx_tup& idxs) const {
    for (std::size_t i = 0; i < idxs.size() / 2; ++i) { 
      if (idxs.at(i) != idxs.at(i + idxs.size() / 2)) { 
        return 0;
      }
    }

    return 1;
  }

  void IdentityTensor::swap(qtnh::tidx_tup_st, qtnh::tidx_tup_st) {
    utils::throw_unimplemented(); 
    return; 
  }


  Tensor* ConvertTensor::contract_disp(Tensor* tp, const std::vector<qtnh::wire>& ws) {
    return tp->contract(this, ws);
  }

  Tensor* ConvertTensor::contract(SDenseTensor* tp, const std::vector<qtnh::wire>& ws) {
    // TODO: More validation
    return tp->distribute(ws.size());
  }

  Tensor* ConvertTensor::contract(DDenseTensor* tp, const std::vector<qtnh::wire>& ws) {
    // TODO: More validation
    if (ws.size() == 0) {
      return tp->share();
    }

    std::size_t num_dist = 0;
    for (auto w : ws) {
      if (w.second < tp->getDistDims().size()) {
        num_dist++;
      }
    }

    if (num_dist == 0) {
      tp->scatter(ws.size());
      return tp; 
    } else if (num_dist == ws.size()) {
      tp->gather(ws.size());
      return tp;
    }

    throw std::runtime_error("Invalid contraction");
  }

  ConvertTensor::ConvertTensor(const QTNHEnv& env, const qtnh::tidx_tup& in_dims)
    : Tensor(env), IdentityTensor(env, in_dims) {};
}