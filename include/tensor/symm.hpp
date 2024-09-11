#ifndef _TENSOR_SYMM_HPP
#define _TENSOR_SYMM_HPP

#include "tensor/dense.hpp"

namespace qtnh {
  class SymmTensorBase;
  class SymmTensor;
  class SwapTensor;

  /// Symmetric tensor base virtual class, which restricts dense tensors by assuming dimensions can be split into equal input and output. 
  /// While in/out local and distributed dimension may differ, the concatenated in/out parts have to be exactly the same. 
  class SymmTensorBase : public DenseTensorBase {
    public: 
      SymmTensorBase() = delete;
      SymmTensorBase(const SymmTensorBase&) = delete;
      virtual ~SymmTensorBase() = default;

      virtual TT type() const noexcept override { return TT::symmTensorBase; }

      /// @brief Swap indices on current tensor. 
      /// @param tp Ownership of tptr to tensor to swap. 
      /// @param idx1 First index to swap. 
      /// @param idx2 Second index to swap. 
      /// @return Ownership of tptr to swapped tensor. 
      static qtnh::tptr swapIO(qtnh::tptr tp, qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) {
        return utils::one_unique(std::move(tp), tp->cast<SymmTensorBase>()->swapIO(idx1, idx2));
      }
      /// @brief Shift the border between shared and distributed dimensions by a given offset. 
      /// @param tp Ownership of tptr to tensor to re-scatter. 
      /// @param offset New offset between distributed and local dimensions – negative gathers, while positive scatters. 
      /// @return Ownership of tptr to re-scattered tensor. 
      static qtnh::tptr rescatterIO(qtnh::tptr tp, int offset) {
        return utils::one_unique(std::move(tp), tp->cast<SymmTensorBase>()->rescatterIO(offset));
      }
      /// @brief Permute tensor indices according to mappings in the permutation tuple. 
      /// @param tp Ownership of tptr to tensor to permute. 
      /// @param ptup Permutation tuple of the same size as total dimensions, and each entry unique. 
      /// @return Ownership of tptr to permuted tensor. 
      static qtnh::tptr permuteIO(qtnh::tptr tp, std::vector<qtnh::tidx_tup_st> ptup) {
        return utils::one_unique(std::move(tp), tp->cast<SymmTensorBase>()->permuteIO(ptup));
      }

    protected:
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with default distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      SymmTensorBase(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims);
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with given distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param params Distribution parameters of the tensor (str, cyc, off)
      SymmTensorBase(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, BcParams params);

      virtual bool isSymm() const noexcept override { return true; }

      /// @brief Convert any derived tensor to writable symmetric tensor
      /// @return Pointer to an equivalent writable symmetric tensor. 
      virtual SymmTensor* toSymm() noexcept override;

      /// @brief Swap input/output indices on current tensor. 
      /// @param idx1 First index to swap. 
      /// @param idx2 Second index to swap. 
      /// @return Pointer to swapped tensor, which might be of a different derived type. 
      virtual Tensor* swapIO(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2);
      /// @brief Re-broadcast current tensor. 
      /// @param params Broadcast parameters of the tensor (str, cyc, off)
      /// @return Pointer to re-broadcasted tensor, which might be of a different derived type. 
      virtual Tensor* rebcast(BcParams params) override;
      /// @brief Shift the border between input/output shared and distributed dimensions by a given offset. 
      /// @param offset New offset between distributed and local dimensions – negative gathers, while positive scatters. 
      /// @return Pointer to re-scattered tensor, which might be of a different derived type. 
      virtual Tensor* rescatterIO(int offset);
      /// @brief Permute tensor input/output indices according to mappings in the permutation tuple. 
      /// @param ptup Permutation tuple of the same size as total dimensions, and each entry unique. 
      /// @return Pointer to permuted tensor, which might be of a different derived type. 
      virtual Tensor* permuteIO(std::vector<qtnh::tidx_tup_st> ptup);
  };

  /// Writable general symmetric tensor class, which allows direct access to all elements. Restrictions for symmetric tensors apply. 
  class SymmTensor : public SymmTensorBase, private TIDense {
    public:
      friend class SymmTensorBase;

      SymmTensor() = delete;
      SymmTensor(const SymmTensor&) = delete;
      ~SymmTensor() = default;

      /// @brief Construct symmetric tensor with default distribution parameters and transfer its ownership. 
      /// @param env Environment to use for construction. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param loc_dims Local index dimensions. 
      /// @param n_dis_in_dims Number of distributed input dimensions. 
      /// @param els Complex vector of local elements. 
      /// @return Ownership of unique pointer to created tensor.
      static std::unique_ptr<SymmTensor> make(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, std::vector<qtnh::tel>&& els) {
        return std::unique_ptr<SymmTensor>(new SymmTensor(env, dis_dims, loc_dims, std::move(els)));
      }
      /// @brief Construct symmetric tensor and transfer its ownership. 
      /// @param env Environment to use for construction. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param loc_dims Local index dimensions. 
      /// @param n_dis_in_dims Number of distributed input dimensions. 
      /// @param els Complex vector of local elements. 
      /// @param params Distribution parameters (str, cyc, off). 
      /// @return Ownership of unique pointer to created tensor.
      static std::unique_ptr<SymmTensor> make(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, std::vector<qtnh::tel>&& els, BcParams params) {
        return std::unique_ptr<SymmTensor>(new SymmTensor(env, dis_dims, loc_dims, std::move(els), params));
      }

      /// @brief Create a copy of the tensor. 
      /// @return Tptr to duplicated tensor. 
      /// 
      /// Overuse may cause memory shortage. 
      virtual std::unique_ptr<Tensor> copy() const noexcept override;

      virtual TT type() const noexcept override { return TT::symmTensor; }

      /// @brief Rank-unsafe method to get element and given local indices. 
      /// @param idxs Tensor index tuple indicating local position of the element. 
      /// @return Value of the element at given indices. Throws error if value is not present. 
      /// @deprecated Will be superseded by direct addressing of array elements with numeric indices. 
      ///
      /// This method requires ensuring the element is present (i.e. the tensor is active)
      /// on current rank. On all active ranks, it must return an element, but different ranks  
      /// might have different values. 
      virtual qtnh::tel operator[](qtnh::tidx_tup loc_idxs) const override;
      /// @brief Set element on given local indices. 
      /// @param idxs Tensor index tuple indicating local position to be updated. 
      /// @deprecated Will be superseded by direct addressing of array elements with numeric indices. 
      ///
      /// The index update is executed on all active ranks, and different values might be
      /// passed to the method on different ranks. 
      qtnh::tel& operator[](qtnh::tidx_tup loc_idxs);

      /// @brief Directly access local array the tensor. 
      /// @param i Local array index to access. 
      /// @return The element at given index. Throws an error if not present (or out of bounds). 
      ///
      /// Returned element depends on the storage method used. It may differ for two identical 
      /// tensors that use different underlying classes. It may also produce unexpected results 
      /// when virtual elements are stored, i.e. elements useful for calculations, but not actually 
      /// present in the tensor. 
      virtual qtnh::tel operator[](std::size_t i) const override { return loc_els_.at(i); }
      /// @brief Access element at total indices if present. 
      /// @param tot_idxs Indices with total position of the element. 
      /// @return Value of the element at given indices. Throws error if not present. 
      ///
      /// It is advised to ensure the element is present at current rank with Tensor::has method. 
      virtual qtnh::tel at(qtnh::tidx_tup tot_idxs) const override;

      /// @brief Directly access and reference local array the tensor. 
      /// @param i Local array index to access. 
      /// @return Reference to the element at given index. Throws an error if not present (or out of bounds). 
      ///
      /// Returned element depends on the storage method used. It may differ for two identical 
      /// tensors that use different underlying classes. It may also produce unexpected results 
      /// when virtual elements are stored, i.e. elements useful for calculations, but not actually 
      /// present in the tensor. 
      qtnh::tel& operator[](std::size_t i) { return loc_els_.at(i); }
      /// @brief Access and reference element at total indices if present. 
      /// @param tot_idxs Indices with total position of the element. 
      /// @return Reference to the element at given indices. Throws error if not present. 
      ///
      /// It is advised to ensure the element is present at current rank with Tensor::has method. 
      qtnh::tel& at(qtnh::tidx_tup tot_idxs);
      /// @brief Set element on given total indices. 
      /// @param tot_idxs Indices with total position of the element. 
      /// @param el Complex number to be written at the given position. 
      ///
      /// The index update will do nothing on ranks that do not contain the element on given indices. 
      void put(qtnh::tidx_tup tot_idxs, qtnh::tel el);

    protected:
      /// @brief Construct symmetric tensor with default distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param loc_dims Local index dimensions. 
      /// @param n_dis_in_dims Number of distributed input dimensions
      /// @param els Complex vector of local elements. 
      SymmTensor(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, std::vector<qtnh::tel>&& els);
      /// @brief Construct symmetric tensor. 
      /// @param env Environment to use for construction. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param loc_dims Local index dimensions. 
      /// @param n_dis_in_dims Number of distributed input dimensions. 
      /// @param els Complex vector of local elements. 
      /// @param params Distribution parameters (str, cyc, off). 
      SymmTensor(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, std::vector<qtnh::tel>&& els, BcParams params);

      /// @brief Convert any derived tensor to symmetric tensor. 
      /// @return Symmetric tensor equivalent to calling tensor. 
      virtual SymmTensor* toSymm() noexcept override { return this; }

      /// @brief Swap input/output indices on current tensor. 
      /// @param idx1 First index to swap. 
      /// @param idx2 Second index to swap. 
      /// @param io Tensor index input/output label to indicate which indices to swap. 
      /// @return Pointer to swapped tensor, which might be of a different derived type. 
      virtual SymmTensor* swapIO(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) override;
      /// @brief Re-broadcast current tensor. 
      /// @param params Broadcast parameters of the tensor (str, cyc, off)
      /// @return Pointer to re-broadcasted tensor, which might be of a different derived type. 
      virtual SymmTensor* rebcast(BcParams params) override;
      /// @brief Shift the border between input/output shared and distributed dimensions by a given offset. 
      /// @param offset New offset between distributed and local dimensions – negative gathers, while positive scatters. 
      /// @param io Tensor index input/output label to indicate which indices to scatter. 
      /// @return Pointer to re-scattered tensor, which might be of a different derived type. 
      virtual SymmTensor* rescatterIO(int offset) override;
      /// @brief Permute tensor input/output indices according to mappings in the permutation tuple. 
      /// @param ptup Permutation tuple of the same size as total dimensions, and each entry unique. 
      /// @param io Tensor index input/output label to indicate which indices to scatter. 
      /// @return Pointer to permuted tensor, which might be of a different derived type. 
      virtual SymmTensor* permuteIO(std::vector<qtnh::tidx_tup_st> ptup) override;
  };

  /// Rank 4 symmetric swap tensor for swapping two indices with dimension n. The swap tensor must have dimensions (n, n, n, n). 
  class SwapTensor : public SymmTensorBase {
    public:
      SwapTensor() = delete;
      SwapTensor(const SymmTensor&) = delete;
      ~SwapTensor() = default;

      /// @brief Construct swap tensor with default distribution parameters and transfer its ownership. 
      /// @param env Environment to use for construction. 
      /// @param n Dimension of swapped indices (both are expected to have the same size). 
      /// @param d Number of distributed input/output dimensions (can be either 0, 1 or 2), assuming input/output distribution is the same. 
      /// @return Ownership of unique pointer to created tensor. 
      static std::unique_ptr<SwapTensor> make(const QTNHEnv& env, std::size_t n, std::size_t d) {
        return std::unique_ptr<SwapTensor>(new SwapTensor(env, n, d));
      }
      /// @brief Construct swap tensor and transfer its ownership. 
      /// @param env Environment to use for construction. 
      /// @param n Dimension of swapped indices (both are expected to have the same size). 
      /// @param d Number of distributed input/output dimensions (can be either 0, 1 or 2), assuming input/output distribution is the same. 
      /// @param params Distribution parameters (str, cyc, off). 
      /// @return Ownership of unique pointer to created tensor.
      static std::unique_ptr<SwapTensor> make(const QTNHEnv& env, std::size_t n, std::size_t d, BcParams params) {
        return std::unique_ptr<SwapTensor>(new SwapTensor(env, n, d, params));
      }

      /// @brief Create a copy of the tensor. 
      /// @return Tptr to duplicated tensor. 
      /// 
      /// Overuse may cause memory shortage. 
      virtual std::unique_ptr<Tensor> copy() const noexcept override;

      virtual TT type() const noexcept override { return TT::swapTensor; }

      /// @brief Rank-unsafe method to get element and given local indices. 
      /// @param idxs Tensor index tuple indicating local position of the element. 
      /// @return Value of the element at given indices. Throws error if value is not present. 
      /// @deprecated Will be superseded by direct addressing of array elements with numeric indices. 
      ///
      /// This method requires ensuring the element is present (i.e. the tensor is active)
      /// on current rank. On all active ranks, it must return an element, but different ranks  
      /// might have different values. 
      virtual qtnh::tel operator[](qtnh::tidx_tup loc_idxs) const override;

      /// @brief Directly access local array the tensor. 
      /// @param i Local array index to access. 
      /// @return The element at given index. Throws an error if not present (or out of bounds). 
      ///
      /// Returned element depends on the storage method used. It may differ for two identical 
      /// tensors that use different underlying classes. It may also produce unexpected results 
      /// when virtual elements are stored, i.e. elements useful for calculations, but not actually 
      /// present in the tensor. 
      virtual qtnh::tel operator[](std::size_t i) const override;
      /// @brief Access element at total indices if present. 
      /// @param tot_idxs Indices with total position of the element. 
      /// @return Value of the element at given indices. Throws error if not present. 
      ///
      /// It is advised to ensure the element is present at current rank with Tensor::has method. 
      virtual qtnh::tel at(qtnh::tidx_tup tot_idxs) const override;

    protected:
      /// @brief Construct rank 4 swap tensor with given single index size within environment with default distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param n Dimension of swapped indices (both are expected to have the same size). 
      /// @param d Number of distributed input/output dimensions (can be either 0, 1 or 2), assuming input/output distribution is the same. 
      SwapTensor(const QTNHEnv& env, std::size_t n, std::size_t d);
      /// @brief Construct rank 4 swap tensor with given index size within environment with given distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param n Dimension of swapped indices (both are expected to have the same size). 
      /// @param d Number of distributed input/output dimensions (can be either 0, 1 or 2), assuming input/output distribution is the same. 
      /// @param params Distribution parameters of the tensor (str, cyc, off). 
      SwapTensor(const QTNHEnv& env, std::size_t n, std::size_t d, BcParams params);

      /// @brief Re-broadcast current tensor. 
      /// @param params Broadcast parameters of the tensor (str, cyc, off)
      /// @return Pointer to re-broadcasted tensor, which might be of a different derived type. 
      virtual SwapTensor* rebcast(BcParams params) override;
  };
}

#endif