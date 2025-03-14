#ifndef _TEN_TYPE_DIAG_HPP
#define _TEN_TYPE_DIAG_HPP

#include "ten/type/symm.hpp"

namespace qtnh {
  class DiagTensorBase;
  class DiagTensor;
  class IdenTensor;

  /// Diagonal tensor base virtual class, which assumes that a symmetric tensors only has non-zero elements on a diagonal. 
  /// The diagonal is defined by the same total input and output indices. Symmetric tensor dimension restrictions apply. 
  /// Even if given rank is non-diagonal, distributed output index diagonal is stored. The distributed input index can be
  /// shrunk to 0 to limit the number of required ranks. To convert to a more general tensor, full form is necessary. 
  class DiagTensorBase : public SymmTensorBase {
    public: 
      DiagTensorBase() = delete;
      DiagTensorBase(const DiagTensorBase&) = delete;
      virtual ~DiagTensorBase() = default;

      virtual TT type() const noexcept override { return TT::diagTensorBase; }

      // TODO: More sophisticated availability checking. 
      // TODO: - available through base pointer
      // TODO: - element present in tensor vs in memory
      bool shrunk() const noexcept { return shrunk_; }
      bool available(qtnh::tidx_tup tot_dims) const noexcept;

      static qtnh::tptr shrink(qtnh::tptr tp) {
        auto p = tp->cast<DiagTensorBase>()->shrink();
        return utils::one_unique(std::move(tp), p);
      }

      static qtnh::tptr expand(qtnh::tptr tp) {
        auto p = tp->cast<DiagTensorBase>()->expand();
        return utils::one_unique(std::move(tp), p);
      }

    protected:
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with default distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param shrunk Flag for whether the front has been shrunk to 0. 
      DiagTensorBase(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, bool shrunk);
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with given distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param shrunk Flag for whether the front has been shrunk to 0. 
      /// @param params Distribution parameters of the tensor (str, cyc, off). 
      DiagTensorBase(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, bool shrunk, BcParams params);

      virtual bool isDiag() const noexcept override { return true; }

      /// @brief Convert any derived tensor to writable diagonal tensor. 
      /// @return Pointer to an equivalent writable diagonal tensor. 
      virtual DiagTensor* toDiag() noexcept override;

      /// @brief Swap indices on current tensor. 
      /// @param idx1 First index to swap. 
      /// @param idx2 Second index to swap. 
      /// @return Pointer to swapped tensor, which might be of a different derived type. 
      virtual Tensor* swapIO(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) override;
      /// @brief Re-broadcast current tensor. 
      /// @param params Broadcast parameters of the tensor (str, cyc, off)
      /// @return Pointer to re-broadcasted tensor, which might be of a different derived type. 
      virtual Tensor* rebcast(BcParams params) override;
      /// @brief Shift the border between shared and distributed dimensions by a given offset. 
      /// @param offset New offset between distributed and local dimensions – negative gathers, while positive scatters. 
      /// @return Pointer to re-scattered tensor, which might be of a different derived type. 
      virtual Tensor* rescatterIO(int offset) override;

      virtual Tensor* shrink();
      virtual Tensor* expand();

      bool shrunk_;  ///< Flag for whether distributed input dimensions are shrunk to 0. 
  };

  /// Writable diagonal tensor class, which allows direct access to diagonal elements. 
  /// Only non-zero elements are stored in a vector, significantly reducing memory consumption. 
  class DiagTensor : public DiagTensorBase {
    public: 
      friend class DiagTensorBase; 

      DiagTensor() = delete;
      DiagTensor(const DiagTensor&) = delete;
      ~DiagTensor() = default;

      /// @brief Construct diagonal tensor with default distribution parameters and transfer its ownership. 
      /// @param env Environment to use for construction. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param loc_dims Local index dimensions. 
      /// @param shrunk Flag for whether the front has been shrunk to 0. 
      /// @param diag_els Complex vector of local diagonal elements. 
      /// @return Ownership of unique pointer to created tensor.
      static std::unique_ptr<DiagTensor> make(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, bool shrunk, std::vector<qtnh::tel>&& diag_els) {
        return std::unique_ptr<DiagTensor>(new DiagTensor(env, dis_dims, loc_dims, shrunk, std::move(diag_els)));
      }
      /// @brief Construct diagonal tensor and transfer its ownership. 
      /// @param env Environment to use for construction. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param loc_dims Local index dimensions. 
      /// @param shrunk Flag for whether the front has been shrunk to 0. 
      /// @param diag_els Complex vector of local diagonal elements. 
      /// @param params Distribution parameters (str, cyc, off). 
      /// @return Ownership of unique pointer to created tensor.
      static std::unique_ptr<DiagTensor> make(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, bool shrunk, std::vector<qtnh::tel>&& diag_els, BcParams params) {
        return std::unique_ptr<DiagTensor>(new DiagTensor(env, dis_dims, loc_dims, shrunk, std::move(diag_els), params));
      }

      /// @brief Create a copy of the tensor. 
      /// @return Tptr to duplicated tensor. 
      /// 
      /// Overuse may cause memory shortage. 
      virtual std::unique_ptr<Tensor> copy() const noexcept override;

      virtual TT type() const noexcept override { return TT::diagTensor; }

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
      virtual qtnh::tel operator[](std::size_t i) const override { return diagonal_[i]; }
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
      qtnh::tel& operator[](std::size_t i) { return diagonal_[i]; }
      /// @brief Set element on given total indices. 
      /// @param tot_idxs Indices with total position of the element. 
      /// @param el Complex number to be written at the given position. 
      ///
      /// The index update will do nothing on ranks that do not contain the element on given indices. 
      void put(qtnh::tidx_tup tot_idxs, qtnh::tel el);

      /// @brief Reshape distributed and local dimensions of a tensor. 
      /// @param dis_dims New distributed dimensions. 
      /// @param loc_dims New local dimensions. 
      ///
      /// Both distributed and local dimensions must be compatible with the old values. 
      virtual void reshape(qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims) override;

    protected: 
      /// @brief Construct diagonal tensor with default distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param loc_dims Local index dimensions. 
      /// @param shrunk Flag for whether the front has been shrunk to 0. 
      /// @param diag_els Complex vector of local diagonal elements. 
      DiagTensor(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, bool shrunk, std::vector<qtnh::tel>&& diag_els);
      /// @brief Construct diagonal tensor. 
      /// @param env Environment to use for construction. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param loc_dims Local index dimensions. 
      /// @param shrunk Flag for whether the front has been shrunk to 0. 
      /// @param diag_els Complex vector of local diagonal elements. 
      /// @param params Distribution parameters of the tensor (str, cyc, off). 
      DiagTensor(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, bool shrunk, std::vector<qtnh::tel>&& diag_els, BcParams params);

      /// @brief Convert any derived tensor to writable diagonal tensor
      /// @return Pointer to an equivalent writable diagonal tensor. 
      virtual DiagTensor* toDiag() noexcept override { return this; }

      /// @brief Swap indices on current tensor. 
      /// @param idx1 First index to swap. 
      /// @param idx2 Second index to swap. 
      /// @return Pointer to swapped tensor, which might be of a different derived type. 
      virtual DiagTensor* swapIO(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) override;
      /// @brief Re-broadcast current tensor. 
      /// @param params Broadcast parameters of the tensor (str, cyc, off)
      /// @return Pointer to re-broadcasted tensor, which might be of a different derived type. 
      virtual DiagTensor* rebcast(BcParams params) override;
      /// @brief Shift the border between shared and distributed dimensions by a given offset. 
      /// @param offset New offset between distributed and local dimensions – negative gathers, while positive scatters. 
      /// @return Pointer to re-scattered tensor, which might be of a different derived type. 
      virtual DiagTensor* rescatterIO(int offset) override;

      // Truncation should happen implicitly. 
      virtual DiagTensor* shrink() override;
      virtual DiagTensor* expand() override;

    private: 
      // std::vector<qtnh::tel> loc_diag_els_;  ///< Local diagonal elements. 
      DenseTensor diagonal_;
  };

  /// Identity diagonal tensor class. Has symmetric total dimensions and 1s on a diagonal – all other elements are zero. 
  /// Can be used for making indices local/distributed by using different input/output local-distributed dimension splits. 
  class IdenTensor : public DiagTensorBase {
    public:
      IdenTensor() = delete;
      IdenTensor(const IdenTensor&) = delete;
      ~IdenTensor() = default;

      /// @brief Construct identity tensor with default distribution parameters and transfer its ownership. 
      /// @param env Environment to use for construction. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param loc_dims Local index dimensions. 
      /// @param shrunk Flag for whether the front has been shrunk to 0. 
      /// @return Ownership of unique pointer to created tensor.
      static std::unique_ptr<IdenTensor> make(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, bool shrunk) {
        return std::unique_ptr<IdenTensor>(new IdenTensor(env, dis_dims, loc_dims, shrunk));
      }
      /// @brief Construct identity tensor and transfer its ownership. 
      /// @param env Environment to use for construction. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param loc_dims Local index dimensions. 
      /// @param shrunk Flag for whether the front has been shrunk to 0. 
      /// @param params Distribution parameters (str, cyc, off). 
      /// @return Ownership of unique pointer to created tensor. 
      static std::unique_ptr<IdenTensor> make(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, bool shrunk, BcParams params) {
        return std::unique_ptr<IdenTensor>(new IdenTensor(env, dis_dims, loc_dims, shrunk, params));
      }

      /// @brief Create a copy of the tensor. 
      /// @return Tptr to duplicated tensor. 
      /// 
      /// Overuse may cause memory shortage. 
      virtual std::unique_ptr<Tensor> copy() const noexcept override;

      virtual TT type() const noexcept override { return TT::idenTensor; }

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
      virtual qtnh::tel operator[](std::size_t) const override { return 1; }
      /// @brief Access element at total indices if present. 
      /// @param tot_idxs Indices with total position of the element. 
      /// @return Value of the element at given indices. Throws error if not present. 
      ///
      /// It is advised to ensure the element is present at current rank with Tensor::has method. 
      virtual qtnh::tel at(qtnh::tidx_tup tot_idxs) const override;

    protected:
      /// @brief Construct identity tensor with default distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param shrunk Flag for whether the front has been shrunk to 0. 
      IdenTensor(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, bool shrunk);
      /// @brief Construct identity tensor. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param shrunk Flag for whether the front has been shrunk to 0. 
      /// @param params Distribution parameters (str, cyc, off). 
      IdenTensor(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, bool shrunk, BcParams params);

      /// @brief Re-broadcast current tensor. 
      /// @param params Broadcast parameters of the tensor (str, cyc, off)
      /// @return Pointer to re-broadcasted tensor, which might be of a different derived type. 
      virtual IdenTensor* rebcast(BcParams params) override;
  };
}

#endif