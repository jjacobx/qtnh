#ifndef QTNH_TEN_TYPE_DENSE_HPP_INCLUDE
#define QTNH_TEN_TYPE_DENSE_HPP_INCLUDE

#include "ten/type/tensor.hpp"

namespace qtnh {
  class DenseTensorBase;
  class DenseTensor;
  class RescTensor;

  template<typename T1, typename T2, typename Enable = void>
  class PairContractor;

  class TIDense {
    public: 
      TIDense() = delete;
      TIDense(const TIDense&) = delete;
      TIDense(std::vector<qtnh::tel>&& els) : loc_els_(std::move(els)) {}
      virtual ~TIDense() = default;

    protected:
      Broadcaster _swap_internal(Tensor* target, qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2);
      Broadcaster _rebcast_internal(Tensor* target, BcParams params);
      Broadcaster _rescatter_internal(Tensor* target, int offset);
      Broadcaster _permute_internal(Tensor* target, std::vector<tidx_tup_st> ptup);
      Broadcaster _shift_internal(Tensor* target, qtnh::tidx_tup_st from, qtnh::tidx_tup_st to, int offset);
      // Broadcaster _fold_internal(qtnh::tidx_tup_ids idxs, qtnh::tel_fun fun, qtnh::tel init);

      std::vector<qtnh::tel> loc_els_;  ///< Local elements. 
  };

  /// Dense tensor base virtual class, which assumes that all local elements can be stored in a vector. 
  class DenseTensorBase : public Tensor {
    public:
      DenseTensorBase() = delete;
      DenseTensorBase(const DenseTensorBase&) = delete;
      virtual ~DenseTensorBase() = default;

      virtual TT type() const noexcept override { return TT::denseTensorBase; }
    
    protected:
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with default distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param loc_dims Local index dimensions. 
      DenseTensorBase(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims);
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with given distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param loc_dims Local index dimensions. 
      /// @param params Distribution parameters of the tensor (str, cyc, off)
      DenseTensorBase(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, BcParams params);

      virtual bool isDense() const noexcept override { return true; }

      /// @brief Convert derived tensor to dense tensor. 
      /// @return Pointer to equivalent dense tensor. 
      virtual DenseTensor* toDense() noexcept override;

      /// @brief Swap indices on current tensor. 
      /// @param idx1 First index to swap. 
      /// @param idx2 Second index to swap. 
      /// @return Pointer to swapped tensor, which might be of a different derived type. 
      virtual Tensor* swap(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) override;
      /// @brief Re-broadcast current tensor. 
      /// @param params Broadcast parameters of the tensor (str, cyc, off)
      /// @return Pointer to re-broadcasted tensor, which  might be of a different derived type. 
      virtual Tensor* rebcast(BcParams params) override;
      /// @brief Shift the border between shared and distributed dimensions by a given offset. 
      /// @param offset New offset between distributed and local dimensions – negative gathers, while positive scatters. 
      /// @return Pointer to re-scattered tensor, which might be of a different derived type. 
      virtual Tensor* rescatter(int offset) override;
      /// @brief Permute tensor indices according to mappings in the permutation tuple. 
      /// @param ptup Permutation tuple of the same size as total dimensions, and each entry unique. 
      /// @return Pointer to permuted tensor, which might be of a different derived type. 
      virtual Tensor* permute(std::vector<qtnh::tidx_tup_st> ptup) override;
      /// @brief Truncate tensor index down to a given dimension. 
      /// @param idx Index to truncate. 
      /// @param size Target size of the index. 
      /// @return Pointer to truncated tensor, which might be of a different derived type. 
      virtual Tensor* truncate(qtnh::tidx_tup_st idx, std::size_t size) override;
      /// @brief Fold tensor indices with a given function. 
      /// @param idxs Vector of indices to fold. 
      /// @param fun Commutative binary function to use in fold. 
      /// @param init Initial value to the binary function. 
      /// @return Pointer to folded tensor, which might be of a different derived type. 
      virtual Tensor* fold(qtnh::tidx_tup_ids idxs, qtnh::mpi_fun fun, qtnh::tel init) override;
  };

  /// Writable dense tensor class, which allows direct access to all elements. 
  class DenseTensor : public DenseTensorBase, private TIDense {
    public:
      friend class DenseTensorBase;
      friend class DiagTensor;
      friend class PairContractor<DenseTensor, DenseTensor>;

      DenseTensor() = delete;
      DenseTensor(const DenseTensor&) = delete;
      ~DenseTensor() = default;

      /// @brief Construct dense tensor with default distribution parameters and transfer its ownership. 
      /// @param env Environment to use for construction. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param loc_dims Local index dimensions. 
      /// @param els Complex vector of local elements. 
      /// @return Ownership of unique pointer to created tensor.
      static std::unique_ptr<DenseTensor> make(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, std::vector<qtnh::tel>&& els) {
        return std::unique_ptr<DenseTensor>(new DenseTensor(env, dis_dims, loc_dims, std::move(els)));
      }
      /// @brief Construct dense tensor and transfer its ownership. 
      /// @param env Environment to use for construction. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param loc_dims Local index dimensions. 
      /// @param els Complex vector of local elements. 
      /// @param params Distribution parameters (str, cyc, off). 
      /// @return Ownership of unique pointer to created tensor.
      static std::unique_ptr<DenseTensor> make(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, std::vector<qtnh::tel>&& els, BcParams params) {
        return std::unique_ptr<DenseTensor>(new DenseTensor(env, dis_dims, loc_dims, std::move(els), params));
      }

      /// @brief Duplicate dense tensor. 
      /// @return Unique pointer to duplicated dense tensor. 
      /// 
      /// Overuse may cause memory shortage. 
      virtual qtnh::tptr copy() const noexcept override;

      virtual TT type() const noexcept override { return TT::denseTensor; }

      /// @brief Rank-unsafe method to get element and given local indices. 
      /// @param idxs Tensor index tuple indicating local position of the element. 
      /// @return Value of the element at given indices. Throws error if value is not present. 
      /// @deprecated Will be superseded by direct addressing of array elements with numeric indices. 
      ///
      /// This method requires ensuring the element is present (i.e. the tensor is active)
      /// on current rank. On all active ranks, it must return an element, but different ranks  
      /// might have different values. 
      virtual qtnh::tel operator[](qtnh::tidx_tup loc_idxs) const override;
      /// @brief Set element on given local indices. class... U
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
      virtual qtnh::tel operator[](std::size_t i) const override { return loc_els_[i]; }
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
      qtnh::tel& operator[](std::size_t i) { return loc_els_[i]; }
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

      /// TODO: this is temporary, should be changed later. 
      std::vector<qtnh::tel>&& extractEls() { return std::move(loc_els_); }
    
    protected:
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with default distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param loc_dims Local index dimensions. 
      /// @param els Complex vector of local elements. 
      DenseTensor(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, std::vector<qtnh::tel>&& els);
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with given distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param loc_dims Local index dimensions. 
      /// @param els Complex vector of local elements. 
      /// @param params Distribution parameters of the tensor (str, cyc, off)
      DenseTensor(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, std::vector<qtnh::tel>&& els, BcParams params);

      /// @brief Convert any derived tensor to writable dense tensor. 
      /// @return Pointer to equivalent writable dense tensor. 
      virtual DenseTensor* toDense() noexcept override { return this; }
      /// @brief Swap indices on current tensor. 
      /// @param idx1 First index to swap. 
      /// @param idx2 Second index to swap. 
      /// @return Pointer to swapped tensor, which might be of a different derived type. 
      virtual DenseTensor* swap(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) override;
      /// @brief Re-broadcast current tensor. 
      /// @param params Broadcast parameters of the tensor (str, cyc, off)
      /// @return Pointer to re-broadcasted tensor, which might be of a different derived type. 
      virtual DenseTensor* rebcast(BcParams params) override;
      /// @brief Shift the border between shared and distributed dimensions by a given offset. 
      /// @param offset New offset between distributed and local dimensions – negative gathers, while positive scatters. 
      /// @return Pointer to re-scattered tensor, which might be of a different derived type. 
      virtual DenseTensor* rescatter(int offset) override;
      /// @brief Permute tensor indices according to mappings in the permutation tuple. 
      /// @param ptup Permutation tuple of the same size as total dimensions, and each entry unique. 
      /// @return Pointer to permuted tensor, which might be of a different derived type. 
      virtual DenseTensor* permute(std::vector<qtnh::tidx_tup_st> ptup) override;
      /// @brief Truncate tensor index down to a given dimension. 
      /// @param idx Index to truncate. 
      /// @param size Target size of the index. 
      /// @return Pointer to truncated tensor, which might be of a different derived type. 
      virtual DenseTensor* truncate(qtnh::tidx_tup_st idx, std::size_t size) override;
      /// @brief Fold tensor indices with a given function. 
      /// @param idxs Vector of indices to fold. 
      /// @param fun Commutative binary function to use in fold. 
      /// @param init Initial value to the binary function. 
      /// @return Pointer to folded tensor, which might be of a different derived type. 
      virtual DenseTensor* fold(qtnh::tidx_tup_ids idxs, qtnh::mpi_fun fun, qtnh::tel init) override;
  };

  /// Rank-2 rescatter tensor, which can be used to scatter/gather specific indices. 
  class RescTensor : public DenseTensorBase {
    public:
      RescTensor() = delete;
      RescTensor(const RescTensor&) = delete;
      ~RescTensor() = default;

      /// @brief Construct rescatter tensor with default distribution parameters and transfer its ownership. 
      /// @param env Environment to use for construction. 
      /// @param n Dimension of each of two indices. 
      /// @return Ownership of unique pointer to created tensor.
      static std::unique_ptr<RescTensor> make(const QTNHEnv& env, std::size_t n) {
        return std::unique_ptr<RescTensor>(new RescTensor(env, n));
      }
      /// @brief Construct rescatter tensor and transfer its ownership. 
      /// @param env Environment to use for construction. 
      /// @param n Dimension of each of two indices. 
      /// @param params Distribution parameters (str, cyc, off). 
      /// @return Ownership of unique pointer to created tensor. 
      static std::unique_ptr<RescTensor> make(const QTNHEnv& env, std::size_t n, BcParams params) {
        return std::unique_ptr<RescTensor>(new RescTensor(env, n, params));
      }

      /// @brief Create a copy of the tensor. 
      /// @return Tptr to duplicated tensor. 
      /// 
      /// Overuse may cause memory shortage. 
      virtual qtnh::tptr copy() const noexcept override;

      virtual TT type() const noexcept override { return TT::rescTensor; }

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
      /// @brief Construct rescatter tensor with default distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param n Dimension of each of two indices. 
      RescTensor(const QTNHEnv& env, std::size_t n);
      /// @brief Construct rescatter tensor. 
      /// @param env Environment to use for construction. 
      /// @param n Dimension of each of two indices. 
      /// @param params Distribution parameters (str, cyc, off). 
      RescTensor(const QTNHEnv& env, std::size_t n, BcParams params);

      /// @brief Re-broadcast current tensor. 
      /// @param params Broadcast parameters of the tensor (str, cyc, off)
      /// @return Pointer to re-broadcasted tensor, which might be of a different derived type. 
      virtual RescTensor* rebcast(BcParams params) override;
  };

  /// Three-input tensor that only returns one for equal inputs. 
  class CopyTensor : public DenseTensorBase {
    public:
      CopyTensor() = delete;
      CopyTensor(const CopyTensor&) = delete;
      ~CopyTensor() = default;

      /// @brief Construct copy tensor with default distribution parameters and transfer its ownership. 
      /// @param env Environment to use for construction. 
      /// @param dis_dims_3 Triple of distributed dimensions for each input. 
      /// @param loc_dims_3 Triple of local dimensions for each input. 
      /// @return Ownership of unique pointer to created tensor.
      static std::unique_ptr<CopyTensor> make(
        const QTNHEnv& env, 
        std::array<tidx_tup, 3> dis_dims_3, 
        std::array<tidx_tup, 3> loc_dims_3
      ) {
        return std::unique_ptr<CopyTensor>(new CopyTensor(env, dis_dims_3, loc_dims_3));
      }
      /// @brief Construct copy tensor and transfer its ownership. 
      /// @param env Environment to use for construction. 
      /// @param dis_dims_3 Triple of distributed dimensions for each input. 
      /// @param loc_dims_3 Triple of local dimensions for each input. 
      /// @param params Distribution parameters (str, cyc, off). 
      /// @return Ownership of unique pointer to created tensor. 
      static std::unique_ptr<CopyTensor> make(
        const QTNHEnv& env, 
        std::array<tidx_tup, 3> dis_dims_3, 
        std::array<tidx_tup, 3> loc_dims_3, 
        BcParams params
      ) {
        return std::unique_ptr<CopyTensor>(new CopyTensor(env, dis_dims_3, loc_dims_3, params));
      }

      /// @brief Create a copy of the tensor. 
      /// @return Tptr to duplicated tensor. 
      /// 
      /// Overuse may cause memory shortage. 
      virtual qtnh::tptr copy() const noexcept override;

      virtual TT type() const noexcept override { return TT::copyTensor; }

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
      /// @brief Construct copy tensor with default distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param dis_dims_3 Triple of distributed dimensions for each input. 
      /// @param loc_dims_3 Triple of local dimensions for each input. 
      CopyTensor(const QTNHEnv& env, std::array<tidx_tup, 3> dis_dims_3, 
                 std::array<tidx_tup, 3> loc_dims_3);
      /// @brief Construct copy tensor. 
      /// @param env Environment to use for construction. 
      /// @param dis_dims_3 Triple of distributed dimensions for each input. 
      /// @param loc_dims_3 Triple of local dimensions for each input. 
      /// @param params Distribution parameters (str, cyc, off). 
      CopyTensor(const QTNHEnv& env, std::array<tidx_tup, 3> dis_dims_3, 
                 std::array<tidx_tup, 3> loc_dims_3, BcParams params);

      /// @brief Re-broadcast current tensor. 
      /// @param params Broadcast parameters of the tensor (str, cyc, off)
      /// @return Pointer to re-broadcasted tensor, which might be of a different derived type. 
      virtual CopyTensor* rebcast(BcParams params) override;
    
    private:
      std::array<tidx_tup, 3> dis_dims_3_;
      std::array<tidx_tup, 3> loc_dims_3_;
  };
}

#endif