#ifndef __TEN_TYPE_TENSOR__
#define __TEN_TYPE_TENSOR__

#include <memory>

#include "util/env.hpp"
#include "util/typedefs.hpp"
#include "util/utils.hpp"

namespace qtnh {
  class Tensor;
  typedef std::unique_ptr<Tensor> tptr;

  // Forward declaration of these is required for conversion methods. 
  class DenseTensor; 
  class SymmTensor;
  class DiagTensor;

  /// Tensor type labels for determining contraction function to use. 
  enum class TT {
    tensor, 
    denseTensorBase, 
    denseTensor, 
    rescTensor, 
    symmTensorBase, 
    symmTensor, 
    swapTensor, 
    diagTensorBase, 
    diagTensor, 
    idenTensor
  };

  /// Broadcaster parameters container for sharing tensors across processes. 
  struct BcParams {
    qtnh::uint str;  ///< Number of times each local tensor chunk is repeated across contiguous processes. 
    qtnh::uint cyc;  ///< Number of times the entire tensor structure is repeated. 
    qtnh::uint off;  ///< Number of empty processes before the tensor begins. 
  };
  
  struct ConParams {
    public:
      ConParams() = delete;
      ConParams(std::vector<qtnh::wire> wires) :
        wires(wires), useDefRepls(true), dimRepls1(0), dimRepls2(0) {}
      ConParams(std::vector<qtnh::wire> wires, std::vector<qtnh::tidx_tup_st> dim_repls1, std::vector<qtnh::tidx_tup_st> dim_repls2) :
        wires(wires), useDefRepls(false), dimRepls1(dim_repls1), dimRepls2(dim_repls2) {}
      ~ConParams() = default;

      std::vector<qtnh::wire> wires;

      bool useDefRepls;

      // TODO: reimplement the following as maps. 
      std::vector<qtnh::tidx_tup_st> dimRepls1;
      std::vector<qtnh::tidx_tup_st> dimRepls2;
  };


  /// @brief Tensor broadcaster class responsible for handling how tensor is shared in distributed memory. 
  class Broadcaster {
    public:
      Broadcaster() = delete;
      Broadcaster(Broadcaster& bc) = delete;

      Broadcaster(Broadcaster&& bc);
      Broadcaster(const QTNHEnv& env, qtnh::uint base, BcParams params);
      Broadcaster(const QTNHEnv& env, qtnh::uint base, BcParams params, bool create_comm);
      ~Broadcaster();

      Broadcaster& operator=(Broadcaster&& b) noexcept;
      Broadcaster& operator=(Broadcaster& b) = delete;

      constexpr qtnh::uint base() const { return base_; }
      constexpr bool hasComm() const { return has_comm_; }
      constexpr bool isActive() const { return is_active_; }
      constexpr int gid() const { return gid_; }

      const QTNHEnv& env() const {return env_; }
      const MPI_Comm& gcomm();

      /// @brief Helper to return all params at once. 
      /// @return Broadcaster parameters struct. 
      constexpr BcParams params() const { return { str_, cyc_, off_ }; }
      /// @brief Helper to calculate span of the entire tensor across contiguous ranks. 
      /// @return Number of contiguous ranks that store the tensor. 
      constexpr qtnh::uint span() const { return str_ * base_ * cyc_; }
      /// @brief Helper to calculate between which ranks the tensor is contained. 
      /// @return A tuple containing first and last rank that store the tensor. 
      constexpr std::pair<qtnh::uint, qtnh::uint> range() const { return { off_, off_ + span() }; }

      void createComm();  ///< Initialise group communicator. 
      void deleteComm();  ///< Free group communicator. 

    private:
      const QTNHEnv& env_;  ///< Environment to use MPI/OpenMP in. 
      qtnh::uint base_;     ///< Base distributed size of the tensor. 

      qtnh::uint str_;  ///< Number of times each local tensor chunk is repeated across contiguous processes. 
      qtnh::uint cyc_;  ///< Number of times the entire tensor structure is repeated. 
      qtnh::uint off_;  ///< Number of empty processes before the tensor begins. 

      bool has_comm_ = false;  ///< Flag whether group communicator has been initialised. 

      // The following may differ with rank so defaults are a must. 
      bool is_active_ = false;          ///< Flag whether the tensor is stored on calling MPI rank. 
      MPI_Comm gcomm_ = MPI_COMM_NULL;  ///< Communicator that contains exactly one copy of the tensor. 
      int gid_ = 0;                     ///< Rank ID within current group. 
  };

  /// General virtual tensor class
  class Tensor {
    public:
      Tensor() = delete;
      Tensor(const Tensor&) = delete;
      virtual ~Tensor() = default;

      // This can be made constexpr in C++ 20
      virtual TT type() const noexcept { return TT::tensor; }

      /// @brief Cast to derived tensor class. 
      /// @tparam T Derived tensor class to cast to. 
      /// @return Pointer to derived tensor class, nullptr if cast is not possible. 
      template<class T> T* cast() { return dynamic_cast<T*>(this); }
      /// @brief Cast and transfer ownership to derived unique pointer. 
      /// @tparam T Derived tensor class to cast to. 
      /// @param tp Ownership of tptr to tensor to cast. 
      /// @return Ownership of cast tensor unique pointer, nullptr if cast is not possible. 
      /// 
      /// Use this at your own risk, as the tensor will be destroyed if cast is unsuccessful. 
      template<class T>
      static std::unique_ptr<T> cast(qtnh::tptr tp) {
        return std::unique_ptr<T>(dynamic_cast<T*>(tp.release()));
      }

      /// @brief Check if conversion to given tensor class is possible. 
      /// @tparam T Tensor class to convert to. 
      /// @return True if conversion is possible, false otherwise. 
      template<class T>  bool canConvert() { return false; }
      /// @brief Convert to given tensor class. 
      /// @tparam T Tensor class to convert to. 
      /// @param tp Ownership of tptr to tensor to convert. 
      /// @return Ownership of tptr with converted tensor, nullptr if conversion is not possible. 
      template<class T> 
      static std::unique_ptr<T> convert(qtnh::tptr tp) { 
        return std::unique_ptr<T>(nullptr); 
      }

      /// @brief Create a copy of the tensor. 
      /// @return Tptr to duplicated tensor. 
      /// 
      /// Overuse may cause memory shortage. 
      virtual qtnh::tptr copy() const noexcept = 0;
      
      qtnh::tidx_tup locDims() const noexcept { return loc_dims_; }
      qtnh::tidx_tup disDims() const noexcept { return dis_dims_; }
      
      Broadcaster& bc() noexcept { return bc_; }
      const Broadcaster& bc() const noexcept { return bc_; }
      
      /// @brief Helper to access complete tensor dimensions. 
      /// @return Concatenated distributed and local dimensions. 
      qtnh::tidx_tup totDims() const { return utils::concat_dims(dis_dims_, loc_dims_); }
      
      /// @brief Helper to calculate size of the local part of the tensor. 
      /// @return Number of local elements in the tensor. 
      std::size_t locSize() const { return utils::dims_to_size(loc_dims_); }
      /// @brief Helper to calculate size of the distributed part of the tensor. 
      /// @return Number of ranks to which one instance of the tensor is distributed. 
      std::size_t disSize() const { return utils::dims_to_size(dis_dims_); }
      /// @brief Helper to calculate size of the entire tensor. 
      /// @return Number of elements in the entire tensor. 
      std::size_t totSize() const { return utils::dims_to_size(totDims()); }

      /// @brief Directly access local array the tensor (or corresponding element). 
      /// @param i Local array index to access. 
      /// @return The element at given index. Throws an error if not present (or out of bounds). 
      ///
      /// Returned element depends on the storage method used. It may differ for two identical 
      /// tensors that use different underlying classes. It may also produce unexpected results 
      /// when virtual elements are stored, i.e. elements useful for calculations, but not actually 
      /// present in the tensor. 
      virtual qtnh::tel operator[](std::size_t i) const = 0;
      /// @brief Rank-unsafe method to get element and given local indices. 
      /// @param idxs Tensor index tuple indicating local position of the element. 
      /// @return Value of the element at given indices. Throws error if value is not present. 
      /// @deprecated Will be superseded by direct addressing of array elements with numeric indices. 
      ///
      /// This method requires ensuring the element is present (i.e. the tensor is active)
      /// on current rank. On all active ranks, it must return an element, but different ranks  
      /// might have different values. 
      virtual qtnh::tel operator[](qtnh::tidx_tup loc_idxs) const = 0;

      /// @brief Access element at total indices if present. 
      /// @param tot_idxs Indices with total position of the element. 
      /// @return Value of the element at given indices. Throws error if value is not present. 
      ///
      /// It is advised to ensure the element is present at current rank with Tensor::has method. 
      virtual qtnh::tel at(qtnh::tidx_tup tot_idxs) const = 0;
      /// @brief Check if element at total indices is present on current rank. 
      /// @param tot_idxs Indices with total position of the element. 
      /// @return True if the element is present, false otherwise. 
      virtual bool has(qtnh::tidx_tup tot_idxs) const;
      /// @brief Fetch element at global indices and broadcast it to every rank. 
      /// @param tot_idxs Indices with total position of the element. 
      /// @return Value of the element at given indices. 
      ///
      /// This method doesn't require checking if the value is present or if the tensor is active. 
      /// Because of the broadcast, it is inefficient to use it too often. 
      virtual qtnh::tel fetch(qtnh::tidx_tup tot_idxs) const;

      void activate() { if (!bc_.hasComm()) bc_.createComm(); }
      void deactivate() { bc_.deleteComm(); }

      /// @brief Reshape distributed and local dimensions of a tensor. 
      /// @param dis_dims New distributed dimensions. 
      /// @param loc_dims New local dimensions. 
      ///
      /// Both distributed and local dimensions must be compatible with the old values. 
      virtual void reshape(qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims);

      /// @brief Swap indices on current tensor. 
      /// @param tp Ownership of tptr to tensor to swap. 
      /// @param idx1 First index to swap. 
      /// @param idx2 Second index to swap. 
      /// @return Ownership of tptr to swapped tensor. 
      static qtnh::tptr swap(qtnh::tptr tp, qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) {
        auto p = tp->swap(idx1, idx2);
        return utils::one_unique(std::move(tp), p);
      }
      /// @brief Re-broadcast current tensor. 
      /// @param tp Ownership of tptr to tensor to re-broadcast. 
      /// @param params Broadcast parameters of the tensor (str, cyc, off)
      /// @return Ownership of tptr to re-broadcasted tensor. 
      static qtnh::tptr rebcast(qtnh::tptr tp, BcParams params) {
        auto p = tp->rebcast(params);
        return utils::one_unique(std::move(tp), p);
      }
      /// @brief Shift the border between shared and distributed dimensions by a given offset. 
      /// @param tp Ownership of tptr to tensor to re-scatter. 
      /// @param offset New offset between distributed and local dimensions – negative gathers, while positive scatters. 
      /// @return Ownership of tptr to re-scattered tensor. 
      static qtnh::tptr rescatter(qtnh::tptr tp, int offset) {
        auto p = tp->rescatter(offset);
        return utils::one_unique(std::move(tp), p);
      }
      /// @brief Permute tensor indices according to mappings in the permutation tuple. 
      /// @param tp Ownership of tptr to tensor to permute. 
      /// @param ptup Permutation tuple of the same size as total dimensions, and each entry unique. 
      /// @return Ownership of tptr to permuted tensor. 
      static qtnh::tptr permute(qtnh::tptr tp, std::vector<qtnh::tidx_tup_st> ptup) {
        auto p = tp->permute(ptup);
        return utils::one_unique(std::move(tp), p);
      }
      /// @brief Truncate tensor index down to a given dimension. 
      /// @param tp Ownership of tptr to tensor to truncate. 
      /// @param idx Index to truncate. 
      /// @param size Target size of the index. 
      /// @return Ownership of tptr to truncated tensor. 
      static qtnh::tptr truncate(qtnh::tptr tp, qtnh::tidx_tup_st idx, std::size_t size) {
        auto p = tp->truncate(idx, size);
        return utils::one_unique(std::move(tp), p);
      }

    protected:
      /// @brief Construct empty tensor of zero size within environment and with default distribution parameters. 
      /// @param env Environment to use for construction. 
      Tensor(const QTNHEnv& env);
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with default distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      Tensor(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims);
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with given distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param params Distribution parameters of the tensor (str, cyc, off)
      Tensor(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, BcParams params);

      qtnh::tidx_tup dis_dims_;  ///< Distributed index dimensions. 
      qtnh::tidx_tup loc_dims_;  ///< Local index dimensions. 

      Broadcaster bc_;  ///< Tensor distributor. 

      virtual DenseTensor* toDense() noexcept { return nullptr; }
      virtual SymmTensor* toSymm() noexcept { return nullptr; }
      virtual DiagTensor* toDiag() noexcept { return nullptr; }

      virtual bool isDense() const noexcept { return false; }
      virtual bool isSymm() const noexcept { return false; }
      virtual bool isDiag() const noexcept { return false; }

      // ! The following methods only work if DenseTensor overrides all of them
      /// @brief Swap indices on current tensor. 
      /// @param idx1 First index to swap. 
      /// @param idx2 Second index to swap. 
      /// @return Pointer to swapped tensor, which might be of a different derived type. 
      virtual Tensor* swap(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) = 0;
      /// @brief Re-broadcast current tensor. 
      /// @param params Broadcast parameters of the tensor (str, cyc, off)
      /// @return Pointer to re-broadcasted tensor, which might be of a different derived type. 
      virtual Tensor* rebcast(BcParams params) = 0;
      /// @brief Shift the border between shared and distributed dimensions by a given offset. 
      /// @param offset New offset between distributed and local dimensions – negative gathers, while positive scatters. 
      /// @return Pointer to re-scattered tensor, which might be of a different derived type. 
      virtual Tensor* rescatter(int offset) = 0;
      /// @brief Permute tensor indices according to mappings in the permutation tuple. 
      /// @param ptup Permutation tuple of the same size as total dimensions, and each entry unique. 
      /// @return Pointer to permuted tensor, which might be of a different derived type. 
      virtual Tensor* permute(std::vector<qtnh::tidx_tup_st> ptup) = 0;
      /// @brief Truncate tensor index down to a given dimension. 
      /// @param idx Index to truncate. 
      /// @param size Target size of the index. 
      /// @return Pointer to truncated tensor, which might be of a different derived type. 
      virtual Tensor* truncate(qtnh::tidx_tup_st idx, std::size_t size) = 0;

  };

  // Specialised template declarations must be outside class scope. 
  template<> bool Tensor::canConvert<DenseTensor>();
  template<> bool Tensor::canConvert<SymmTensor>();
  template<> bool Tensor::canConvert<DiagTensor>();

  template<> std::unique_ptr<DenseTensor> Tensor::convert<DenseTensor>(qtnh::tptr tp);
  template<> std::unique_ptr<SymmTensor> Tensor::convert<SymmTensor>(qtnh::tptr tp);
  template<> std::unique_ptr<DiagTensor> Tensor::convert<DiagTensor>(qtnh::tptr tp);
}

#endif