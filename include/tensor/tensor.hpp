#ifndef _TENSOR_NEW__TENSOR_HPP
#define _TENSOR_NEW__TENSOR_HPP

#include <memory>

#include "core/env.hpp"
#include "core/typedefs.hpp"
#include "core/utils.hpp"

namespace qtnh {
  class Tensor;
  typedef std::unique_ptr<Tensor> tptr;

  // Forward declaration of these is required for conversion methods. 
  class DenseTensor; 
  class SymmTensor;
  class DiagTensor;

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

  /// General virtual tensor class
  class Tensor {
    public:
      Tensor() = delete;
      Tensor(const Tensor&) = delete;
      virtual ~Tensor() = default;

      /// @brief Tensor broadcaster class responsible for handling how tensor is shared in distributed memory. 
      struct Broadcaster {
        const QTNHEnv& env;   ///< Environment to use MPI/OpenMP in. 
        qtnh::uint base;      ///< Base distributed size of the tensor. 

        qtnh::uint str;       ///< Number of times each local tensor chunk is repeated across contiguous processes. 
        qtnh::uint cyc;       ///< Number of times the entire tensor structure is repeated. 
        qtnh::uint off;       ///< Number of empty processes before the tensor begins. 

        // The following may differ with rank so defaults are a must. 
        bool active = false;                  ///< Flag whether the tensor is stored on calling MPI rank. 
        MPI_Comm group_comm = MPI_COMM_NULL;  ///< Communicator that contains exactly one copy of the tensor. 
        int group_id = 0;                     ///< Rank ID within current group. 

        Broadcaster() = delete;
        Broadcaster(const QTNHEnv& env, qtnh::uint base, BcParams params);
        ~Broadcaster();

        Broadcaster& operator=(Broadcaster&& b) noexcept;

        /// @brief Helper to calculate span of the entire tensor across contiguous ranks. 
        /// @return Number of contiguous ranks that store the tensor. 
        qtnh::uint span() const noexcept { return str * base * cyc; }
        /// @brief Helper to calculate between which ranks the tensor is contained. 
        /// @return A tuple containing first and last rank that store the tensor. 
        std::pair<qtnh::uint, qtnh::uint> range() const noexcept { return { off, off + span() }; }
      };

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
      template<class T>  static std::unique_ptr<T> convert(tptr tp) { return std::unique_ptr<T>(nullptr); }

      /// @brief Create a copy of the tensor. 
      /// @return Tptr to duplicated tensor. 
      /// 
      /// Overuse may cause memory shortage. 
      virtual qtnh::tptr copy() const noexcept = 0;
      
      qtnh::tidx_tup locDims() const noexcept { return loc_dims_; }
      qtnh::tidx_tup disDims() const noexcept { return dis_dims_; }
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

      /// @brief Access element at total indices if pr      virtual bool isSymm() const noexcept override { return true; }esent. 
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

      /// @brief Swap indices on current tensor. 
      /// @param tu Unique pointer to the tensor. 
      /// @param idx1 First index to swap. 
      /// @param idx2 Second index to swap. 
      /// @return Unique to swapped tensor, which might be of a different derived type. 
      static qtnh::tptr swap(qtnh::tptr tu, qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) {
        return utils::one_unique(std::move(tu), tu->swap(idx1, idx2));
      }
      /// @brief Re-broadcast current tensor. 
      /// @param tu Unique pointer to the tensor. 
      /// @param params Broadcast parameters of the tensor (str, cyc, off)
      /// @return Unique pointer to redistributed tensor, which might be of a different derived type. 
      static qtnh::tptr rebcast(qtnh::tptr tu, BcParams params) {
        return utils::one_unique(std::move(tu), tu->rebcast(params));
      }
      /// @brief Shift the border between shared and distributed dimensions by a given offset. 
      /// @param tu Unique pointer to the tensor. 
      /// @param offset New offset between distributed and local dimensions – negative gathers, while positive scatters. 
      /// @return Pointer to re-scattered tensor, which might be of a different derived type. 
      static qtnh::tptr rescatter(qtnh::tptr tu, int offset) {
        return utils::one_unique(std::move(tu), tu->rescatter(offset));
      }
      /// @brief Permute tensor indices according to mappings in the permutation tuple. 
      /// @param tu Unique pointer to the tensor. 
      /// @param ptup Permutation tuple of the same size as total dimensions, and each entry unique. 
      /// @return Unique pointer to permuted tensor, which might be of a different derived type. 
      static qtnh::tptr permute(qtnh::tptr tu, std::vector<qtnh::tidx_tup_st> ptup) {
        return utils::one_unique(std::move(tu), tu->permute(ptup));
      }

      /// @brief Contract two tensors via given wires. 
      /// @param t1u Unique pointer to first tensor to contract. 
      /// @param t2u Unique pointer to second tensor      virtual bool isSymm() const noexcept override { return true; } to contract. 
      /// @param ws A vector of wires which indicate which pairs of indices to sum over. 
      /// @return Contracted tensor unique pointer. 
      static qtnh::tptr contract(qtnh::tptr t1u, qtnh::tptr t2u, ConParams& params);

      static qtnh::tptr contract(qtnh::tptr t1u, qtnh::tptr t2u, std::vector<qtnh::wire> wires) {
        ConParams params(wires);
        return contract(std::move(t1u), std::move(t2u), params);
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
  };

  // Specialised template declarations must be outside class scope. 
  template<> bool Tensor::canConvert<DenseTensor>();
  template<> bool Tensor::canConvert<SymmTensor>();
  template<> bool Tensor::canConvert<DiagTensor>();

  template<> std::unique_ptr<DenseTensor> Tensor::convert<DenseTensor>(tptr tp);
  template<> std::unique_ptr<SymmTensor> Tensor::convert<SymmTensor>(tptr tp);
  template<> std::unique_ptr<DiagTensor> Tensor::convert<DiagTensor>(tptr tp);

  namespace ops {
    /// Print tensor elements via std::cout. 
    std::ostream& operator<<(std::ostream&, const Tensor&);
  }
}

#endif