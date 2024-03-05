#ifndef _TENSOR__INDEXING_HPP
#define _TENSOR__INDEXING_HPP

#include "../core/typedefs.hpp"

namespace qtnh{
  namespace ops {
    /// Print tensor index tuple via std::cout. 
    std::ostream& operator<<(std::ostream&, const qtnh::tidx_tup&);
  }

  /// This class can be used to easily iterate through tensor index tuples based on given restrictions 
  /// due to dimensions or index flags. 
  class TIndexing {
    private:
      qtnh::tidx_tup dims;  ///< Dimensions of each element of the indexing. 
      qtnh::tifl_tup ifls;  ///< Flags of each element of the indexing. 
    
    public:
      /// Empty constructor creates a single element indexing of dimension 1. 
      TIndexing();
      /// @brief Create indexing with given dimensions and all elements open. 
      /// @param dims Index dimensions of the indexing. 
      TIndexing(const qtnh::tidx_tup& dims);
      /// @brief Create indexing with given dimensions and all elements open except n. 
      /// @param dims Index dimensions of the indexing. 
      /// @param n Position of the closed index. 
      ///
      /// Index n is assigned { TIdxT::closed, 0 }. 
      TIndexing(const qtnh::tidx_tup& dims, std::size_t n);
      /// @brief Creat indexing with given dimensions and flags for each index. 
      /// @param dims Index dimensions of the indexing. 
      /// @param ifls Index flags of the indexing. 
      TIndexing(const qtnh::tidx_tup& dims, const qtnh::tifl_tup& ifls);

      const qtnh::tidx_tup& getDims() const;  ///< Index dimensions getter. 
      const qtnh::tifl_tup& getIFls() const;  ///< Index flags getter. 

      /// @brief Check if tensor index tuple is valid in current indexing. 
      /// @param idxs Tensor index tuple to check.
      /// @return A boolean, true if the tuple is valid and false otherwise. 
      bool isValid(const qtnh::tidx_tup& idxs);
      /// @brief Check if two tensor index tuples are equal in current indexing. 
      /// @param idxs1 First tensor index tuple to compare. 
      /// @param idxs2 Second tensor index tuple to compare. 
      /// @param type Type of the index flag to check. 
      /// @param tag Tag of the index flag to check. 
      /// @return A boolean, true if the tuples are equal and false otherwise. 
      ///
      /// Only indices with the given flag are checked. 
      bool isEqual(const qtnh::tidx_tup& idxs1, const qtnh::tidx_tup& idxs2, TIdxT type = TIdxT::open, qtnh::tidx_tup_st tag = 0);
      /// @brief Check if tensor index tuple is the last allowed in current indexing. 
      /// @param idxs Tensor index tuple to check. 
      /// @param type Type of the index flag to check. 
      /// @param tag Tag of the index flag to check. 
      /// @return A boolean, true if the tuple is the last one and false otherwise. 
      ///
      /// Only indices with the given flag are checked. 
      bool isLast(const qtnh::tidx_tup& idxs, TIdxT type = TIdxT::open, qtnh::tidx_tup_st tag = 0);

      /// @brief Go to next tensor index tuple in current indexing, relative to the input tuple.  
      /// @param idxs Tensor index tuple to update. 
      /// @param type Type of the index flag to check. 
      /// @param tag Tag of the index flag to check. 
      /// @return Next tensor index tuple. 
      ///
      /// Only indices with the given flag are updated. 
      qtnh::tidx_tup& next(qtnh::tidx_tup& idxs, TIdxT type = TIdxT::open, qtnh::tidx_tup_st tag = 0);
      /// @brief Go to previous tensor index tuple in current indexing, relative to the input tuple.  
      /// @param idxs Tensor index tuple to update. 
      /// @param type Type of the index flag to check. 
      /// @param tag Tag of the index flag to check. 
      /// @return Previous tensor index tuple. 
      ///
      /// Only indices with the given flag are updated. 
      qtnh::tidx_tup& prev(qtnh::tidx_tup& idxs, TIdxT type = TIdxT::open, qtnh::tidx_tup_st tag = 0);
      /// @brief Reset tensor index tuple to zero in current indexing.  
      /// @param idxs Tensor index tuple to update. 
      /// @param type Type of the index flag to check. 
      /// @param tag Tag of the index flag to check. 
      /// @return Reset tensor index tuple. 
      ///
      /// Only indices with the given flag are updated. 
      qtnh::tidx_tup& reset(qtnh::tidx_tup& idxs, TIdxT type = TIdxT::open, qtnh::tidx_tup_st tag = 0);

      /// @brief Cut given tensor index type out of the indexing. 
      /// @param type Tensor index type to cut out. 
      /// @return Trimmed indexing without the indicated type. 
      TIndexing cut(TIdxT type = TIdxT::open);
      /// @brief Cut given tensor index flag out of the indexing. 
      /// @param ifl Tensor index flag to cut out, i.e. a pair of an index type and a tag. 
      /// @return Trimmed indexing without the indicated flag. 
      TIndexing cut(qtnh::tifl ifl);

      bool operator==(const TIndexing&);  ///< Equality operator for the indexing. 
      bool operator!=(const TIndexing&);  ///< Inequality operator for the indexing. 

      /// Tensor indexing iterator class to enable for loops over given indexing. 
      class iterator {
        private:
          qtnh::tidx_tup dims;  ///< Dimensions of each element of related indexing. 
          qtnh::tifl_tup ifls;  ///< Flags of each element of related indexing. 

          qtnh::tidx_tup current;  ///< Current tensor index tuple. 
          qtnh::tifl active_ifl;   ///< Active tensor index flag to iterate over. 

        public:
          /// @brief General constructor of the tensor indexing iterator. 
          /// @param dims Tensor index dimensions of related indexing. 
          /// @param ifls Tensor infex flags of related indexing. 
          /// @param c_idx Current index in related indexing. 
          /// @param type Type of index flag to iterate over. 
          /// @param tag Tag of index flag to iterate over. 
          ///
          /// The constructed iterator only iterates over indices of given flag. 
          iterator(const qtnh::tidx_tup& dims, const qtnh::tifl_tup& ifls, const qtnh::tidx_tup& c_idx, TIdxT type = TIdxT::open, qtnh::tidx_tup_st tag = 0);

          const qtnh::tifl& getActiveIFl() const; ///< Active tensor index flag getter. 

          iterator& operator++();            ///< Next element operator. 
          bool operator!=(const iterator&);  ///< Inequality operator. 
          qtnh::tidx_tup operator*() const;  ///< Dereference operator. 
      };

      /// @brief Get begin iterator with given tensor index flag type and tag. 
      /// @param type Type of index flag to iterate over. 
      /// @param tag Tag of index flag to iterate over. 
      /// @return Tensor indexing iterator from zero element in current indexing. 
      iterator begin(TIdxT type = TIdxT::open, qtnh::tidx_tup_st tag = 0);
      /// @brief Get end iterator of current indexing. 
      /// @return Tensor indexing iterator with out-of-bounds elements. 
      iterator end();

      /// @brief Append two tensor indexings. 
      /// @param ti1 First tensor indexing to append. 
      /// @param ti2 Second tensor indexing to append. 
      /// @return Tensor indexing with appended parameters of both arguments. 
      static TIndexing app(const TIndexing& ti1, const TIndexing&ti2);
  };  
}

#endif