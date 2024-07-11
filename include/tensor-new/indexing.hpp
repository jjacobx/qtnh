#ifndef _TENSOR_NEW__INDEXING_HPP
#define _TENSOR_NEW__INDEXING_HPP

#include <string>

#include "../core/typedefs.hpp"

namespace qtnh {
  struct TIFlag {
    std::string label;
    int tag;

    TIFlag(std::string label, int tag) : label(label), tag(tag) {}
  };

  class TIndexing {
    public:
      /// @brief Create indexing with given dimensions. 
      /// @param dims Index dimensions of the indexing. 
      TIndexing(qtnh::tidx_tup dims);
      /// @brief Create indexing with given dimensions, all with given flag. 
      /// @param dims Index dimensions of the indexing. 
      /// @param ifl Default index flag to be used. 
      TIndexing(qtnh::tidx_tup dims, TIFlag ifl);
      /// @brief Create indexing with given dimensions and flags. 
      /// @param dims Index dimensions of the indexing. 
      /// @param ifl_label Index flags of the indexing. 
      TIndexing(qtnh::tidx_tup dims, std::vector<TIFlag> ifls);

      /// @brief Access and/or modify given index flag. 
      /// @param i Position of the flag in the indexing. 
      /// @return Reference to given flag. 
      TIFlag& iflsAt(std::size_t i);

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
      qtnh::tidx_tup next(qtnh::tidx_tup idxs, TIdxT type = TIdxT::open, qtnh::tidx_tup_st tag = 0);
      /// @brief Go to previous tensor index tuple in current indexing, relative to the input tuple.  
      /// @param idxs Tensor index tuple to update. 
      /// @param type Type of the index flag to check. 
      /// @param tag Tag of the index flag to check. 
      /// @return Previous tensor index tuple. 
      ///
      /// Only indices with the given flag are updated. 
      qtnh::tidx_tup prev(qtnh::tidx_tup idxs, TIdxT type = TIdxT::open, qtnh::tidx_tup_st tag = 0);
      /// @brief Reset tensor index tuple to zero in current indexing.  
      /// @param idxs Tensor index tuple to update. 
      /// @param type Type of the index flag to check. 
      /// @param tag Tag of the index flag to check. 
      /// @return Reset tensor index tuple. 
      ///
      /// Only indices with the given flag are updated. 
      qtnh::tidx_tup reset(qtnh::tidx_tup idxs, TIdxT type = TIdxT::open, qtnh::tidx_tup_st tag = 0);

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

      struct TupIterator {
        TupIterator(qtnh::tidx_tup dims, qtnh::tidx_tup start, std::vector<std::size_t> order);

        TupIterator begin();
        TupIterator end();

        TupIterator& operator++();
        bool operator!=(const TupIterator&);
        qtnh::tidx_tup operator*() const;

        private:
          qtnh::tidx_tup dims_;
          std::vector<qtnh::tidx_tup_st> order_;

          qtnh::tidx_tup current_;
      };

      struct NumIterator {
        NumIterator(qtnh::tidx_tup dims, std::vector<std::size_t> order, qtnh::tidx_tup start);

        NumIterator begin();
        NumIterator end();

        NumIterator& operator++();
        bool operator!=(const NumIterator&);
        std::size_t operator*() const;

        private:
          qtnh::tidx_tup dims_;  ///< Dimensions for the following increments (already sorted in order). 
          std::vector<std::size_t> incrs_;
          std::size_t zero_;

          qtnh::tidx_tup current_idxs_;
      };

      TupIterator tup(std::string ifl_label, qtnh::tidx_tup start);
      NumIterator num(std::string ifl_label, qtnh::tidx_tup start);

      /// @brief Append two tensor indexings. 
      /// @param ti1 First tensor indexing to append. 
      /// @param ti2 Second tensor indexing to append. 
      /// @return Tensor indexing with appended parameters of both arguments. 
      static TIndexing app(const TIndexing& ti1, const TIndexing&ti2);

    private:
      qtnh::tidx_tup dims_;
      std::vector<TIFlag> ifls_;
  };
}

#endif