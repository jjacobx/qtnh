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
      /// @brief Create indexing with given dimensions and flags. 
      /// @param dims Index dimensions of the indexing. 
      /// @param ifl_label Index flags of the indexing. 
      TIndexing(qtnh::tidx_tup dims, std::vector<TIFlag> ifls);

      const qtnh::tidx_tup& dims() const noexcept { return dims_; }
      const std::vector<TIFlag>& ifls() const noexcept { return ifls_; }

      /// @brief Check if tensor index tuple is valid in current indexing. 
      /// @param idxs Tensor index tuple to check.
      /// @return A boolean, true if the tuple is valid and false otherwise. 
      bool isValid(qtnh::tidx_tup idxs) const;
      /// @brief Check if two tensor index tuples are equal in current indexing. 
      /// @param idxs1 First tensor index tuple to compare. 
      /// @param idxs2 Second tensor index tuple to compare. 
      /// @param ifl_label Index flag label to check. 
      /// @return A boolean, true if the tuples are equal and false otherwise. 
      ///
      /// Only indices with the given flag are checked. 
      bool isEqual(qtnh::tidx_tup idxs1, qtnh::tidx_tup idxs2, std::string ifl_label = "default") const;
      /// @brief Check if tensor index tuple is the last allowed in current indexing. 
      /// @param idxs Tensor index tuple to check. 
      /// @param ifl_label Index flag label to check. 
      /// @return A boolean, true if the tuple is the last one and false otherwise. 
      ///
      /// Only indices with the given flag are checked. 
      bool isLast(qtnh::tidx_tup idxs, std::string ifl_label = "default") const;

      /// @brief Go to next tensor index tuple in current indexing, relative to the input tuple.  
      /// @param idxs Tensor index tuple to update. 
      /// @param ifl_label Index flag label to update. 
      /// @return Next tensor index tuple. 
      ///
      /// Only indices with the given flag are updated. 
      qtnh::tidx_tup next(qtnh::tidx_tup idxs, std::string ifl_label = "default") const;
      /// @brief Go to previous tensor index tuple in current indexing, relative to the input tuple.  
      /// @param idxs Tensor index tuple to update. 
      /// @param ifl_label Index flag label to update. 
      /// @return Previous tensor index tuple. 
      ///
      /// Only indices with the given flag are updated. 
      qtnh::tidx_tup prev(qtnh::tidx_tup idxs, std::string ifl_label = "default") const;
      /// @brief Reset tensor index tuple to zero in current indexing. 
      /// @param idxs Tensor index tuple to update. 
      /// @param ifl_label Index flag label to update. 
      /// @return Reset tensor index tuple. 
      ///
      /// Only indices with the given flag are updated. 
      qtnh::tidx_tup reset(qtnh::tidx_tup idxs, std::string ifl_label = "default") const;

      /// @brief Cut given label type out of the indexing. 
      /// @param ifl_label Index flag label to cut out. 
      /// @return Trimmed indexing without the indicated label. 
      TIndexing cut(std::string ifl_label = "default") const;

      /// @brief Keep only given label in the indexing. 
      /// @param ifl_label Index flag label to keep. 
      /// @return Trimmed indexing without any other label. 
      TIndexing keep(std::string ifl_label = "default") const;

      struct TupIterator {
        TupIterator(qtnh::tidx_tup dims, std::vector<std::size_t> order, qtnh::tidx_tup start, bool is_end);

        TupIterator begin();
        TupIterator end();

        TupIterator& operator++();
        TupIterator operator++(int);
        constexpr bool operator!=(const TupIterator& rhs) {
          // Only valid for checking the end element. 
          return (is_end_ != rhs.is_end_);
        }
        qtnh::tidx_tup operator*() const;

        private:
          qtnh::tidx_tup dims_;
          std::vector<std::size_t> order_;

          qtnh::tidx_tup current_;
          bool is_end_;
      };

      struct NumIterator {
        NumIterator(qtnh::tidx_tup dims, std::vector<std::size_t> incrs, std::size_t zero, qtnh::tidx_tup current_idxs, bool is_end);

        NumIterator begin();
        NumIterator end();

        NumIterator& operator++();
        NumIterator operator++(int);
        constexpr bool operator!=(const NumIterator& rhs) {
          // Only valid for checking the end element. 
          return (is_end_ != rhs.is_end_);
        }
        std::size_t operator*() const;

        private:
          qtnh::tidx_tup dims_;  ///< Dimensions for the following increments (already sorted in order). 
          std::vector<std::size_t> incrs_;
          std::size_t zero_;

          qtnh::tidx_tup current_idxs_;
          bool is_end_;
      };

      TupIterator tup(std::string ifl_label = "default") const;
      TupIterator tup(std::string ifl_label, qtnh::tidx_tup start) const;
      NumIterator num(std::string ifl_label = "default") const;
      NumIterator num(std::string ifl_label, qtnh::tidx_tup start) const;

      template<typename T>
      static T app(T t) {
        return t;
      }

      /// @brief Append multiple instances of tensor indexing. 
      /// @param t First indexing instance to append. 
      /// @param args All other indexing instances to append
      /// @return Tensor indexing with appended parameters of all arguments. 
      template<typename T, typename... Args>
      static T app(T t, Args... args) {
        return _app(t, app(args...));
      }

    private:
      qtnh::tidx_tup dims_;
      std::vector<TIFlag> ifls_;
      std::vector<std::size_t> maps_;

      static TIndexing _app(const TIndexing& ti1, const TIndexing& ti2);
  };
}

#endif