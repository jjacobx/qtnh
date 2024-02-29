#ifndef _TENSOR__INDEXING_HPP
#define _TENSOR__INDEXING_HPP

#include "../core/typedefs.hpp"

namespace qtnh{
  namespace ops {
    std::ostream& operator<<(std::ostream&, const qtnh::tidx_tup&);
  }

  class TIndexing {
    private:
      qtnh::tidx_tup dims;
      qtnh::tifl_tup ifls;
    
    public:
      TIndexing();
      TIndexing(const qtnh::tidx_tup&);
      TIndexing(const qtnh::tidx_tup&, std::size_t);
      TIndexing(const qtnh::tidx_tup&, const qtnh::tifl_tup&);

      const qtnh::tidx_tup& getDims() const;
      const qtnh::tifl_tup& getIFls() const;

      bool isValid(const qtnh::tidx_tup&);
      bool isEqual(const qtnh::tidx_tup&, const qtnh::tidx_tup&, TIdxT = TIdxT::open, qtnh::tidx_tup_st = 0);
      bool isLast(const qtnh::tidx_tup&, TIdxT = TIdxT::open, qtnh::tidx_tup_st = 0);

      qtnh::tidx_tup& next(qtnh::tidx_tup&, TIdxT = TIdxT::open, qtnh::tidx_tup_st = 0);
      qtnh::tidx_tup& prev(qtnh::tidx_tup&, TIdxT = TIdxT::open, qtnh::tidx_tup_st = 0);
      qtnh::tidx_tup& reset(qtnh::tidx_tup&, TIdxT = TIdxT::open, qtnh::tidx_tup_st = 0);

      TIndexing cut(TIdxT = TIdxT::open, qtnh::tidx_tup_st = 0);
      TIndexing cut_all(TIdxT = TIdxT::open);

      bool operator==(const TIndexing&);
      bool operator!=(const TIndexing&);

      class iterator {
        private:
          qtnh::tidx_tup dims;
          qtnh::tifl_tup ifls;

          qtnh::tidx_tup current;
          qtnh::tifl active_ifl;

        public:
          iterator(const qtnh::tidx_tup&, const qtnh::tifl_tup&, const qtnh::tidx_tup&, TIdxT = TIdxT::open, qtnh::tidx_tup_st = 0);

          const qtnh::tifl& getActiveIFl() const;

          iterator& operator++();
          bool operator!=(const iterator&);
          qtnh::tidx_tup operator*() const;
      };

      iterator begin(TIdxT = TIdxT::open, qtnh::tidx_tup_st = 0);
      iterator end();

      static TIndexing app(const TIndexing&, const TIndexing&);
  };  
}

#endif