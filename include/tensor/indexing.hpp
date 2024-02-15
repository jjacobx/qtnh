#ifndef INDEXING_HPP
#define INDEXING_HPP

#include <memory>

#include "../core/typedefs.hpp"

namespace qtnh{
  namespace ops {
    std::ostream& operator<<(std::ostream&, const qtnh::tidx_tup&);
  }

  class TIndexing {
    private:
      qtnh::tidx_tup dims;
      tidx_flags flags;
    
    public:
      TIndexing();
      TIndexing(const qtnh::tidx_tup&);
      TIndexing(const qtnh::tidx_tup&, std::size_t);
      TIndexing(const qtnh::tidx_tup&, const tidx_flags&);

      const qtnh::tidx_tup& getDims() const;
      const tidx_flags& getFlags() const;

      bool isValid(const qtnh::tidx_tup&);
      bool isEqual(const qtnh::tidx_tup&, const qtnh::tidx_tup&, TIdxFlag = TIdxFlag::open);
      bool isLast(const qtnh::tidx_tup&, TIdxFlag = TIdxFlag::open);

      qtnh::tidx_tup& next(qtnh::tidx_tup&, TIdxFlag = TIdxFlag::open);
      qtnh::tidx_tup& prev(qtnh::tidx_tup&, TIdxFlag = TIdxFlag::open);
      qtnh::tidx_tup& reset(qtnh::tidx_tup&, TIdxFlag = TIdxFlag::open);

      TIndexing cut(TIdxFlag = TIdxFlag::open);

      bool operator==(const TIndexing&);
      bool operator!=(const TIndexing&);

      class iterator {
        private:
          qtnh::tidx_tup dims;
          tidx_flags flags;

          qtnh::tidx_tup current;
          TIdxFlag active_flag;

        public:
          iterator(const qtnh::tidx_tup&, const tidx_flags&, const qtnh::tidx_tup&, TIdxFlag = TIdxFlag::open);

          const TIdxFlag& getActiveFlag() const;

          iterator& operator++();
          bool operator!=(const iterator&);
          qtnh::tidx_tup operator*() const;
      };

      iterator begin(TIdxFlag = TIdxFlag::open);
      iterator end();

      static TIndexing app(const TIndexing&, const TIndexing&);
  };  
}

#endif