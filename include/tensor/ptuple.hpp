#ifndef __TENSOR_PTUPLE__
#define __TENSOR_PTUPLE__

#include <map>
#include "core/typedefs.hpp"

namespace qtnh {
  class PTuple {
    public:
      PTuple() = delete;
      PTuple(std::size_t len);
      ~PTuple() = default;

      const auto& tup() const noexcept { return tup_; }

      PTuple operator*(const PTuple& ptup);
      PTuple inv();

      class shifter {
        public:
          shifter() = delete;
          shifter(std::vector<qtnh::tidx_tup_st>& tup, std::size_t pos);
          shifter(std::vector<qtnh::tidx_tup_st>& tup, std::size_t from, std::size_t to);
          ~shifter() = default;

          void operator>>(int n);
          void operator<<(int n);
        
        private:
          std::vector<qtnh::tidx_tup_st>& tup_;
          std::size_t from_;
          std::size_t to_;
      };

      shifter at(std::size_t pos);
      shifter at(std::size_t from, std::size_t to);

    private:
      std::vector<qtnh::tidx_tup_st> tup_;

  };
}

#endif