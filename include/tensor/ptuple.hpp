#ifndef __TENSOR_PTUPLE__
#define __TENSOR_PTUPLE__

#include <map>
#include "core/typedefs.hpp"

namespace qtnh {
  using tup_t = std::vector<qtnh::tidx_tup_st>;

  class PTuple {
    public:
      PTuple() = delete;
      PTuple(std::size_t len);
      PTuple(tup_t tup);
      ~PTuple() = default;

      const auto& tup() const noexcept { return tup_; }

      PTuple operator*(const PTuple& ptup);
      PTuple inv();

      class shifter {
        public:
          shifter() = delete;
          shifter(tup_t& tup, std::size_t pos);
          shifter(tup_t& tup, std::size_t from, std::size_t to);
          ~shifter() = default;

          void operator>>(int n);
          void operator<<(int n);
        
        private:
          tup_t& tup_;
          std::size_t from_;
          std::size_t to_;
      };

      shifter at(std::size_t pos);
      shifter at(std::size_t from, std::size_t to);

    private:
      tup_t tup_;

  };

  class IndexGroup {
    public:
      IndexGroup() = delete;
      IndexGroup(std::vector<std::string> labels, std::vector<tup_t> groups);
      ~IndexGroup() = default;

      const std::vector<std::string>& labels() const noexcept { return labels_; }
      PTuple ptup() const;

      qtnh::tidx_tup_st& at(std::string k, std::size_t i);
      void reorder(std::vector<std::string> labels);

    private:
      std::vector<std::string> labels_;
      std::map<std::string, tup_t> groups_;
  };
}

#endif