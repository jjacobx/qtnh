#ifndef __TENSOR_PTUPLE__
#define __TENSOR_PTUPLE__

#include <map>
#include "core/typedefs.hpp"

namespace qtnh {
  using tup_t = std::vector<qtnh::tidx_tup_st>;

  class PTupleTar;
  class PTupleSrc;

  class PTuple {
    public:
      PTuple() = delete;
      PTuple(std::size_t len);
      PTuple(tup_t tup);
      ~PTuple() = default;

      const auto& tup() const noexcept { return tup_; }

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

      virtual PTupleTar toTar() const = 0;
      virtual PTupleSrc toSrc() const = 0;

    private:
      tup_t tup_;
  };

  class PTupleTar : public PTuple {
    public:
      using PTuple::PTuple;
      ~PTupleTar() = default;

      virtual PTupleTar toTar() const override;
      virtual PTupleSrc toSrc() const override;

      PTupleTar operator*(const PTupleTar& ptup) const;
      PTupleTar inv() const;
  };

  class PTupleSrc : public PTuple {
    public:
    using PTuple::PTuple;
      ~PTupleSrc() = default;

      virtual PTupleTar toTar() const override;
      virtual PTupleSrc toSrc() const override;

      PTupleSrc operator*(const PTupleSrc& ptup) const;
      PTupleSrc inv() const;
  };

  class IndexGroup {
    public:
      IndexGroup() = delete;
      IndexGroup(std::vector<std::string> labels, std::vector<tup_t> groups);
      ~IndexGroup() = default;

      const std::vector<std::string>& labels() const noexcept { return labels_; }
      PTupleSrc ptup() const;

      tup_t& at(std::string k);
      qtnh::tidx_tup_st& at(std::string k, std::size_t i);
      void reorder(std::vector<std::string> labels);

    private:
      std::vector<std::string> labels_;
      std::map<std::string, tup_t> groups_;
  };
}

#endif