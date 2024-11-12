#ifndef __CONTRACT_GENERAL
#define __CONTRACT_GENERAL

#include "tensor/tensor.hpp"

namespace qtnh {
  class Contractor {
    public: 
      Contractor() = delete;
      Contractor(const Contractor&) = delete;

      Contractor(Contractor&&) = default;
      virtual ~Contractor() = default;

      Contractor& operator=(const Contractor&) = delete;
      Contractor& operator=(Contractor&&) = default;

      static Contractor make(qtnh::tptr tp1, qtnh::tptr tp2, ConParams params);
      virtual qtnh::tptr contract() = 0;

      const ConParams& params() const noexcept { return params_; }

    protected:
      Contractor(qtnh::tptr tp1, qtnh::tptr tp2, ConParams params)
        : tp1_(std::move(tp1)), tp2_(std::move(tp2)), params_(params) {}

      qtnh::tptr tp1_;
      qtnh::tptr tp2_;

      ConParams params_;
  };

  class GTCon : public Contractor {
    public:
      GTCon() = delete;
      GTCon(const GTCon&) = delete;

      GTCon(qtnh::tptr tp1, qtnh::tptr tp2, ConParams params)
        : Contractor(std::move(tp1), std::move(tp2), params) {}

      GTCon(GTCon&&) = default;
      ~GTCon() = default;

      GTCon& operator=(const GTCon&) = delete;
      GTCon& operator=(GTCon&&) = default;
    
      qtnh::tptr contract() override;
  };

  class GSIpCon : public Contractor {
    qtnh::tptr contract() override;
  };

  class GDIpCon : public Contractor {
    qtnh::tptr contract() override;
  };

  class SSIpCon : public Contractor {
    qtnh::tptr contract() override;
  };

  class SDIpCon : public Contractor {
    qtnh::tptr contract() override;
  };

  class DDIpCon : public Contractor {
    qtnh::tptr contract() override;
  };

  class GSelfCon : public Contractor {
    qtnh::tptr contract() override;
  };

  class SSelfCon : public Contractor {
    qtnh::tptr contract() override;
  };

  class DSelfCon : public Contractor {
    qtnh::tptr contract() override;
  };
}

#endif