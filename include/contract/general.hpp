#ifndef __CONTRACT_GENERAL
#define __CONTRACT_GENERAL

#include "tensor/tensor.hpp"

namespace qtnh {
  class Contractor {
    protected: 
      qtnh::tptr tp1;
      qtnh::tptr tp2;
  };

  class GTCon : public Contractor {

  };

  class GSIpCon : public Contractor {

  };

  class GDIpCon : public Contractor {

  };

  class SSIpCon : public Contractor {

  };

  class SDIpCon : public Contractor {

  };

  class DDIpCon : public Contractor {

  };

  class GSelfCon : public Contractor {

  };

  class SSelfCon : public Contractor {

  };

  class DSelfCon : public Contractor {

  };
}

#endif