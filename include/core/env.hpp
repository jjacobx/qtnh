#ifndef _CORE__ENV_HPP
#define _CORE__ENV_HPP

#include <mpi.h>

namespace qtnh {
  struct QTNHEnv {
    public:
      unsigned int proc_id;
      unsigned int num_processes;
      unsigned int num_threads;

      MPI_Datatype swap_p2_types[32];
      bool swap_p2_types_committed[32];

      QTNHEnv();
      ~QTNHEnv();

      void print() const;

    private:
      void init_swap_p2_types();
      void free_swap_p2_types();
  };
}

#endif