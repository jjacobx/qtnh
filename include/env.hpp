#ifndef ENV_HPP
#define ENV_HPP

#include <mpi.h>

namespace qtnh {
  struct QTNHEnv {
    public:
      unsigned int proc_id;
      unsigned int num_processes;
      unsigned int num_threads;

      MPI_Datatype swap_types[4][32];
      bool swap_types_committed[4][32];

      QTNHEnv();
      ~QTNHEnv();

      void print() const;

    private:
      void init_swap_types();
      void free_swap_types();
  };
}

#endif