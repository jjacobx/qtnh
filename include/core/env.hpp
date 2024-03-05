#ifndef _CORE__ENV_HPP
#define _CORE__ENV_HPP

#include <mpi.h>

namespace qtnh {
  /// QTNH environment, responsible for keeping track of MPI and OpenMP. 
  /// An instance must be created at the beginning of the program. 
  /// It should be passed to classes that work in parallel, e.g. Tensor
  struct QTNHEnv {
    public:
      unsigned int proc_id;        ///< ID of calling process. 
      unsigned int num_processes;  ///< Number of MPI processes. 
      unsigned int num_threads;    ///< Number of OpenMP threads. 

      MPI_Datatype swap_p2_types[32];
      bool swap_p2_types_committed[32];

      /// Default constructor. 
      /// Initialises MPI and populates struct members accordingly. 
      QTNHEnv();

      /// Default destructor. 
      /// Finalises MPI. 
      ~QTNHEnv();

      /// Prints environment information. 
      void print() const;

    private:
      void init_swap_p2_types();
      void free_swap_p2_types();
  };
}

#endif