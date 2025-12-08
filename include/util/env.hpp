#ifndef QTNH_UTIL_ENV_HPP_INCLUDE
#define QTNH_UTIL_ENV_HPP_INCLUDE

#include <mpi.h>

namespace qtnh {
  /// QTNH environment, responsible for keeping track of MPI and OpenMP. 
  /// An instance must be created at the beginning of the program. 
  /// It should be passed to classes that work in parallel, e.g. Tensor
  struct QTNHEnv {
    public:
      unsigned int proc_id;        ///< ID of calling process. 
      unsigned int num_processes;  ///< Number of MPI processes. 
      // unsigned int num_threads;    ///< Number of OpenMP threads. 

      inline static unsigned int num_comms = 1;

      /// Default constructor. 
      /// Initialises MPI and populates struct members accordingly. 
      QTNHEnv();

      QTNHEnv(const QTNHEnv&) = delete;
      QTNHEnv& operator=(const QTNHEnv&) = delete;

      QTNHEnv(QTNHEnv&&) = delete;
      QTNHEnv& operator=(QTNHEnv&&) = delete;

      /// Default destructor. 
      /// Finalises MPI. 
      ~QTNHEnv();

      /// Prints environment information. 
      void print() const;
  };
}

#endif