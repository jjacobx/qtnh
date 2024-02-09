#include <iostream>
#include <mpi.h>
#include <omp.h>

#include "env.hpp"

namespace qtnh {
  QTNHEnv::QTNHEnv() {
    this->num_threads = omp_get_max_threads();

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &this->proc_id);
    MPI_Comm_size(MPI_COMM_WORLD, &this->num_processes);
  }

  QTNHEnv::~QTNHEnv() {
    MPI_Finalize();
  }

  void QTNHEnv::print() const {
    std::cout << "Process ID: " << this->proc_id << std::endl;
    std::cout << "Process count: " << this->num_processes << std::endl;
    std::cout << "Thread count: " << this-> num_threads << std::endl;
  }
}