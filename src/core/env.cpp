#include <iostream>
#include <mpi.h>
#include <omp.h>

#include "core/env.hpp"
#include "core/typedefs.hpp"

namespace qtnh {
  QTNHEnv::QTNHEnv() {
    num_threads = omp_get_max_threads();

    int _proc_id, _num_processes;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &_proc_id);
    MPI_Comm_size(MPI_COMM_WORLD, &_num_processes);

    proc_id = _proc_id;
    num_processes = _num_processes;
  }

  QTNHEnv::~QTNHEnv() {
    MPI_Finalize();
  }

  void QTNHEnv::print() const {
    std::cout << "Process ID: " << proc_id << std::endl;
    std::cout << "Process count: " << num_processes << std::endl;
    std::cout << "Thread count: " << num_threads << std::endl;
  }
}