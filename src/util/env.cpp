#include <iostream>
#include <mpi.h>

#include "util/env.hpp"

namespace qtnh {
  QTNHEnv::QTNHEnv() {
    int _proc_id, _num_processes;

    MPI_Init(0, 0);
    MPI_Comm_rank(MPI_COMM_WORLD, &_proc_id);
    MPI_Comm_size(MPI_COMM_WORLD, &_num_processes);

    proc_id = static_cast<unsigned int>(_proc_id);
    num_processes = static_cast<unsigned int>(_num_processes);
  }

  QTNHEnv::~QTNHEnv() {
    MPI_Finalize();
  }

  void QTNHEnv::print() const {
    std::cout << "Process ID: " << proc_id << std::endl;
    std::cout << "Process count: " << num_processes << std::endl;
    std::cout << "Communicator count: " << num_comms << std::endl;
  }
}
