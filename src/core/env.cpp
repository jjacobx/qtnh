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

    init_swap_p2_types();

    #ifdef DEBUG
      ROOT_COUT << "DEF_STENSOR_BCAST is " << (DEF_STENSOR_BCAST ? "on" : "off") << std::endl;
    #endif
  }

  QTNHEnv::~QTNHEnv() {
    free_swap_p2_types();
    MPI_Finalize();
  }

  void QTNHEnv::print() const {
    std::cout << "Process ID: " << proc_id << std::endl;
    std::cout << "Process count: " << num_processes << std::endl;
    std::cout << "Thread count: " << num_threads << std::endl;
  }

  void QTNHEnv::init_swap_p2_types() {
    for (std::size_t i = 0; i < 32; ++i) {
      auto block_length = (std::size_t)pow(2, i);
      auto stride = 2 * block_length;

      // Safeguard as type_vector only accepts int arguments
      // TODO: use chained datatypes to overcome that limit
      if (stride <= std::numeric_limits<int>::max()) {
        MPI_Datatype non_resized;
        MPI_Type_vector(1, block_length, stride, MPI_C_DOUBLE_COMPLEX, &non_resized);
        MPI_Type_create_resized(non_resized, 0, block_length * sizeof(qtnh::tel), &swap_p2_types[i]);
      }

      swap_p2_types_committed[i] = false;
    }
  }

  void QTNHEnv::free_swap_p2_types() {
    for (std::size_t i = 0; i < 32; ++i) {
      if (swap_p2_types_committed[i]) {
        MPI_Type_free(&swap_p2_types[i]);
        swap_p2_types_committed[i] = false;
      }
    }
  }
}