#include <iostream>
#include <mpi.h>
#include <omp.h>

#include "env.hpp"
#include "typedefs.hpp"

namespace qtnh {
  QTNHEnv::QTNHEnv() {
    num_threads = omp_get_max_threads();

    int _proc_id, _num_processes;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &_proc_id);
    MPI_Comm_size(MPI_COMM_WORLD, &_num_processes);

    proc_id = _proc_id;
    num_processes = _num_processes;

    init_swap_types();

    #ifdef DEBUG
      ROOT_COUT << "DEF_STENSOR_BCAST is " << (DEF_STENSOR_BCAST ? "on" : "off") << std::endl;
    #endif
  }

  QTNHEnv::~QTNHEnv() {
    free_swap_types();
    MPI_Finalize();
  }

  void QTNHEnv::print() const {
    std::cout << "Process ID: " << proc_id << std::endl;
    std::cout << "Process count: " << num_processes << std::endl;
    std::cout << "Thread count: " << num_threads << std::endl;
  }

  void QTNHEnv::init_swap_types() {
    for (std::size_t i = 0; i < 4; ++i) {
      for (std::size_t j = 0; j < 32; ++j) {
        auto b = i + 2;
        auto block_length = (long long)pow(b, j);
        auto stride = b * block_length;
        if (stride <= std::numeric_limits<int>::max()) {
          MPI_Type_vector(1, block_length, stride, MPI_C_DOUBLE_COMPLEX, &swap_types[i][j]);
        }
        swap_types_committed[i][j] = false;
      }
    }
  }

  void QTNHEnv::free_swap_types() {
    for (std::size_t i = 0; i < 4; ++i) {
      for (std::size_t j = 0; j < 32; ++j) {
        if (swap_types_committed[i][j]) {
          MPI_Type_free(&swap_types[i][j]);
          swap_types_committed[i][j] = false;
        }
      }
    }
  }
}