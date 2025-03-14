#include <iostream>
#include <mpi.h>

#include "env.hpp"

const auto SIZE = 20UL;

int main() {
  qtnh::QTNHEnv env;

  MPI_Datatype type, type_resized;
  MPI_Type_vector(4, 1, 4, MPI_DOUBLE, &type);
  MPI_Type_create_resized(type, 0, 1 * sizeof(double), &type_resized);
  MPI_Type_commit(&type_resized);

  MPI_Aint lb1, lb2, ext1, ext2;
  MPI_Type_get_extent(type_resized, &lb1, &ext1);
  MPI_Type_get_true_extent(type_resized, &lb2, &ext2);


  if (!env.proc_id) std::cout << lb1 << ", " << lb2 << ", " << ext1 << ", " << ext2 << "\n";
  MPI_Barrier(MPI_COMM_WORLD);

  double send_buffer[SIZE];
  double recv_buffer[SIZE];

  for (auto i = 0UL; i < SIZE; ++i) {
    send_buffer[i] = static_cast<double>(i);
    recv_buffer[i] = 0;
  }

  if (env.proc_id == 0) {
    std::cout << "P" << env.proc_id << ": ";
    for (int i = 0; i < 16; ++i) {
      std::cout << send_buffer[i];
      if (i < 15) std::cout << ", ";
      else std::cout << std::endl;
    }
  }
  
  MPI_Scatter(&send_buffer[0], 2, type_resized, &recv_buffer[0], 8, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::cout << "P" << env.proc_id << ": ";
  for (int i = 0; i < 8; ++i) {
    std::cout << recv_buffer[i];
    if (i < 7) std::cout << ", ";
    else std::cout << std::endl;
  }

  return 0;
}