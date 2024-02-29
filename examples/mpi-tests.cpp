#include <iostream>
#include <mpi.h>

#include "core/env.hpp"

int main() {
  qtnh::QTNHEnv env;

  MPI_Datatype type, type_resized;
  MPI_Type_vector(4, 4, 16, MPI_INT, &type);
  MPI_Type_create_resized(type, 0, 4 * sizeof(int), &type_resized);
  MPI_Type_commit(&type_resized);

  int send_buffer[64];
  int recv_buffer[64];

  for (int i = 0; i < 64; ++i) {
    send_buffer[i] = 2 * env.proc_id + (i % 8) / 4 + 1;
    recv_buffer[i] = 0;
  }

  for (int i = 0; i < 4; ++i) {
    MPI_Scatter(&send_buffer[0], 1, type_resized, &recv_buffer[i * 4], 1, type_resized, i, MPI_COMM_WORLD);
  }
  

  std::cout << "P" << env.proc_id << ": ";
  for (int i = 0; i < 64; ++i) {
    std::cout << recv_buffer[i];
    if (i < 63) std::cout << ", ";
    else std::cout << std::endl;
  }

  return 0;
}