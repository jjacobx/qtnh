#include <iostream>
#include <vector>
#include <mpi.h>

int main() {
  MPI_Init(0, 0);

  std::vector<int> send { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
  std::vector<int> recv(16, 0);

  MPI_Datatype send_type, recv_type;
  MPI_Type_contiguous(16, MPI_INT, &send_type);

  std::vector<MPI_Datatype> temp_types(5);
  MPI_Type_create_resized(MPI_INT, 0, 2 * sizeof(int), &temp_types.at(0));
  MPI_Type_contiguous(2, temp_types.at(0), &temp_types.at(1));
  MPI_Type_create_resized(temp_types.at(1), 0, sizeof(int), &temp_types.at(2));
  MPI_Type_contiguous(2, temp_types.at(2), &temp_types.at(3));
  MPI_Type_create_resized(temp_types.at(3), 0, 4 * sizeof(int), &temp_types.at(4));
  MPI_Type_contiguous(4, temp_types.at(4), &recv_type);

  MPI_Type_commit(&send_type);
  MPI_Type_commit(&recv_type);

  MPI_Sendrecv(send.data(), 1, send_type, 0, 0, 
               recv.data(), 1, recv_type, 0, 0, 
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
  // Expected result: { 1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12, 13, 15, 14, 16 }
  for (auto e : recv) std::cout << e << ", ";
  std::cout << "\n";

  MPI_Finalize();
}
