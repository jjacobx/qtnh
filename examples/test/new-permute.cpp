#include <complex>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <vector>

const auto SIZE = 16UL;

int main() {
  MPI_Init(0, 0);

  std::vector<std::size_t> dims(SIZE, 2);
  std::vector<std::size_t> cmldims(SIZE, 1);

  // Compute cumulative dimensions
  for (auto i = 1UL; i < SIZE; ++i) {
    cmldims.at(SIZE - i - 1) = dims.at(SIZE - i) * cmldims.at(SIZE - i);
  }

  // Permutation tuple to rotate tensor by 2
  std::vector<std::size_t> ptup(SIZE);
  std::iota(ptup.begin(), ptup.end(), 2);
  ptup.at(SIZE - 2) = 0;
  ptup.at(SIZE - 1) = 1;

  // Vectors with intermediate datatypes
  std::vector<MPI_Datatype> send_types(2 * SIZE + 1, MPI_C_DOUBLE_COMPLEX);
  std::vector<MPI_Datatype> recv_types(2 * SIZE + 1, MPI_C_DOUBLE_COMPLEX);
  auto cdbl_size = sizeof(std::complex<double>);

  auto i1 = 0UL, i2 = 0UL;
  auto sc = 1UL, rc = 1UL;
  for (auto k = 0UL; k < SIZE; ++k) {
    auto i = SIZE - k - 1;
    auto j = ptup.at(i);

    sc *= dims.at(i);
    MPI_Aint lb, ext;
    MPI_Type_get_extent(send_types.at(i1), &lb, &ext);
    std::cout << "Send extent: " << ext << ", expected: " << cmldims.at(i) * cdbl_size << "\n";
    if (sc * ext != cmldims.at(i) * cdbl_size || i == 0) {
      std::cout << "Resizing: \n";
      MPI_Type_contiguous(int(sc), send_types.at(i1), &send_types.at(i1 + 1));
      MPI_Type_create_resized(send_types.at(i1 + 1), 0, cmldims.at(i) * cdbl_size, &send_types.at(i1 + 2));
      sc = 1UL;
      i1 += 2;
    }

    rc *= dims.at(j);
    MPI_Type_get_extent(recv_types.at(i2), &lb, &ext);
    std::cout << "Recv extent: " << ext << ", expected: " << cmldims.at(j) * cdbl_size << "\n";
    if (rc * ext != cmldims.at(j) * cdbl_size || i == 0) {
      std::cout << "Resizing: \n";
      MPI_Type_contiguous(int(rc), recv_types.at(i2), &recv_types.at(i2 + 1));
      MPI_Type_create_resized(recv_types.at(i2 + 1), 0, cmldims.at(j) * cdbl_size, &recv_types.at(i2 + 2));
      rc = 1UL;
      i2 += 2;
    }
  }

  MPI_Datatype send_type, recv_type;
  MPI_Type_create_resized(send_types.at(i1), 0, cdbl_size, &send_type);
  MPI_Type_create_resized(recv_types.at(i2), 0, cdbl_size, &recv_type);

  MPI_Type_commit(&send_type);
  MPI_Type_commit(&recv_type);

  // Free intermediate datatypes
  for (auto i = 1UL; i <= 2 * SIZE; ++i) {
      if (send_types.at(i) != MPI_C_DOUBLE_COMPLEX) MPI_Type_free(&send_types.at(i));
      if (recv_types.at(i) != MPI_C_DOUBLE_COMPLEX) MPI_Type_free(&recv_types.at(i));
  }

  std::vector<std::complex<double>> send_tensor(1 << SIZE);
  std::vector<std::complex<double>> recv_tensor(1 << SIZE);

  // Initialise send tensor
  std::iota(send_tensor.begin(), send_tensor.end(), 0);

  MPI_Request request;
  MPI_Isend(send_tensor.data(), 1, send_type, 0, 0, MPI_COMM_WORLD, &request);
  MPI_Recv(recv_tensor.data(), 1, recv_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Wait(&request, MPI_STATUS_IGNORE);
  MPI_Type_free(&send_type);
  MPI_Type_free(&recv_type);
  
  // Check tensor elements
  std::cout << send_tensor.at(0) << "\n";
  std::cout << send_tensor.at(2) << "\n";
  std::cout << recv_tensor.at(0) << "\n";
  std::cout << recv_tensor.at(2) << "\n";

  MPI_Finalize();

  return 0;
}
