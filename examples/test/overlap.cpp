#include <chrono>
#include <cmath>
#include <iostream>
#include "qtnh.hpp"

#include <adios2.h>
#include <mpi.h>
#include <string>

using namespace qtnh;
using namespace std::chrono;
using namespace std::complex_literals;

BCMPS load_mps(const QTNHEnv& env, std::string filename) {
  utils::barrier();
  auto start = high_resolution_clock::now();

  adios2::ADIOS adios(MPI_COMM_WORLD);
  auto bpIO = adios.DeclareIO("BPReader");
  auto bpReader = bpIO.Open(filename, adios2::Mode::Read);

  std::vector<tptr> sites;
  std::vector<std::size_t> info;
  std::vector<std::size_t> dims;

  bpReader.BeginStep();
  auto bpMPS = bpIO.InquireVariable<std::size_t>("bpMPS");
  bpMPS.SetSelection({{ 0 }, { 4 }});
  bpReader.Get(bpMPS, info, adios2::Mode::Sync);
  bpReader.EndStep();

  bpReader.BeginStep();
  auto bpDims = bpIO.InquireVariable<std::size_t>("bpDims");
  bpDims.SetSelection({{ 0 }, { info.at(0) }});
  bpReader.Get(bpDims, dims, adios2::Mode::Sync);
  bpReader.EndStep();

  for (auto i = 0UL; i < info.at(0); ++i) {
    auto loc_size = dims.at(i) * info.at(1) * info.at(1) * info.at(3) * info.at(3);
    std::vector<tel> els;

    bpReader.BeginStep();
    auto bpSite = bpIO.InquireVariable<tel>("bpSite" + std::to_string(i));
    bpSite.SetSelection({{ env.proc_id * loc_size }, { loc_size }});
    bpReader.Get(bpSite, els, adios2::Mode::Sync);
    bpReader.EndStep();

    tidx_tup dis_dims { info.at(2), info.at(2) };
    tidx_tup loc_dims { dims.at(i), info.at(1), info.at(3), info.at(1), info.at(3) };
    tptr tp_site = DenseTensor::make(env, dis_dims, loc_dims, std::move(els));
    sites.push_back(std::move(tp_site));
  }

  bpReader.Close();

  utils::barrier();
  auto stop = high_resolution_clock::now();
  auto delta = duration_cast<milliseconds>(stop - start);

  if (utils::is_root()) {
    std::cout << "Read file " << filename << " (" << delta.count() << " ms)" << std::endl;
  }
  
  return BCMPS(std::move(sites));
}

const PTupleTar ptup_ab_cd({
               16, 
            9, 17, 25, 
        4, 10, 18, 26, 34, 
     1, 5, 11, 19, 27, 35, 
  0, 2, 6, 12, 20, 28, 36, 42, 47, 
     3, 7, 13, 21, 29, 37, 43, 48, 51, 
        8, 14, 22, 30, 38, 44, 49, 52, 
           15, 23, 31, 39, 45, 50, 
               24, 32, 40, 46, 
                   33, 41
});

void permute(BCMPS& mps, PTupleTar ptup, bool update_dims) {
  auto tup = ptup.tup();
  auto start = high_resolution_clock::now();

  auto swapped_count = 0UL;
  auto print_frequency = 20UL;

  for (auto i = 0UL; i < tup.size() - 1; ++i) {
    if (tup.at(i) > tup.at(i + 1)) {
      ++swapped_count;
      
      mps.leftCanonicalise(i + 1);
      mps.rightCanonicalise(i);

      mps.swap(i, update_dims);
      std::swap(tup.at(i), tup.at(i + 1));
      
      if (swapped_count % 20 == 0 && utils::is_root()) {
        auto stop = high_resolution_clock::now();
        auto delta = duration_cast<milliseconds>(stop - start);

        std::cout << "Swapped " << print_frequency << "(" << swapped_count << 
          ") sites (" << delta.count() << " ms)" << std::endl;

        start = high_resolution_clock::now();
      }

      i = 0UL;
    }
  }
}

int main(int, char* argv[]) {
  QTNHEnv env;

  std::string FILENAME_IN_1 = argv[1];
  std::string FILENAME_IN_2 = argv[2];
  
  auto mps1 = load_mps(env, FILENAME_IN_1);
  auto mps2 = load_mps(env, FILENAME_IN_2);

  permute(mps2, ptup_ab_cd, false);

  auto norm1 = mps1.norm();
  auto norm2 = mps2.norm();

  mps1.renormalise();
  mps2.renormalise();

  auto overlap = mps1.overlap(mps2);
  auto sim = std::sqrt(std::abs(overlap));
  if (utils::is_root()) {
    std::cout << "|MPS1| = " << norm1 << std::endl;
    std::cout << "|MPS2| = " << norm2 << std::endl;
    std::cout << "Overlap(MPS1, MPS2) = " << overlap << std::endl;
    std::cout << "Sim(MPS1, MPS2) = " << sim << std::endl;
  }
}
