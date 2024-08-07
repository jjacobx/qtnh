#include <catch2/catch_session.hpp>
#include <catch2/catch_test_case_info.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <catch2/reporters/catch_reporter_streaming_base.hpp>

#include <iostream>
#include "qtnh.hpp"

#include "gen/random-tensors.hpp"
#include "gen/qft.hpp"

qtnh::QTNHEnv ENV;

class RootReporter : public Catch::StreamingReporterBase {
  public:
    using StreamingReporterBase::StreamingReporterBase;

    static std::string getDescription() {
        return "Reporter for testing events on multiple MPI ranks. ";
    }

    void testCaseStarting(Catch::TestCaseInfo const& testInfo) override {
      if (ENV.proc_id == 0) {
        std::cout << "Starting test case: " << testInfo.name << '\n';
      }
    }

    void testCaseEnded(Catch::TestCaseStats const& testCaseStats) override {
      MPI_Barrier(MPI_COMM_WORLD);
      if (ENV.proc_id == 0) {
        std::cout << "Test case ended: " << testCaseStats.testInfo->name << "\n";
        if (testCaseStats.totals.assertions.failed == 0 && testCaseStats.totals.testCases.failed == 0) {
          std::cout << "Passed\n";
        } else {
          std::cout << "Failed\n";
        }
      }
    }

  private:
    static unsigned int counter;
};

unsigned int RootReporter::counter = 0;

CATCH_REGISTER_REPORTER("root", RootReporter)

TEST_CASE("scatter-tensor", "[mpi][2rank]") {
  using namespace qtnh;
  using namespace std::complex_literals;

  std::vector<tel> dt1_els = { 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i, 5.0 + 5.0i, 6.0 + 6.0i, 7.0 + 7.0i, 8.0 + 8.0i };
  std::vector<tel> dt2_els = { 5.0 + 5.0i, 6.0 + 6.0i, 7.0 + 7.0i, 8.0 + 8.0i, 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i };

  tptr tp1 = DenseTensor::make(ENV, {}, { 2, 2, 2 }, std::move(dt1_els));
  tptr tp2 = DenseTensor::make(ENV, {}, { 4, 2 }, std::move(dt2_els));
  tptr tp3, tp4;
  
  REQUIRE_NOTHROW(tp3 = Tensor::rescatter(std::move(tp1), 1));

  SECTION("distribute") {
    // ! This should be inside the section, but for some reason it causes segmentation fault. 
    // REQUIRE_NOTHROW(t3u = std::unique_ptr<DDenseTensor>(t1u->distribute(1)));

    if (ENV.proc_id == 0) {
      REQUIRE(tp3->at({ 0, 0, 0 }) == 1.0 + 1.0i);
    } else if (ENV.proc_id == 1) {
      REQUIRE(tp3->at({ 1, 0, 0 }) == 5.0 + 5.0i);
    }
  }

  SECTION("contract") {
    REQUIRE_NOTHROW(tp4 = Tensor::contract(std::move(tp2), std::move(tp3), {{ 1, 1 }}));

    // TODO: check elements
  }
}

TEST_CASE("contract-tensor", "[mpi][2rank]") {
  using namespace qtnh;

  for (auto& cv : gen::mpi2r_vals) {
    tptr tp1 = DenseTensor::make(ENV, {}, cv.t1_info.dims, std::vector<tel>(cv.t1_info.els));
    tptr tp2 = DenseTensor::make(ENV, {}, cv.t2_info.dims, std::vector<tel>(cv.t2_info.els));
    
    tp1 = Tensor::rescatter(std::move(tp1), 1);

    tptr tp3 = Tensor::contract(std::move(tp1), std::move(tp2), cv.wires);

    auto dims = cv.t3_info.dims;
    auto els = cv.t3_info.els;

    REQUIRE(tp3->totDims() == dims);
    TIndexing ti(tp3->locDims());

    for (auto idxs : ti.tup()) {
      idxs.insert(idxs.begin(), ENV.proc_id);

      auto el = els.at(utils::idxs_to_i(idxs, dims));
      REQUIRE(utils::equal(tp3->at(idxs), el));
    }
  }
}

TEST_CASE("contract-tensor", "[mpi][3rank]") {
  using namespace qtnh;

  for (auto& cv : gen::mpi3r_vals) {
    tptr tp1 = DenseTensor::make(ENV, {}, cv.t1_info.dims, std::vector<tel>(cv.t1_info.els));
    tptr tp2 = DenseTensor::make(ENV, {}, cv.t2_info.dims, std::vector<tel>(cv.t2_info.els));
    
    tp1 = Tensor::rescatter(std::move(tp1), 1);

    tptr tp3 = Tensor::contract(std::move(tp1), std::move(tp2), cv.wires);

    auto dims = cv.t3_info.dims;
    auto els = cv.t3_info.els;

    REQUIRE(tp3->totDims() == dims);
    TIndexing ti(tp3->locDims());

    for (auto idxs : ti.tup()) {
      idxs.insert(idxs.begin(), ENV.proc_id);

      auto el = els.at(utils::idxs_to_i(idxs, dims));
      REQUIRE(utils::equal(tp3->at(idxs), el));
    }
  }
}

TEST_CASE("contract-tensor", "[mpi][4rank]") {
  using namespace qtnh;

  for (auto& cv : gen::mpi4r_vals) {
    tptr tp1 = DenseTensor::make(ENV, {}, cv.t1_info.dims, std::vector<tel>(cv.t1_info.els));
    tptr tp2 = DenseTensor::make(ENV, {}, cv.t2_info.dims, std::vector<tel>(cv.t2_info.els));
    
    tp1 = Tensor::rescatter(std::move(tp1), 1);
    tp2 = Tensor::rescatter(std::move(tp2), 1);

    tptr tp3 = Tensor::contract(std::move(tp1), std::move(tp2), cv.wires);

    auto dims = cv.t3_info.dims;
    auto els = cv.t3_info.els;

    REQUIRE(tp3->totDims() == dims);
    TIndexing ti(tp3->locDims());

    for (auto idxs : ti.tup()) {
      auto dist_idxs = utils::i_to_idxs(ENV.proc_id, { 2, 2 });
      idxs.insert(idxs.begin(), dist_idxs.begin(), dist_idxs.end());

      auto el = els.at(utils::idxs_to_i(idxs, dims));
      REQUIRE(utils::equal(tp3->at(idxs), el));
    }
  }
}

TEST_CASE("contract-tensor", "[mpi][6rank]") {
  using namespace qtnh;

  for (auto& cv : gen::mpi6r_vals) {
    tptr tp1 = DenseTensor::make(ENV, {}, cv.t1_info.dims, std::vector<tel>(cv.t1_info.els));
    tptr tp2 = DenseTensor::make(ENV, {}, cv.t2_info.dims, std::vector<tel>(cv.t2_info.els));
    
    tp1 = Tensor::rescatter(std::move(tp1), 1);
    tp2 = Tensor::rescatter(std::move(tp2), 1);

    tptr tp3 = Tensor::contract(std::move(tp1), std::move(tp2), cv.wires);

    auto dims = cv.t3_info.dims;
    auto els = cv.t3_info.els;

    REQUIRE(tp3->totDims() == dims);
    TIndexing ti(tp3->locDims());

    for (auto idxs : ti.tup()) {
      auto dist_idxs = utils::i_to_idxs(ENV.proc_id, { 3, 2 });
      idxs.insert(idxs.begin(), dist_idxs.begin(), dist_idxs.end());

      auto el = els.at(utils::idxs_to_i(idxs, dims));
      REQUIRE(utils::equal(tp3->at(idxs), el));
    }
  }
}

TEST_CASE("contract-tensor", "[mpi][8rank]") {
  using namespace qtnh;

  for (auto& cv : gen::mpi8r_vals) {
        tptr tp1 = DenseTensor::make(ENV, {}, cv.t1_info.dims, std::vector<tel>(cv.t1_info.els));
    tptr tp2 = DenseTensor::make(ENV, {}, cv.t2_info.dims, std::vector<tel>(cv.t2_info.els));
    
    tp1 = Tensor::rescatter(std::move(tp1), 2);
    tp2 = Tensor::rescatter(std::move(tp2), 1);

    tptr tp3 = Tensor::contract(std::move(tp1), std::move(tp2), cv.wires);

    auto dims = cv.t3_info.dims;
    auto els = cv.t3_info.els;

    REQUIRE(tp3->totDims() == dims);
    TIndexing ti(tp3->locDims());

    for (auto idxs : ti.tup()) {
      auto dist_idxs = utils::i_to_idxs(ENV.proc_id, { 2, 2, 2 });
      idxs.insert(idxs.begin(), dist_idxs.begin(), dist_idxs.end());

      auto el = els.at(utils::idxs_to_i(idxs, dims));
      REQUIRE(utils::equal(tp3->at(idxs), el));
    }
  }
}

TEST_CASE("collectives", "[mpi][4rank]") {
  using namespace qtnh;
  using namespace std::complex_literals;
  
  std::vector<tel> els = { 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i, 5.0 + 5.0i, 6.0 + 6.0i, 7.0 + 7.0i, 8.0 + 8.0i };
  
  tptr tp = DenseTensor::make(ENV, {}, { 2, 2, 2 }, std::move(els));
  tp = Tensor::rescatter(std::move(tp), 1);

  SECTION("scatter") {
    REQUIRE_NOTHROW(tp = Tensor::rescatter(std::move(tp), 1));
    
    if (ENV.proc_id == 0) {
      REQUIRE(tp->at({ 0, 0, 0 }) == 1.0 + 1.0i);
    } else if (ENV.proc_id == 1) {
      REQUIRE(tp->at({ 0, 1, 0 }) == 3.0 + 3.0i);
    } else if (ENV.proc_id == 2) {
      REQUIRE(tp->at({ 1, 0, 0 }) == 5.0 + 5.0i);
    } else if (ENV.proc_id == 3) {
      REQUIRE(tp->at({ 1, 1, 0 }) == 7.0 + 7.0i);
    }
  }
}

// TODO: Re-implement QFT. 
TEST_CASE("qft", "[qft][mpi][16rank]") {
  using namespace qtnh;

  SECTION("5-qubits") {
    TensorNetwork tn;
    auto con_ord = gen::qft(ENV, tn, 5, 2);

    auto id = tn.contractAll(con_ord);
    auto tp = tn.extract(id);

    auto idxs = utils::i_to_idxs(0, tp->totDims());

    if (ENV.proc_id == 0) {
      REQUIRE(utils::equal(tp->at(idxs), 1, 1E-4));
    }
  }

  SECTION("6-qubits") {
    TensorNetwork tn;
    auto con_ord = gen::qft(ENV, tn, 6, 2);

    auto id = tn.contractAll(con_ord);
    auto tp = tn.extract(id);

    auto idxs = utils::i_to_idxs(0, tp->totDims());

    if (ENV.proc_id == 0) {
      REQUIRE(utils::equal(tp->at(idxs), 1, 1E-4));
    }
  }
}

TEST_CASE("qft", "[mpi][32rank]") {
  using namespace qtnh;

  SECTION("7-qubits") {
    TensorNetwork tn;
    auto con_ord = gen::qft(ENV, tn, 5, 1);

    auto id = tn.contractAll(con_ord);
    auto tp = tn.extract(id);

    auto idxs = utils::i_to_idxs(0, tp->totDims());

    if (ENV.proc_id == 0) {
      REQUIRE(utils::equal(tp->at(idxs), 1, 1E-4));
    }
  }

  // SECTION("8-qubits") {
  //   TensorNetwork tn;
  //   auto con_ord = gen::qft(ENV, tn, 8, 3);

  //   auto id = tn.contractAll(con_ord);
  //   auto tp = tn.extract(id);

  //   auto idxs = utils::i_to_idxs(0, tp->totDims());

  //   if (ENV.proc_id == 0) {
  //     REQUIRE(utils::equal(tp->at(idxs), 1, 1E-4));
  //   }
  // }
}

// Swap local/distributed and distributed/distributed
