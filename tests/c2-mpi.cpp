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

TEST_CASE("distribute-tensor", "[mpi][2rank]") {
  using namespace qtnh;
  using namespace std::complex_literals;

  std::vector<qtnh::tel> dt1_els = { 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i, 5.0 + 5.0i, 6.0 + 6.0i, 7.0 + 7.0i, 8.0 + 8.0i };
  std::vector<qtnh::tel> dt2_els = { 5.0 + 5.0i, 6.0 + 6.0i, 7.0 + 7.0i, 8.0 + 8.0i, 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i };

  auto t1u = std::make_unique<SDenseTensor>(ENV, qtnh::tidx_tup { 2, 2, 2 }, dt1_els);
  auto t2u = std::make_unique<SDenseTensor>(ENV, qtnh::tidx_tup { 4, 2 }, dt2_els);

  std::unique_ptr<Tensor> t3u;
  REQUIRE_NOTHROW(t3u = std::unique_ptr<DDenseTensor>(t1u->distribute(1)));

  SECTION("distribute") {
    // ! This should be inside the loop, but for some reason it causes segmentation fault. 
    // REQUIRE_NOTHROW(t3u = std::unique_ptr<DDenseTensor>(t1u->distribute(1)));

    if (ENV.proc_id == 0) {
      REQUIRE(t3u->getLocEl({ 0, 0 }).value() == 1.0 + 1.0i);
    } else if (ENV.proc_id == 1) {
      REQUIRE(t3u->getLocEl({ 0, 0 }).value() == 5.0 + 5.0i);
    }
  }

  std::unique_ptr<Tensor> t4u;

  SECTION("contract") {
    REQUIRE_NOTHROW(t4u = Tensor::contract(std::move(t2u), std::move(t3u), {{ 1, 1 }}));

    // TODO: check elements
  }
}

TEST_CASE("contract-tensor", "[mpi][2rank]") {
  using namespace qtnh;

  // DDenseTensor x SDenseTensor
  for (auto& cv : gen::mpi2r_vals) {
    auto t_sden1_u = std::make_unique<SDenseTensor>(ENV, cv.t1_info.dims, cv.t1_info.els);
    auto t_dden1_u = std::unique_ptr<DDenseTensor>(t_sden1_u->distribute(1));
    auto t_sden2_u = std::make_unique<SDenseTensor>(ENV, cv.t2_info.dims, cv.t2_info.els);
    

    auto t_r1_u = Tensor::contract(std::move(t_dden1_u), std::move(t_sden2_u), cv.wires);

    qtnh::tidx_tup t_r1_dims = cv.t3_info.dims;
    std::vector<qtnh::tel> t_r1_els = cv.t3_info.els;

    REQUIRE(t_r1_u->getDims() == t_r1_dims);
    TIndexing ti_r1(t_r1_dims, 0);

    for (auto idxs : ti_r1) {
      idxs.at(0) = ENV.proc_id;
      auto el = t_r1_els.at(utils::idxs_to_i(idxs, t_r1_dims));

      idxs.erase(idxs.begin());
      REQUIRE(utils::equal(t_r1_u->getLocEl(idxs).value(), el));
    }
  }

  // DDenseTensor x DDenseTensor
  for (auto& cv : gen::mpi2r_vals) {
    auto t_sden1_u = std::make_unique<SDenseTensor>(ENV, cv.t1_info.dims, cv.t1_info.els);
    auto t_dden1_u = std::unique_ptr<DDenseTensor>(t_sden1_u->distribute(1));
    auto t_sden2_u = std::make_unique<SDenseTensor>(ENV, cv.t2_info.dims, cv.t2_info.els);
    auto t_dden2_u = std::unique_ptr<DDenseTensor>(t_sden2_u->distribute(0));
    

    auto t_r1_u = Tensor::contract(std::move(t_dden1_u), std::move(t_dden2_u), cv.wires);

    qtnh::tidx_tup t_r1_dims = cv.t3_info.dims;
    std::vector<qtnh::tel> t_r1_els = cv.t3_info.els;

    REQUIRE(t_r1_u->getDims() == t_r1_dims);
    TIndexing ti_r1(t_r1_dims, 0);

    for (auto idxs : ti_r1) {
      idxs.at(0) = ENV.proc_id;
      auto el = t_r1_els.at(utils::idxs_to_i(idxs, t_r1_dims));

      idxs.erase(idxs.begin());
      REQUIRE(utils::equal(t_r1_u->getLocEl(idxs).value(), el));
    }
  }
}

TEST_CASE("contract-tensor", "[mpi][3rank]") {
  using namespace qtnh;

  // DDenseTensor x SDenseTensor
  for (auto& cv : gen::mpi3r_vals) {
    auto t_sden1_u = std::make_unique<SDenseTensor>(ENV, cv.t1_info.dims, cv.t1_info.els);
    auto t_dden1_u = std::unique_ptr<DDenseTensor>(t_sden1_u->distribute(1));
    auto t_sden2_u = std::make_unique<SDenseTensor>(ENV, cv.t2_info.dims, cv.t2_info.els);
    

    auto t_r1_u = Tensor::contract(std::move(t_dden1_u), std::move(t_sden2_u), cv.wires);

    qtnh::tidx_tup t_r1_dims = cv.t3_info.dims;
    std::vector<qtnh::tel> t_r1_els = cv.t3_info.els;

    REQUIRE(t_r1_u->getDims() == t_r1_dims);
    TIndexing ti_r1(t_r1_dims, 0);

    for (auto idxs : ti_r1) {
      idxs.at(0) = ENV.proc_id;
      auto el = t_r1_els.at(utils::idxs_to_i(idxs, t_r1_dims));

      idxs.erase(idxs.begin());
      REQUIRE(utils::equal(t_r1_u->getLocEl(idxs).value(), el));
    }
  }

  // DDenseTensor x DDenseTensor
  for (auto& cv : gen::mpi3r_vals) {
    auto t_sden1_u = std::make_unique<SDenseTensor>(ENV, cv.t1_info.dims, cv.t1_info.els);
    auto t_dden1_u = std::unique_ptr<DDenseTensor>(t_sden1_u->distribute(1));
    auto t_sden2_u = std::make_unique<SDenseTensor>(ENV, cv.t2_info.dims, cv.t2_info.els);
    auto t_dden2_u = std::unique_ptr<DDenseTensor>(t_sden2_u->distribute(0));
    

    auto t_r1_u = Tensor::contract(std::move(t_dden1_u), std::move(t_dden2_u), cv.wires);

    qtnh::tidx_tup t_r1_dims = cv.t3_info.dims;
    std::vector<qtnh::tel> t_r1_els = cv.t3_info.els;

    REQUIRE(t_r1_u->getDims() == t_r1_dims);
    TIndexing ti_r1(t_r1_dims, 0);

    for (auto idxs : ti_r1) {
      idxs.at(0) = ENV.proc_id;
      auto el = t_r1_els.at(utils::idxs_to_i(idxs, t_r1_dims));

      idxs.erase(idxs.begin());
      REQUIRE(utils::equal(t_r1_u->getLocEl(idxs).value(), el));
    }
  }
}

TEST_CASE("contract-tensor", "[mpi][4rank]") {
  using namespace qtnh;

  // DDenseTensor x DDenseTensor
  for (auto& cv : gen::mpi4r_vals) {
    auto t_sden1_u = std::make_unique<SDenseTensor>(ENV, cv.t1_info.dims, cv.t1_info.els);
    auto t_dden1_u = std::unique_ptr<DDenseTensor>(t_sden1_u->distribute(1));
    auto t_sden2_u = std::make_unique<SDenseTensor>(ENV, cv.t2_info.dims, cv.t2_info.els);
    auto t_dden2_u = std::unique_ptr<DDenseTensor>(t_sden2_u->distribute(1));
    

    auto t_r1_u = Tensor::contract(std::move(t_dden1_u), std::move(t_dden2_u), cv.wires);

    qtnh::tidx_tup t_r1_dims = cv.t3_info.dims;
    std::vector<qtnh::tel> t_r1_els = cv.t3_info.els;

    REQUIRE(t_r1_u->getDims() == t_r1_dims);

    qtnh::tifl_tup ifls(t_r1_dims.size(), { TIdxT::open, 0 });
    ifls.at(0) = ifls.at(1) = { TIdxT::closed, 0 };

    TIndexing ti_r1(t_r1_dims, ifls);

    for (auto idxs : ti_r1) {
      auto dist_idxs = utils::i_to_idxs(ENV.proc_id, { 2, 2 });

      idxs.at(0) = dist_idxs.at(0);
      idxs.at(1) = dist_idxs.at(1);

      auto el = t_r1_els.at(utils::idxs_to_i(idxs, t_r1_dims));

      idxs.erase(idxs.begin(), idxs.begin() + 2);
      REQUIRE(utils::equal(t_r1_u->getLocEl(idxs).value(), el));
    }
  }
}

TEST_CASE("contract-tensor", "[mpi][6rank]") {
  using namespace qtnh;

  // DDenseTensor x DDenseTensor
  for (auto& cv : gen::mpi6r_vals) {
    auto t_sden1_u = std::make_unique<SDenseTensor>(ENV, cv.t1_info.dims, cv.t1_info.els);
    auto t_dden1_u = std::unique_ptr<DDenseTensor>(t_sden1_u->distribute(1));
    auto t_sden2_u = std::make_unique<SDenseTensor>(ENV, cv.t2_info.dims, cv.t2_info.els);
    auto t_dden2_u = std::unique_ptr<DDenseTensor>(t_sden2_u->distribute(1));
    

    auto t_r1_u = Tensor::contract(std::move(t_dden1_u), std::move(t_dden2_u), cv.wires);

    qtnh::tidx_tup t_r1_dims = cv.t3_info.dims;
    std::vector<qtnh::tel> t_r1_els = cv.t3_info.els;

    REQUIRE(t_r1_u->getDims() == t_r1_dims);

    qtnh::tifl_tup ifls(t_r1_dims.size(), { TIdxT::open, 0 });
    ifls.at(0) = ifls.at(1) = { TIdxT::closed, 0 };

    TIndexing ti_r1(t_r1_dims, ifls);

    for (auto idxs : ti_r1) {
      auto dist_idxs = utils::i_to_idxs(ENV.proc_id, { 3, 2 });

      idxs.at(0) = dist_idxs.at(0);
      idxs.at(1) = dist_idxs.at(1);

      auto el = t_r1_els.at(utils::idxs_to_i(idxs, t_r1_dims));

      idxs.erase(idxs.begin(), idxs.begin() + 2);
      REQUIRE(utils::equal(t_r1_u->getLocEl(idxs).value(), el));
    }
  }
}

TEST_CASE("contract-tensor", "[mpi][8rank]") {
  using namespace qtnh;

  // DDenseTensor x DDenseTensor
  for (auto& cv : gen::mpi8r_vals) {
    auto t_sden1_u = std::make_unique<SDenseTensor>(ENV, cv.t1_info.dims, cv.t1_info.els);
    auto t_dden1_u = std::unique_ptr<DDenseTensor>(t_sden1_u->distribute(2));
    auto t_sden2_u = std::make_unique<SDenseTensor>(ENV, cv.t2_info.dims, cv.t2_info.els);
    auto t_dden2_u = std::unique_ptr<DDenseTensor>(t_sden2_u->distribute(1));

    auto t_r1_u = Tensor::contract(std::move(t_dden1_u), std::move(t_dden2_u), cv.wires);

    qtnh::tidx_tup t_r1_dims = cv.t3_info.dims;
    std::vector<qtnh::tel> t_r1_els = cv.t3_info.els;

    REQUIRE(t_r1_u->getDims() == t_r1_dims);

    qtnh::tifl_tup ifls(t_r1_dims.size(), { TIdxT::open, 0 });
    ifls.at(0) = ifls.at(1) = ifls.at(2) = { TIdxT::closed, 0 };

    TIndexing ti_r1(t_r1_dims, ifls);

    for (auto idxs : ti_r1) {
      auto dist_idxs = utils::i_to_idxs(ENV.proc_id, { 2, 2, 2 });

      idxs.at(0) = dist_idxs.at(0);
      idxs.at(1) = dist_idxs.at(1);
      idxs.at(2) = dist_idxs.at(2);

      auto el = t_r1_els.at(utils::idxs_to_i(idxs, t_r1_dims));

      idxs.erase(idxs.begin(), idxs.begin() + 3);
      REQUIRE(utils::equal(t_r1_u->getLocEl(idxs).value(), el));
    }
  }
}

TEST_CASE("collectives", "[mpi][4rank]") {
  using namespace qtnh;
  using namespace std::complex_literals;
  
  std::vector<qtnh::tel> dt1_els = { 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i, 5.0 + 5.0i, 6.0 + 6.0i, 7.0 + 7.0i, 8.0 + 8.0i };
  
  auto t1u = std::make_unique<SDenseTensor>(ENV, qtnh::tidx_tup { 2, 2, 2 }, dt1_els);
  auto t2u = std::unique_ptr<DDenseTensor>(t1u->distribute(1));

  SECTION("scatter") {
    REQUIRE_NOTHROW(t2u->scatter(1));
    
    if (ENV.proc_id == 0) {
      REQUIRE(t2u->getLocEl({ 0 }).value() == 1.0 + 1.0i);
    } else if (ENV.proc_id == 1) {
      REQUIRE(t2u->getLocEl({ 0 }).value() == 3.0 + 3.0i);
    } else if (ENV.proc_id == 2) {
      REQUIRE(t2u->getLocEl({ 0 }).value() == 5.0 + 5.0i);
    } else if (ENV.proc_id == 3) {
      REQUIRE(t2u->getLocEl({ 0 }).value() == 7.0 + 7.0i);
    }
  }
}

TEST_CASE("qft", "[mpi][4rank]") {
  using namespace qtnh;

  SECTION("5-qubits") {
    TensorNetwork tn;
    auto con_ord = gen::qft(ENV, tn, 5, 2);

    auto id = tn.contractAll(con_ord);
    auto tfu = tn.extractTensor(id);

    auto idxs = utils::i_to_idxs(0, tfu->getLocDims());

    if (ENV.proc_id == 0) {
      REQUIRE(utils::equal(tfu->getLocEl(idxs).value(), 1, 1E-4));
    } else {
      REQUIRE(utils::equal(tfu->getLocEl(idxs).value(), 0, 1E-4));
    }
  }

  SECTION("6-qubits") {
    TensorNetwork tn;
    auto con_ord = gen::qft(ENV, tn, 6, 2);

    auto id = tn.contractAll(con_ord);
    auto tfu = tn.extractTensor(id);

    auto idxs = utils::i_to_idxs(0, tfu->getLocDims());

    if (ENV.proc_id == 0) {
      REQUIRE(utils::equal(tfu->getLocEl(idxs).value(), 1, 1E-4));
    } else {
      REQUIRE(utils::equal(tfu->getLocEl(idxs).value(), 0, 1E-4));
    }
  }
}

TEST_CASE("qft", "[mpi][8rank]") {
  using namespace qtnh;

  SECTION("7-qubits") {
    TensorNetwork tn;
    auto con_ord = gen::qft(ENV, tn, 7, 3);

    auto id = tn.contractAll(con_ord);
    auto tfu = tn.extractTensor(id);

    auto idxs = utils::i_to_idxs(0, tfu->getLocDims());

    if (ENV.proc_id == 0) {
      REQUIRE(utils::equal(tfu->getLocEl(idxs).value(), 1, 1E-4));
    } else {
      REQUIRE(utils::equal(tfu->getLocEl(idxs).value(), 0, 1E-4));
    }
  }

  SECTION("8-qubits") {
    TensorNetwork tn;
    auto con_ord = gen::qft(ENV, tn, 8, 3);

    auto id = tn.contractAll(con_ord);
    auto tfu = tn.extractTensor(id);

    auto idxs = utils::i_to_idxs(0, tfu->getLocDims());

    if (ENV.proc_id == 0) {
      REQUIRE(utils::equal(tfu->getLocEl(idxs).value(), 1, 1E-4));
    } else {
      REQUIRE(utils::equal(tfu->getLocEl(idxs).value(), 0, 1E-4));
    }
  }
}

// Swap local/distributed and distributed/distributed