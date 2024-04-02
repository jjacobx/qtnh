#include <catch2/catch_session.hpp>
#include <catch2/catch_test_case_info.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <catch2/reporters/catch_reporter_streaming_base.hpp>

#include <iostream>
#include "qtnh.hpp"

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

TEST_CASE("test-mpi", "[mpi][2rank]") {
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

TEST_CASE("new-tensor", "[mpi][4rank]") {
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
    } else if (ENV.proc_id == 1) {
      REQUIRE(t2u->getLocEl({ 0 }).value() == 5.0 + 5.0i);
    } else if (ENV.proc_id == 1) {
      REQUIRE(t2u->getLocEl({ 0 }).value() == 7.0 + 7.0i);
    }
  }
}
