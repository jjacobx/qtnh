#include <iostream>
#include "qtnh.hpp"

using namespace qtnh;
using namespace qtnh::ops;
using namespace std::complex_literals;


int main() {
  QTNHEnv env;

  qtnh::tidx_tup t1_dis_dims = { 2 }, t1_loc_dims = { 2, 2, 2 };
  std::vector<qtnh::tel> t1_els;
  if (env.proc_id == 0) {
    t1_els = { 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i, 5.0 + 5.0i, 6.0 + 6.0i, 7.0 + 7.0i, 8.0 + 8.0i };
  } else if (env.proc_id == 1) {
    t1_els = { 5.0 - 5.0i, 6.0 - 6.0i, 7.0 - 7.0i, 8.0 - 8.0i, 1.0 - 1.0i, 2.0 - 2.0i, 3.0 - 3.0i, 4.0 - 4.0i };
  }

  tptr t1u = DenseTensor::make(env, t1_dis_dims, t1_loc_dims, std::move(t1_els));
  std::cout << env.proc_id << " | T1 = " << *t1u << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  t1u = Tensor::rebcast(std::move(t1u), { 2, 2, 0 });
  std::cout << env.proc_id << " | T1 (re-bcast 1) = " << *t1u << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  t1u = Tensor::rescatter(std::move(t1u), -1);
  std::cout << env.proc_id << " | T1 (re-scatter 1) = " << *t1u << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  t1u = Tensor::rebcast(std::move(t1u), { 1, 1, 0 });
  std::cout << env.proc_id << " | T1 (re-bcast 2) = " << *t1u << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  t1u = Tensor::permute(std::move(t1u), { 0, 1, 2, 3 });
  std::cout << env.proc_id << " | T1 (permute 1) = " << *t1u << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  t1u = Tensor::rescatter(std::move(t1u), 2);
  std::cout << env.proc_id << " | T1 (re-scatter 2) = " << *t1u << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  t1u = Tensor::permute(std::move(t1u), { 1, 0, 2, 3 });
  std::cout << env.proc_id << " | T1 (permute 2) = " << *t1u << std::endl;
  auto t1u1 = t1u->copy();
  auto t1u2 = Tensor::cast<DenseTensor>(std::move(t1u1));
  auto t1u3 = t1u->copy();
  auto t1u4 = DenseTensorBase::toDense(Tensor::cast<DenseTensor>(std::move(t1u3)));


  qtnh::tidx_tup t2_dis_dims = { }, t2_loc_dims = { 2, 2 };
  std::vector<qtnh::tel> t2_els = { 1.0, 0.0, 0.0, -1.0 };
  qtnh::tptr t2u = DenseTensor::make(env, t2_dis_dims, t2_loc_dims, std::move(t2_els));
  std::cout << env.proc_id << " | T2 = " << *t2u << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  auto t3u = Tensor::contract(std::move(t1u), std::move(t2u), {{ 3, 0 }});
  std::cout << env.proc_id << " | T3 (contract 1) = " << *t3u << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  t1u = t1u2->copy();
  t2u = std::move(t3u);
  t3u = Tensor::contract(std::move(t1u), std::move(t2u), {{ 1, 1 }, { 2, 2 }});
  std::cout << env.proc_id << " | T3 (contract 2) = " << *t3u << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  t3u = Tensor::rebcast(std::move(t3u), { 1, 1, 4 });
  std::cout << env.proc_id << " | T3 (re-bcast 1) = " << *t3u << std::endl;

  qtnh::tidx_tup t4_dis_dims = { 2 }, t4_loc_dims = { 2, 3 };
  std::vector<qtnh::tel> t4_els;
  if (env.proc_id == 0) {
    t4_els = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
  } else {
    t4_els = { 6.0, 7.0, 8.0, 9.0, 10.0, 11.0 };
  }

  qtnh::tptr t4u = DenseTensor::make(env, t4_dis_dims, t4_loc_dims, std::move(t4_els));
  std::cout << env.proc_id << " | T4 = " << *t4u << std::endl;
  t4u = Tensor::rebcast(std::move(t4u), { 1, 1, 1 });


  // qtnh::tidx_tup t1_dims = { 2, 2, 2 };
  // qtnh::tidx_tup t2_dims = { 4, 2 };

  // std::vector<qtnh::tel> t1_els = { 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i, 5.0 + 5.0i, 6.0 + 6.0i, 7.0 + 7.0i, 8.0 + 8.0i };
  // std::vector<qtnh::tel> t2_els = { 5.0 + 5.0i, 6.0 + 6.0i, 7.0 + 7.0i, 8.0 + 8.0i, 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i };

  // auto t1u = std::make_unique<SDenseTensor>(env, t1_dims, t1_els);
  // auto t2u = std::make_unique<SDenseTensor>(env, t2_dims, t2_els);
  // auto t3u = std::unique_ptr<DDenseTensor>(t1u->distribute(1));

  // std::cout << env.proc_id << " | T3 = " << *t3u << "\n";

  // qtnh::TensorNetwork tn;
  // auto t2_id = tn.insertTensor(std::move(t2u));
  // auto t3_id = tn.insertTensor(std::move(t3u));
  // auto b1_id = tn.createBond(t2_id, t3_id, {{ 1, 2 }});

  // auto t4_id = tn.contractBond(b1_id);
  // auto& t4 = tn.getTensor(t4_id);
  // std::cout << env.proc_id << " | T4 = " << t4 << "\n";

  // t4.swap(0, 2);
  // std::cout << env.proc_id << " | T4' = " << t4 << "\n";

  // qtnh::tidx_tup t5_dims = { 3, 3 };
  // std::vector<qtnh::tel> t5_els = { 1, 2, 3, 1, 2, 3, 1, 2, 3 };
  // auto t5u = std::make_unique<SDenseTensor>(env, t5_dims, t5_els);

  // auto t6u = std::unique_ptr<DDenseTensor>(t5u->distribute(1));
  // std::cout << env.proc_id << " | T6 = " << *t6u << "\n";

  // t6u->swap(0, 1);
  // std::cout << env.proc_id << " | T6' = " << *t6u << "\n";

  // qtnh::tidx_tup t7_dims = { 2, 2, 2, 2 };
  // std::vector<qtnh::tel> t7_els = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
  // auto t7u = std::make_unique<SDenseTensor>(env, t7_dims, t7_els);

  // auto t8u = std::unique_ptr<DDenseTensor>(t7u->distribute(2));
  // std::cout << env.proc_id << " | T8 = " << *t8u << "\n";

  // t8u->swap(0, 1);
  // std::cout << env.proc_id << " | T8' = " << *t8u << "\n";

  // t8u->swap(1, 0);
  // t8u->swap(2, 3);
  // std::cout << env.proc_id << " | T8'' = " << *t8u << "\n";

  // t8u->swap(3, 2);
  // std::cout << env.proc_id << " | T8 = " << *t8u << "\n";

  // auto stu = std::make_unique<SwapTensor>(env, 2, 2);
  // std::cout << env.proc_id << " | SWAP = " << *stu << "\n";

  // auto t9u = Tensor::contract(std::move(t8u), std::move(stu), {{ 2, 0 }, { 3, 1 }});
  // std::cout << env.proc_id << " | T9 = " << *t9u << "\n";
  // if (env.proc_id == 0) {
  //   std::cout << "T9.dims = " << t9u->getDims() << "\n";
  // }

  // auto idu = std::make_unique<IdentityTensor>(env, qtnh::tidx_tup{ 2, 2 });
  // std::cout << env.proc_id << " | ID = " << *idu << "\n";
  
  // auto t10u = Tensor::contract(std::move(t9u), std::move(idu), {{ 3, 0 }, {2, 1}});
  // std::cout << env.proc_id << " | T10 = " << *t10u << "\n";

  // auto cvu = std::make_unique<ConvertTensor>(env, qtnh::tidx_tup{ 2, 2 });
  // std::cout << env.proc_id << " | CV = " << *cvu << "\n";

  // auto t11u = Tensor::contract(std::move(t10u), std::move(cvu), {{1, 1}, {0, 0}});
  // std::cout << env.proc_id << " | T11 = " << *t11u << "\n";

  // auto shu = std::make_unique<ConvertTensor>(env, qtnh::tidx_tup{});
  // auto t12u = Tensor::contract(std::move(t11u), std::move(shu), {});
  // std::cout << env.proc_id << " | T12 = " << *t12u << "\n";

  return 0;
}
