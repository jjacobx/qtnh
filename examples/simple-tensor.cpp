#include <iostream>
#include "qtnh.hpp"

using namespace qtnh;
using namespace qtnh::ops;
using namespace std::complex_literals;


int main() {
  QTNHEnv env;

  tidx_tup t1_dis_dims = { 2 }, t1_loc_dims = { 2, 2, 2 };
  std::vector<tel> t1_els;
  if (env.proc_id == 0) {
    t1_els = { 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i, 5.0 + 5.0i, 6.0 + 6.0i, 7.0 + 7.0i, 8.0 + 8.0i };
  } else if (env.proc_id == 1) {
    t1_els = { 5.0 - 5.0i, 6.0 - 6.0i, 7.0 - 7.0i, 8.0 - 8.0i, 1.0 - 1.0i, 2.0 - 2.0i, 3.0 - 3.0i, 4.0 - 4.0i };
  }

  tptr tp1 = DenseTensor::make(env, t1_dis_dims, t1_loc_dims, std::move(t1_els));
  std::cout << env.proc_id << " | T1 = " << *tp1 << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  tp1 = Tensor::rebcast(std::move(tp1), { 2, 2, 0 });
  std::cout << env.proc_id << " | T1 (re-bcast 1) = " << *tp1 << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  tp1 = Tensor::rescatter(std::move(tp1), -1);
  std::cout << env.proc_id << " | T1 (re-scatter 1) = " << *tp1 << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  tp1 = Tensor::rebcast(std::move(tp1), { 1, 1, 0 });
  std::cout << env.proc_id << " | T1 (re-bcast 2) = " << *tp1 << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  tp1 = Tensor::permute(std::move(tp1), { 0, 1, 2, 3 });
  std::cout << env.proc_id << " | T1 (permute 1) = " << *tp1 << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  tp1 = Tensor::rescatter(std::move(tp1), 2);
  std::cout << env.proc_id << " | T1 (re-scatter 2) = " << *tp1 << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  tp1 = Tensor::permute(std::move(tp1), { 1, 0, 2, 3 });
  std::cout << env.proc_id << " | T1 (permute 2) = " << *tp1 << std::endl;

  // Only casts return something more specific than tptr. 
  auto tp1_dt = Tensor::cast<DenseTensor>(tp1->copy());
  tptr tp1_2 = Tensor::convert<DenseTensor>(tp1->copy());
  tptr tp1_1 = tp1->copy();


  tidx_tup t2_dis_dims = {}, t2_loc_dims = { 2, 2 };
  std::vector<tel> t2_els = { 1.0, 0.0, 0.0, -1.0 };
  qtnh::tptr tp2 = DenseTensor::make(env, t2_dis_dims, t2_loc_dims, std::move(t2_els));
  std::cout << env.proc_id << " | T2 = " << *tp2 << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  tp1 = Tensor::contract(std::move(tp1), std::move(tp2), {{ 3, 0 }});
  std::cout << env.proc_id << " | T1 (contract 1) = " << *tp1 << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  tp2 = tp1_1->copy();
  tp1 = Tensor::contract(std::move(tp2), std::move(tp1), {{ 1, 1 }, { 2, 2 }});
  std::cout << env.proc_id << " | T1 (contract 2) = " << *tp1 << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  tp1 = Tensor::rebcast(std::move(tp1), { 1, 1, 4 });
  std::cout << env.proc_id << " | T1 (re-bcast 3) = " << *tp1 << std::endl;

  tidx_tup t3_dis_dims = { 2 }, t3_loc_dims = { 2, 3 };
  std::vector<tel> t3_els;
  if (env.proc_id == 0) {
    t3_els = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
  } else {
    t3_els = { 6.0, 7.0, 8.0, 9.0, 10.0, 11.0 };
  }

  qtnh::tptr tp3 = DenseTensor::make(env, t3_dis_dims, t3_loc_dims, std::move(t3_els));
  tp3 = Tensor::rebcast(std::move(tp3), { 1, 1, 1 });
  std::cout << env.proc_id << " | T3 = " << *tp3 << std::endl;

  tidx_tup t4_dis_dims = {}, t4_loc_dims = { 4, 2 };
  tidx_tup t5_dis_dims = {}, t5_loc_dims = { 2, 2, 2 };
  std::vector<tel> t4_els = { 5.0 + 5.0i, 6.0 + 6.0i, 7.0 + 7.0i, 8.0 + 8.0i, 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i };
  std::vector<tel> t5_els = { 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i, 5.0 + 5.0i, 6.0 + 6.0i, 7.0 + 7.0i, 8.0 + 8.0i };

  tptr tp4 = DenseTensor::make(env, t4_dis_dims, t4_loc_dims, std::move(t4_els));
  tptr tp5 = DenseTensor::make(env, t5_dis_dims, t5_loc_dims, std::move(t5_els));
  tp4 = Tensor::rescatter(std::move(tp4), 1);

  std::cout << env.proc_id << " | T4 = " << *tp4 << "\n";
  std::cout << env.proc_id << " | T5 = " << *tp5 << "\n";

  TensorNetwork tn;
  auto tid4 = tn.insert(std::move(tp4));
  auto tid5 = tn.insert(std::move(tp5));
  auto bid1 = tn.addBond(tid4, tid5, {{ 1, 2 }});

  tid4 = tn.contractBond(bid1);
  std::cout << env.proc_id << " | TN(4) = " << *tn.tensor(tid4) << "\n";

  tp4 = tn.extract(tid4);
  tp4 = Tensor::permute(std::move(tp4), { 2, 1, 0 });
  std::cout << env.proc_id << " | T4 (swap) = " << *tp4 << "\n";

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
