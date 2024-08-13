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

  MPI_Barrier(MPI_COMM_WORLD);
  qtnh::tptr tp3 = DenseTensor::make(env, t3_dis_dims, t3_loc_dims, std::move(t3_els));
  tp3 = Tensor::rebcast(std::move(tp3), { 1, 1, 1 });
  std::cout << env.proc_id << " | T3 = " << *tp3 << std::endl;

  tidx_tup t4_dis_dims = {}, t4_loc_dims = { 4, 2 };
  tidx_tup t5_dis_dims = {}, t5_loc_dims = { 2, 2, 2 };
  std::vector<tel> t4_els = { 5.0 + 5.0i, 6.0 + 6.0i, 7.0 + 7.0i, 8.0 + 8.0i, 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i };
  std::vector<tel> t5_els = { 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i, 5.0 + 5.0i, 6.0 + 6.0i, 7.0 + 7.0i, 8.0 + 8.0i };

  MPI_Barrier(MPI_COMM_WORLD);
  tptr tp4 = DenseTensor::make(env, t4_dis_dims, t4_loc_dims, std::move(t4_els));
  tptr tp5 = DenseTensor::make(env, t5_dis_dims, t5_loc_dims, std::move(t5_els));
  tp4 = Tensor::rescatter(std::move(tp4), 1);

  std::cout << env.proc_id << " | T4 = " << *tp4 << "\n";
  std::cout << env.proc_id << " | T5 = " << *tp5 << "\n";

  TensorNetwork tn;
  auto tid4 = tn.insert(std::move(tp4));
  auto tid5 = tn.insert(std::move(tp5));
  auto bid1 = tn.addBond(tid4, tid5, {{ 1, 2 }});

  MPI_Barrier(MPI_COMM_WORLD);
  tid4 = tn.contractBond(bid1);
  std::cout << env.proc_id << " | TN(4) = " << *tn.tensor(tid4) << "\n";

  MPI_Barrier(MPI_COMM_WORLD);
  tp4 = tn.extract(tid4);
  tp4 = Tensor::permute(std::move(tp4), { 2, 1, 0 });
  std::cout << env.proc_id << " | T4 (permute) = " << *tp4 << "\n";

  MPI_Barrier(MPI_COMM_WORLD);
  tp5 = DenseTensor::make(env, {}, { 3, 3 }, { 0 ,1, 2, 0, 1, 2, 0, 1, 2 });
  tp5 = Tensor::rescatter(std::move(tp5), 1);
  std::cout << env.proc_id << " | T5 (new) = " << *tp5 << "\n";

  MPI_Barrier(MPI_COMM_WORLD);
  tp5 = Tensor::permute(std::move(tp5), { 1, 0 });
  std::cout << env.proc_id << " | T5 (permute) = " << *tp5 << "\n";

  MPI_Barrier(MPI_COMM_WORLD);
  tptr tp6 = DenseTensor::make(env, {}, { 2, 2, 2, 2 }, { 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7 });
  tp6 = Tensor::rescatter(std::move(tp6), 2);
  std::cout << env.proc_id << " | T6 = " << *tp6 << "\n";

  MPI_Barrier(MPI_COMM_WORLD);
  tp6 = Tensor::permute(std::move(tp6), { 1, 0, 2, 3 });
  std::cout << env.proc_id << " | T6 (permute 1) = " << *tp6 << "\n";

  MPI_Barrier(MPI_COMM_WORLD);
  tp6 = Tensor::permute(std::move(tp6), { 1, 0, 3, 2 });
  std::cout << env.proc_id << " | T6 (permute 2) = " << *tp6 << "\n";

  MPI_Barrier(MPI_COMM_WORLD);
  tp6 = Tensor::permute(std::move(tp6), { 0, 1, 3, 2 });
  std::cout << env.proc_id << " | T6 (normal) = " << *tp6 << "\n";

  MPI_Barrier(MPI_COMM_WORLD);
  tptr tp7 = SwapTensor::make(env, 2, 0);
  std::cout << env.proc_id << " | T7 = " << *tp7 << "\n";

  MPI_Barrier(MPI_COMM_WORLD);
  tptr tp8 = Tensor::contract(std::move(tp6), std::move(tp7), {{ 2, 0 }, { 3, 1 }});
  std::cout << env.proc_id << " | T8 = " << *tp8 << "\n";

  MPI_Barrier(MPI_COMM_WORLD);
  tptr tpi = IdenTensor::make(env, {}, { 2, 2, 2, 2 }, 0, 0);
  std::cout << env.proc_id << " | ID = " << *tpi << "\n";
  
  MPI_Barrier(MPI_COMM_WORLD);
  tp8 = Tensor::contract(std::move(tp8), std::move(tpi), {{ 2, 1 }, { 3, 0 }});
  std::cout << env.proc_id << " | T8 (id) = " << *tp8 << "\n";

  MPI_Barrier(MPI_COMM_WORLD);
  tpi = IdenTensor::make(env, { 2 }, { 2 }, 1, 0);
  std::cout << env.proc_id << " | ID (distributed) = " << *tpi << "\n";

  MPI_Barrier(MPI_COMM_WORLD);
  tp8 = Tensor::contract(std::move(tp8), std::move(tpi), {{ 1, 0 }});
  std::cout << env.proc_id << " | T8 (convert 1) = " << *tp8 << "\n";

  MPI_Barrier(MPI_COMM_WORLD);
  tp8 = Tensor::rebcast(std::move(tp8), { 1, 1 ,0 });
  std::cout << env.proc_id << " | T8 (re-bcast) = " << *tp8 << "\n";

  tpi = IdenTensor::make(env, { 2 }, { 2 }, 0, 0);
  tp8 = Tensor::contract(std::move(tp8), std::move(tpi), {{ 3, 1 }});
  std::cout << env.proc_id << " | T8 (convert 2) = " << *tp8 << "\n";

  tp1 = DenseTensor::make(env, {}, { 2 }, { 1, 1 });
  tp2 = IdenTensor::make(env, { 2 }, { 2 }, 0, 0);
  tp1 = Tensor::contract(std::move(tp1), std::move(tp2), {{ 0, 1 }});
  std::cout << env.proc_id << " | T1 (fully distributed) = " << *tp1 << "\n";

  tp2 = IdenTensor::make(env, { 2 }, { 2 }, 1, 0);
  tp1 = Tensor::contract(std::move(tp1), std::move(tp2), {{ 0, 0 }});
  tp1 = Tensor::rebcast(std::move(tp1), { 1, 1, 0 });
  std::cout << env.proc_id << " | T1 (fully local) = " << *tp1 << "\n";

  return 0;
}
