/*
* Copyright 2016 [See AUTHORS file for list of authors]
* Modifications copyright 2017 [Zhouyuan Huo]
*
*    Licensed under the Apache License, Version 2.0 (the "License");
*    you may not use this file except in compliance with the License.
*    You may obtain a copy of the License at
*
*        http://www.apache.org/licenses/LICENSE-2.0
*
*    Unless required by applicable law or agreed to in writing, software
*    distributed under the License is distributed on an "AS IS" BASIS,
*    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*    See the License for the specific language governing permissions and
*    limitations under the License.
*/

// how to use:
// mpirun -np 9 ./mpiplatform -logistic_l2_l1 -num_workers=8 -data_file="/home/jonny/ZHOU/Data/libsvm/covtype/8/" -print_loss_per_epoch -d1=54 -learning_rate=1e-2 -n_epochs=100 -mini_batch=100 -in_iters=1000 -group_size=8 -sgd
// mpirun -np 9 ./mpiplatform -logistic_l2_l1 -num_workers=8 -data_file="/home/jonny/ZHOU/Data/libsvm/covtype/8/" -print_loss_per_epoch -d1=54 -learning_rate=1e-1 -n_epochs=100 -mini_batch=100 -in_iters=1000 -group_size=8 -svrg
// mpirun -np 9 ./mpiplatform -fcn -num_workers=8 -data_file="/home/jonny/ZHOU/Data/libsvm/multi_covtype/8/" -print_loss_per_epoch -d1=54 -d2=20 -d3=7 -learning_rate=1e-3 -n_epochs=100 -mini_batch=10 -in_iters=1000 -group_size=8 -svrg

#include <iostream>
#include "run.h"
#include "mpi.h"

// Flags for application types.
DEFINE_bool(logistic_l2_l1, false, "logistic loss with l2 and l1 norm regularization type.");
DEFINE_bool(fcn, false, "three layers fully connected network with l2 norm regularization type.");

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  int taskid, numtasks;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

  if (FLAGS_logistic_l2_l1) {
	Run<LOGISTICL2L1Model, ARMADatapoint>(taskid);
  }
  else if (FLAGS_fcn) {
	Run<FCNModel, ARMADatapoint>(taskid);
  }

  MPI_Finalize();
  return 0;
}
