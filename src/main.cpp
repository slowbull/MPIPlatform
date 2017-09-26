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

#include <iostream>
#include "run.h"
#include "mpi.h"


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
  else if (FLAGS_multi_class_trace) {
	Run<MULTICLASSTRACEModel, ARMADatapoint>(taskid);
  }

  MPI_Finalize();
  return 0;
}
