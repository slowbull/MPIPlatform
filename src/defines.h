
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
#ifndef _DEFINES_
#define _DEFINES_

#include <math.h>
#include <cstdlib>
#include <map>
#include <set>
#include <cstring>
#include <cstdlib>
#include <gflags/gflags.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <time.h>
#include <sys/time.h>
#include <armadillo>
#include "mpi.h"

#include "Datapoint/Datapoint.h"
#include "Gradient/Gradient.h"
#include "Model/Model.h"

using namespace arma;

// Timer use std::chrono maybe a faster way.
class Timer {
 public:
  double start;
  double end;
  double elapsed;

  Timer(){}

  virtual ~Timer(){}

  inline void Tick(){
    start = MPI_Wtime();
  }

  inline void Tock(){
    end = MPI_Wtime();
    elapsed = end - start;
  }
};


DEFINE_string(data_file, "blank", "Input data file.");
DEFINE_string(model_snapshot, "", "model snapshot to be loaded.");
DEFINE_int32(n_epochs, 100, "Number of passes of data in training.");
DEFINE_int32(mini_batch, 1, "mini batch size in each epoch.");
DEFINE_int32(in_iters, 10, "Inside iterations");
DEFINE_int32(max_delay, 10, "max delay for each worker");
DEFINE_int32(group_size, 1, "group size of workers received by server in each inner iteration.");
DEFINE_int32(num_workers, 1, "Number of workers in in the cluster.");
DEFINE_int32(d1, 10, "dimmension d1");
DEFINE_int32(d2, 10, "dimmension d2");
DEFINE_int32(d3, 10, "dimmension d3");
DEFINE_double(learning_rate, .001, "Learning rate.");
DEFINE_double(learning_rate_dec, 0, "Learning rate decay. 1/(1+epoch)^beta");
DEFINE_double(l1_lambda, 0, "regularization parameter for l1 norm.");
DEFINE_double(l2_lambda, 0, "regularization parameter for l2 norm.");
DEFINE_double(trace_lambda, 0, "regularization parameter for trace norm.");
DEFINE_bool(print_loss_per_epoch, false, "Should compute and print loss every epoch.");
DEFINE_int32(interval_print, 1, "Interval in which to print the loss.");
DEFINE_bool(distribute, false, "code is run at distributed cluster or run on localhost otherwise.");
DEFINE_bool(decouple, false, "proximal operator is running in the worker. decoupled algorithm.");
DEFINE_double(lambda, 100, "estimated largetst singular value in PCA problem.");

// Flags for application types.
DEFINE_bool(logistic_l2_l1, false, "logistic loss with l2 and l1 norm regularization type.");
DEFINE_bool(least_l2_l1, false, "least square loss with l2 and l1 norm regularization type.");
DEFINE_bool(fcn, false, "three layers fully connected network with l2 norm regularization type.");
DEFINE_bool(multi_class_trace, false, "multiclass logistic loss with l2 norm and trace norm regularization type.");
DEFINE_bool(pca, false, "convex optimization for PCA. ref: Fast and Simple PCA via Convex Optimization.");

// Flags for updating types.
DEFINE_bool(svrg, false, "Use SVRG.");
DEFINE_bool(dfsdca, false, "Use dual free sdca.");
DEFINE_bool(sgd, false, "Use SGD.");


#include "Updater/Updater.h"
#include "Updater/SGDUpdater.h"
#include "Updater/SVRGUpdater.h"
#include "Updater/DFSDCAUpdater.h"
//#include "Updater/DisSVRGUpdater.h"

#include "Trainer/Trainer.h"
#include "Trainer/ServerTrainer.h"
#include "Trainer/WorkerTrainer.h"
#include "Trainer/DecoupledServerTrainer.h"
#include "Trainer/DecoupledWorkerTrainer.h"

#include "Datapoint/ARMADatapoint.h"
#include "Datapoint/PCADatapoint.h"

#include "Model/LOGISTICL2L1Model.h"
#include "Model/LSL2L1Model.h"
#include "Model/FCNModel.h"
#include "Model/MULTICLASSTRACEModel.h"
#include "Model/PCAModel.h"

#endif
