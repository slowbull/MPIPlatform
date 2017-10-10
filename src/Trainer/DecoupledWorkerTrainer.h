/*
* Copyright 2017 [Zhouyuan Huo]
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

#ifndef _DECOUPLED_WORKER_TRAINER_
#define _DECOUPLED_WORKER_TRAINER_ 

#include <time.h>
#include "mpi.h"
#include "../Gradient/Gradient.h"


class DecoupledWorkerTrainer : public Trainer {
 public:
  DecoupledWorkerTrainer(Model *model, Datapoint *datapoints) : Trainer(model, datapoints) {
  }
  ~DecoupledWorkerTrainer() {
  }

  TrainStatistics Train(Model *model, Datapoint *datapoints, Updater *updater) override {

    // Keep track of statistics of training.
	TrainStatistics stats;
	stats.working_time = 0;
	stats.waiting_time = 0;

	// members definition
	double flag_epoch = 1;
	double flag_break = 0;
	int epoch = 0;
	MPI_Status status;
	std::vector<double> &local_model = model->ModelData();
	// messages. format: 0-model.size()-1: model, model.size(): flag_epoch[0,1], model.size()+1: break ([0,1]);
	std::vector<double> message(local_model.size()+2, 0);
	std::vector<int> left_right(2, 0);

    double learning_rate = FLAGS_learning_rate;

	Timer gradient_timer;
	// Train.
	while (true) {

	  // epoch signal from server.
	  if (flag_epoch) {
	    this->EpochBegin(0, gradient_timer, model, datapoints, &stats);
	    updater->EpochBegin();
		flag_epoch = 0;
		learning_rate = FLAGS_learning_rate / std::pow(1+epoch, FLAGS_learning_rate_dec);
	    srand(epoch);
		epoch++;
		continue;
	  }
	  if (flag_break) {
	    break;
	  }

	  gradient_timer.Tick();

	  int left_index = rand() % (datapoints->GetSize() - FLAGS_mini_batch);
	  int right_index = left_index + FLAGS_mini_batch; 
	  if (right_index > datapoints->GetSize()) 
		right_index = datapoints->GetSize();	

	  left_right[0] = left_index;
	  left_right[1] = right_index;

	  updater->Update(model, datapoints, gradient, left_right);

	  // proximal operator occurs in the worker if not decoupled.
	  std::vector<double> model_copy = local_model;
	  updater->ApplyGradient(gradient, learning_rate);

	  if(FLAGS_l1_lambda){
	    updater->ApplyProximalOperator(learning_rate * FLAGS_l1_lambda);
	  }
	  else if(FLAGS_trace_lambda){
		updater->ApplyProximalOperator(learning_rate * FLAGS_trace_lambda);
	  }

	  for(size_t i = 0; i < model_copy.size(); i++){
		gradient->coeffs[i] = model_copy[i] - local_model[i];
	  }

	  gradient_timer.Tock();
	  stats.working_time += gradient_timer.elapsed;

	  gradient_timer.Tick();

	  MPI_Send(&gradient->coeffs[0], gradient->coeffs.size(), MPI_DOUBLE, 0, 101, MPI_COMM_WORLD);
		
	  MPI_Recv(&message[0], local_model.size()+2, MPI_DOUBLE, 0, 102, MPI_COMM_WORLD, &status);

	  gradient_timer.Tock();
	  stats.waiting_time += gradient_timer.elapsed;

	  local_model.assign(message.begin(), message.end()-2);
	  flag_epoch = *(message.end()-2);
	  flag_break = *(message.end()-1);
	}

	return stats;
  }

  virtual void EpochBegin(int epoch, Timer &gradient_timer, Model *model, Datapoint *datapoints, TrainStatistics *stats) override {
	double worker_eval = 0; 
	double worker_loss = 0;
	int worker_num = 0;
	double worker_working_time = stats->working_time;
	double worker_waiting_time = stats->waiting_time;

	worker_num = datapoints->GetSize();
	worker_loss = model->ComputeLoss(datapoints, worker_eval);

	worker_loss *=  worker_num; 
	worker_eval *=  worker_num; 

	std::vector<double> worker_message, server_message;
	worker_message.push_back(worker_num);
	worker_message.push_back(worker_eval);
	worker_message.push_back(worker_loss);
	worker_message.push_back(worker_working_time);
	worker_message.push_back(worker_waiting_time);

	MPI_Reduce(&worker_message[0], &server_message[0], 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }

};

#endif
