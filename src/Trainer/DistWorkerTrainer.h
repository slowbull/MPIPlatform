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

#ifndef _DIST_WORKER_TRAINER_
#define _DIST_WORKER_TRAINER_ 

#include <time.h>
#include "mpi.h"
#include "../Datapoint/ARMADatapoint.h"
#include "../Gradient/Gradient.h"


class DistWorkerTrainer : public Trainer {
 public:
  DistWorkerTrainer(Model *model, Datapoint *datapoints) : Trainer(model, datapoints) {
  }
  ~DistWorkerTrainer() {
  }

  TrainStatistics Train(Model *model, Datapoint *datapoints, Updater *updater) override {

    // Keep track of statistics of training.
	TrainStatistics stats;

	Datapoint *sub_datapoints = new ARMADatapoint();

	// members definition
	double flag_epoch = 1;
	double flag_break = 0;
	MPI_Status status;
	std::vector<double> &local_model = model->ModelData();
	std::vector<double> server_model(local_model);
	// messages. format: 0-model.size()-1: model, model.size(): flag_epoch[0,1], model.size()+1: break ([0,1]);
	std::vector<double> message(local_model.size()+2, 0);

	Timer gradient_timer;
	// Train.
	int epoch = 0;
	while (true) {
	  // update learning_rate
      double learning_rate = FLAGS_learning_rate / std::pow(1+epoch, FLAGS_learning_rate_dec);
	  epoch += 1;

	  srand(epoch);

	  // epoch signal from server.
	  if (flag_epoch) {
	    this->EpochBegin(0, gradient_timer, model, datapoints, &stats);
	    updater->EpochBegin();
		flag_epoch = 0;
	  }
	  if (flag_break) {
	    break;
	  }


	  for (int iter = 0; iter < FLAGS_in_iters; iter++) {
		  int left_index = rand() % datapoints->GetSize();
		  int right_index = left_index + FLAGS_mini_batch; 
		  if (right_index > datapoints->GetSize()) 
			right_index = datapoints->GetSize();	

		  sub_datapoints->SetFeatures(datapoints->GetFeaturesCols(left_index, right_index - 1));
		  sub_datapoints->SetLabels(datapoints->GetLabelsRows(left_index, right_index - 1));

		  updater->Update(model, sub_datapoints, gradient);
		  updater->ApplyGradient(gradient, learning_rate);
	  }

	  for (int i = 0; i < model->NumParameters(); i++) {
		if (FLAGS_accelerate) 
	      message[i] = local_model[i];
		else
		  message[i] = local_model[i] - server_model[i];

		local_model[i] = local_model[i] - FLAGS_kappa * learning_rate * (local_model[i] - server_model[i]);
	  }


	  MPI_Send(&message[0], model->NumParameters(), MPI_DOUBLE, 0, 101, MPI_COMM_WORLD);

  	  if(FLAGS_accelerate) y_larger_than_past();
		
	  MPI_Recv(&message[0], local_model.size()+2, MPI_DOUBLE, 0, 102, MPI_COMM_WORLD, &status);

	  server_model.assign(message.begin(), message.end()-2);
	  flag_epoch = *(message.end()-2);
	  flag_break = *(message.end()-1);

	}

	delete sub_datapoints;
	return stats;
  }

  bool y_larger_than_past() {
	std::vector<double> &local_model = model->ModelData();
	std::vector<double> copy_model(local_model);

	double worker_eval = 0, master_eval = 0; 
	double worker_loss = 0, master_loss = 0;
	int worker_num = datapoints->GetSize();

	MPI_Bcast(&local_model[0], local_model.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
	worker_loss = model->ComputeLoss(datapoints, worker_eval);
	worker_loss *=  worker_num; 
	MPI_Reduce(&worker_loss, &master_loss, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);


	MPI_Bcast(&local_model[0], local_model.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
	worker_loss = model->ComputeLoss(datapoints, worker_eval);
	worker_loss *=  worker_num; 
	MPI_Reduce(&worker_loss, &master_loss, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	// return to local model.
	local_model = copy_model;

	return true;
  }

  virtual void EpochBegin(int epoch, Timer &gradient_timer, Model *model, Datapoint *datapoints, TrainStatistics *stats) override {
	double worker_eval = 0, master_eval = 0; 
	double worker_loss = 0, master_loss = 0;
	int worker_num = 0, master_num = 0;


	std::vector<double> &local_model = model->ModelData();
	std::vector<double> copy_model(local_model);
	MPI_Bcast(&local_model[0], local_model.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

	worker_num = datapoints->GetSize();
	worker_loss = model->ComputeLoss(datapoints, worker_eval);
	worker_loss *=  worker_num; 
	worker_eval *=  worker_num; 
	MPI_Reduce(&worker_num, &master_num, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&worker_eval, &master_eval, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&worker_loss, &master_loss, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	local_model = copy_model;
  }

};

#endif
