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
#ifndef _DIST_SERVER_TRAINER_
#define _DIST_SERVER_TRAINER_ 

#include "../Datapoint/ARMADatapoint.h"
#include "../Gradient/Gradient.h"
#include "../Tools/Tools.h"

DEFINE_double(moving_rate, 0.9, "moving rate with respect to weight in the server.");
DEFINE_double(kappa, 0.1, "parameter kappa imposed on local problem.");
DEFINE_bool(accelerate, false, "accelerated elastic average.");
DEFINE_int32(past_size, 1, "number of past models stored in accelerated mothoed.");

class DistServerTrainer : public Trainer {
 public:
	
  std::vector<double> past_losses;
  std::vector<std::vector<double> > past_models;

  DistServerTrainer(Model *model, Datapoint *datapoints) : Trainer(model, datapoints) {
	past_losses.resize(FLAGS_past_size, 0);
	past_models.resize(FLAGS_past_size, std::vector<double>(model->NumParameters(), 0));
  }
  ~DistServerTrainer() {}

  TrainStatistics Train(Model *model, Datapoint *datapoints, Updater *updater) override {
 
    // Keep track of statistics of training.
    TrainStatistics stats;

	Datapoint *sub_datapoints = new ARMADatapoint();
	gradient->coeffs.resize(model->NumParameters(), 0);

	MPI_Status status;

	// Train.
	Timer gradient_timer;
	printf("Epoch: 	Time(s): Loss:   Evaluation(AUC or Accuracy): \n");
	for (int epoch = 0; epoch < FLAGS_n_epochs; epoch++) {
	  srand(epoch);

      // compute loss and print working time.
	  if (epoch % FLAGS_interval_print == 0)  {
	  	this->EpochBegin(epoch, gradient_timer, model, datapoints, &stats);
	  	updater->EpochBegin();
	  }

	  gradient_timer.Tick();
      std::vector<int> delay_counter(FLAGS_num_workers, 1);
	  std::vector<double> worker_gradient(model->NumParameters(), 0);
	  std::vector<double> message;

		int cur_worker_size = 0;
		std::vector<int> cur_received_workers(FLAGS_num_workers, 0);
		std::fill(gradient->coeffs.begin(), gradient->coeffs.end(), 0);

		// receive information and update.
		bool flag_receive = true;
		while(flag_receive) {
		  MPI_Probe(MPI_ANY_SOURCE, 101, MPI_COMM_WORLD, &status);	
		  int taskid = status.MPI_SOURCE;
		  MPI_Recv(&worker_gradient[0], worker_gradient.size(), MPI_DOUBLE, taskid, 101, MPI_COMM_WORLD, &status);

		  for(int i = 0; i < gradient->coeffs.size(); i++) {
		  	gradient->coeffs[i] += worker_gradient[i];
		  }

		  cur_worker_size += 1;
		  delay_counter[taskid - 1] = 1;
		  cur_received_workers[taskid - 1] = 1;

		  flag_receive = false;
		  if ( ((cur_worker_size < FLAGS_group_size) || (max_element(delay_counter) > FLAGS_max_delay)) && (epoch < FLAGS_n_epochs - 1))
		    flag_receive = true;
		  if ( (cur_worker_size < FLAGS_num_workers) && (epoch == FLAGS_n_epochs - 1 || (epoch+1) % FLAGS_interval_print == 0)) 
			flag_receive = true;
		}

		// update model. 
		std::vector<double> & master_model = model->ModelData();
		std::vector<double> model_copy(master_model);
		for (int i = 0; i < model->NumParameters(); i++) {
		  if (FLAGS_accelerate)
			master_model[i] = master_model[i] + FLAGS_moving_rate * (gradient->coeffs[i] / cur_worker_size - master_model[i]);
		  else
			master_model[i] = master_model[i] + FLAGS_moving_rate * gradient->coeffs[i] / cur_worker_size;
		}

		// build message.
		if (FLAGS_accelerate) {
		  message.resize(model->NumParameters(), 0);
		  for (int i = 0; i < model->NumParameters(); i++)
		    message[i] = master_model[i] + 1.0 * (epoch - 1) / (epoch + 2) * (master_model[i] - model_copy[i]);
		  if (y_larger_than_past(epoch, master_model, message))
			message = master_model;
		}
		else
		  message = master_model;

	    if ((epoch+1) % FLAGS_interval_print == 0)
		  message.push_back(1);
		else
		  message.push_back(0);

		if (epoch < FLAGS_n_epochs - 1) {
		  message.push_back(0);
		}
		else {
		  message.push_back(1);
		}

		// send messages to workers.
		for(int i = 0; i < FLAGS_num_workers; i++) {
		  if(cur_received_workers[i] == 0)
			delay_counter[i] += 1;
		  else {
			MPI_Send(&message[0], message.size(), MPI_DOUBLE, i+1, 102, MPI_COMM_WORLD);
		  }
		}
		
	  updater->EpochFinish();
	  gradient_timer.Tock();
	}
	model->StoreModel();
	return stats;
  }

  bool y_larger_than_past(int epoch, const std::vector<double>& model, const std::vector<double>& acc_model) {
	double worker_eval = 0, master_eval = 0; 
	double worker_loss = 0, master_loss = 0;
	std::vector<double> local_model(model);

	MPI_Bcast(&local_model[0], model.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Reduce(&worker_loss, &master_loss, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	past_losses[epoch % FLAGS_past_size] = master_loss;
	past_models[epoch % FLAGS_past_size] = model;

	int past_max_idx = 0;
	for (int i = 1; i < FLAGS_past_size; i++)
	  if (past_losses[past_max_idx] < past_losses[i])
	    past_max_idx = i;

	worker_loss = 0, master_loss = 0;
	local_model = acc_model;
	MPI_Bcast(&local_model[0], model.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Reduce(&worker_loss, &master_loss, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	
	if (past_losses[past_max_idx] < master_loss)
	  return true;
	return false;
  }

  virtual void EpochBegin (int epoch, Timer &gradient_timer, Model *model, Datapoint *datapoints, TrainStatistics *stats) override {
	double cur_time;
    if(stats->times.size()==0) 
	  cur_time = 0;
	else
	  cur_time = gradient_timer.elapsed + stats->times[stats->times.size()-1];

	std::vector<double> & local_model = model->ModelData();
	double worker_eval = 0, master_eval = 0; 
	double worker_loss = 0, master_loss = 0;
	int worker_num = 0, master_num = 0;

	MPI_Bcast(&local_model[0], local_model.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Reduce(&worker_num, &master_num, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&worker_eval, &master_eval, 1,  MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&worker_loss, &master_loss, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	master_loss /= master_num; 
	master_eval /= master_num; 

	Trainer::TrackTimeLoss(cur_time, master_loss, stats);
	if (FLAGS_print_loss_per_epoch && epoch % FLAGS_interval_print == 0) 
	  Trainer::PrintTimeLoss(cur_time, master_loss, epoch, master_eval);
  }
};

#endif
