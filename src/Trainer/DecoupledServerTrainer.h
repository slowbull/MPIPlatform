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
#ifndef _DECOUPLED_SERVER_TRAINER_
#define _DECOUPLED_SERVER_TRAINER_ 

#include "../Gradient/Gradient.h"
#include "../Tools/Tools.h"


class DecoupledServerTrainer : public Trainer {
 public:

  DecoupledServerTrainer(Model *model, Datapoint *datapoints) : Trainer(model, datapoints) {}
  ~DecoupledServerTrainer() {}

  TrainStatistics Train(Model *model, Datapoint *datapoints, Updater *updater) override {
 
    // Keep track of statistics of training.
    TrainStatistics stats;

	MPI_Status status;

	// Train.
	Timer gradient_timer;
	printf("Epoch: 	Time(s): Loss:   Evaluation(AUC or Accuracy):   Working_Time:   Waiting_Time: \n");
	for (int epoch = 0; epoch < FLAGS_n_epochs; epoch++) {
	  srand(epoch);

      // compute loss and print working time.
	  this->EpochBegin(epoch, gradient_timer, model, datapoints, &stats);

	  gradient_timer.Tick();
	  updater->EpochBegin();
      std::vector<int> delay_counter(FLAGS_num_workers, 1);
	  std::vector<double> worker_gradient(model->NumParameters(), 0);
	  std::vector<double> message;

	  for(int iter_counter = 0; iter_counter < FLAGS_in_iters; iter_counter++) {
		int cur_worker_size = 0;
		std::vector<int> cur_revceived_workers(FLAGS_num_workers, 0);

		// receive information and update.
		bool flag_receive = true;
		while(flag_receive) {
		  MPI_Probe(MPI_ANY_SOURCE, 101, MPI_COMM_WORLD, &status);	
		  int taskid = status.MPI_SOURCE;
		  MPI_Recv(&worker_gradient[0], worker_gradient.size(), MPI_DOUBLE, taskid, 101, MPI_COMM_WORLD, &status);

		  gradient->coeffs = worker_gradient;
		  updater->ApplyGradient(gradient, 1.0);

		  cur_worker_size += 1;
		  delay_counter[taskid - 1] = 1;
		  cur_revceived_workers[taskid - 1] = 1;

		  flag_receive = false;
		  if ( ((cur_worker_size < FLAGS_group_size) || (max_element(delay_counter) > FLAGS_max_delay)) && (iter_counter < FLAGS_in_iters - 1))
		    flag_receive = true;
		  if ( (cur_worker_size < FLAGS_num_workers) && (iter_counter == FLAGS_in_iters - 1)) 
			flag_receive = true;
		}

		// build message.
		std::vector<double> & master_model = model->ModelData();
		message = master_model;
		if (iter_counter < FLAGS_in_iters - 1) {
		  message.push_back(0);
		  message.push_back(0);
		}
		else if(epoch < FLAGS_n_epochs - 1) {
		  message.push_back(1);
		  message.push_back(0);
		}
		else {
		  message.push_back(0);
		  message.push_back(1);
		}

		// send messages to workers.
		for(int i = 0; i < FLAGS_num_workers; i++) {
		  if(cur_revceived_workers[i] == 0)
			delay_counter[i] += 1;
		  else {
			MPI_Send(&message[0], message.size(), MPI_DOUBLE, i+1, 102, MPI_COMM_WORLD);
		  }
		}
	  }
		
	  updater->EpochFinish();
	  gradient_timer.Tock();
	}
	model->StoreModel();
	return stats;
  }

  virtual void EpochBegin (int epoch, Timer &gradient_timer, Model *model, Datapoint *datapoints, TrainStatistics *stats) override {
	double cur_time;
    if(stats->times.size()==0) 
	  cur_time = 0;
	else
	  cur_time = gradient_timer.elapsed + stats->times[stats->times.size()-1];

	double master_eval = 0; 
	double master_loss = 0;
	double master_num = 0;
	double master_working_time = 0;
	double master_waiting_time = 0;


	std::vector<double> worker_message(5, 0), server_message(5, 0);
	MPI_Reduce(&worker_message[0], &server_message[0], 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	master_num = server_message[0];
	master_eval = server_message[1];
	master_loss = server_message[2];
	master_working_time = server_message[3];
	master_waiting_time = server_message[4];

	master_loss /= master_num; 
	master_eval /= master_num; 

	Trainer::TrackTimeLoss(cur_time, master_loss, stats);
	if (FLAGS_print_loss_per_epoch && epoch % FLAGS_interval_print == 0) {
	  Trainer::PrintTimeLoss(cur_time, master_loss, epoch, master_eval, master_working_time, master_waiting_time);
	}
  }
};

#endif
