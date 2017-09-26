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

#ifndef _TRAINER_
#define _TRAINER_

#include <limits.h>
#include <float.h>

// Contains times / losses / etc
struct TrainStatistics {
  std::vector<double> times;
  std::vector<double> losses;
  double working_time;
  double waiting_time;
};

typedef struct TrainStatistics TrainStatistics;

class Trainer {
 protected:
  // Keep a reference of the model and datapoints, and partition ordering.
  Model *model;
  Datapoint* datapoints;
  Gradient *gradient;

  void TrackTimeLoss(double cur_time, double cur_loss, TrainStatistics *stats) {
	stats->times.push_back(cur_time);
	stats->losses.push_back(cur_loss);
  }

  void PrintTimeLoss(double cur_time, double cur_loss, int epoch, double cur_eval, double working_time, double waiting_time) {
	printf("%d     %f    %.15f    %f    %f    %f\n", epoch, cur_time, cur_loss, cur_eval, working_time, waiting_time);
  }

  virtual void EpochBegin(int epoch, Timer &gradient_timer, Model *model, Datapoint *datapoints, TrainStatistics *stats) {
  }

 public:
  Trainer(Model *model, Datapoint *datapoints) {
    this->model = model;
	this->datapoints = datapoints;
	gradient  = new Gradient();
  }

  virtual ~Trainer() {}

  // Main training method.
  virtual TrainStatistics Train(Model *model, Datapoint *datapoints, Updater *updater) = 0;
};

#endif
