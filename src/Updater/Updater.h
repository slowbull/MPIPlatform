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

#ifndef _UPDATER_
#define _UPDATER_

#include "../Gradient/Gradient.h"

class Updater {
 protected:
  // Keep a reference of the model and datapoints, and partition ordering.
  Model *model;
  Datapoint* datapoints;

  virtual void PrepareGradient(Datapoint *datapoint, Gradient *g) = 0;


 public:
  Updater(Model *model, Datapoint *datapoints) {
    this->model = model;
    this->datapoints = datapoints;
  }

  Updater() {}
  virtual ~Updater() {}

  virtual void ApplyGradient(Gradient *gradient, double learning_rate) {
	std::vector<double> &model_data = model->ModelData();
	int para_size = model->NumParameters();
	for (int i = 0; i < para_size; i++) {
		model_data[i] -= learning_rate *  gradient->coeffs[i];
	}
  }

  virtual void ApplyProximalOperator(double gamma){
	std::vector<double> &model_data = model->ModelData();
	model->ProximalOperator(model_data, gamma);
  }

  // Main update method, which is run by multiple threads.
  virtual void Update(Model *model, Datapoint *sub_datapoints, Gradient *gradient) {
	gradient->Clear();
	gradient->datapoint = sub_datapoints;
	PrepareGradient(sub_datapoints, gradient);
  }

  // Called before epoch begins.
  virtual void EpochBegin() {}

  // Called when the epoch ends.
  virtual void EpochFinish() {}
};

#endif
