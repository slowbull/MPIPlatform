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

#ifndef _SGD_UPDATER_
#define _SGD_UPDATER_

#include "Updater.h"
#include "../Gradient/Gradient.h"

class SGDUpdater : public Updater {
 protected:
  void PrepareGradient(Datapoint *datapoints, Gradient *g, const std::vector<int> &left_right) override {
	std::vector<double> &cur_model = model->ModelData();
	model->PrecomputeCoefficients(datapoints, g, cur_model, left_right);
	model->ComputeL2Gradient(g, cur_model);
  }

 public:
  SGDUpdater(Model *model, Datapoint *datapoints) : Updater(model, datapoints) {}

  ~SGDUpdater() {}
};

#endif
