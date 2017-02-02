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

#ifndef _SVRG_UPDATER_
#define _SVRG_UPDATER_

#include "Updater.h"
#include "../Gradient/Gradient.h"

class SVRGUpdater : public Updater {
 protected:
  std::vector<double> model_copy;
  Gradient * full_gradient;

  virtual void PrepareGradient(Datapoint *sub_datapoints, Gradient *gradient) override {
    Gradient * prev_gradient = new Gradient();
	std::vector<double> &cur_model = model->ModelData();
	model->PrecomputeCoefficients(sub_datapoints, gradient, cur_model);
	model->ComputeL2Gradient(gradient, cur_model);
	model->PrecomputeCoefficients(sub_datapoints, prev_gradient, model_copy);
	model->ComputeL2Gradient(prev_gradient, model_copy);
	for (int i = 0; i < model->NumParameters(); i++){
	  gradient->coeffs[i] += - prev_gradient->coeffs[i] + full_gradient->coeffs[i];
	}

	delete prev_gradient;
  }

  // compute full gradient and store model
  void ModelCopy() {
	int worker_num = 0, master_num = 0;
	Gradient * gradient = new Gradient(); 
	gradient->coeffs.resize(model->NumParameters(), 0);
	full_gradient->coeffs = gradient->coeffs;
	std::vector<double> &cur_model = model->ModelData();
	model_copy = cur_model; 

	if (model->taskid == 0) {
	  MPI_Reduce(&worker_num, &master_num, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	  MPI_Reduce(&gradient->coeffs[0], &full_gradient->coeffs[0], model->NumParameters(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	  for(int i=0; i < model->NumParameters(); i++) {
		full_gradient->coeffs[i] /= master_num;
	  }
	}
	else {
	  worker_num = datapoints->GetSize();
	  model->PrecomputeCoefficients(datapoints, gradient, cur_model);
	  model->ComputeL2Gradient(gradient, cur_model);
	  for(int i=0; i < model->NumParameters(); i++) {
		gradient->coeffs[i] *= worker_num;
	  }
	  MPI_Reduce(&worker_num, &master_num, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	  MPI_Reduce(&gradient->coeffs[0], &full_gradient->coeffs[0], model->NumParameters(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	}
	MPI_Bcast(&full_gradient->coeffs[0], model->NumParameters(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
	delete gradient;
  }

 public:
  SVRGUpdater(Model *model, Datapoint *datapoints) : Updater(model, datapoints) {
	full_gradient = new Gradient();
  }

  virtual void EpochBegin() override {
	Updater::EpochBegin();
	ModelCopy();
  }

  ~SVRGUpdater() {
	delete full_gradient;
  }
};

#endif
