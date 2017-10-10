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

#ifndef _DFSDCA_UPDATER_
#define _DFSDCA_UPDATER_

#include "Updater.h"
#include "../Gradient/Gradient.h"

class DFSDCAUpdater : public Updater {
 protected:
  int total_num;
  mat alpha;

  virtual void PrepareGradient(Datapoint *datapoints, Gradient *gradient, const std::vector<int> &left_right) override {
	std::vector<double> &cur_model = model->ModelData();
	gradient->coeffs.resize(model->NumParameters(), 0);
	std::fill(gradient->coeffs.begin(), gradient->coeffs.end(), 0);
	Gradient* cur_gradient = new Gradient();
	std::vector<int> sub_idx(2, 0);
	for (int i = left_right[0]; i < left_right[1]; i++){
	  sub_idx[0] = i;
	  sub_idx[1] = i+1;

	  model->PrecomputeCoefficients(datapoints, cur_gradient, cur_model, sub_idx);

	  for (int j = 0; j < model->NumParameters(); j++) {
		double tmp = 0;
		if (FLAGS_l2_lambda == 0) {
		  tmp = alpha(i, j) + cur_gradient->coeffs[j] -	1e-4 * cur_model[j];  
	      gradient->coeffs[j] += tmp / total_num / 1e-4;
		}
		else {
		  tmp = alpha(i, j) + cur_gradient->coeffs[j]; 
	      gradient->coeffs[j] += tmp / total_num / FLAGS_l2_lambda;
		}

	  	alpha(i, j) = alpha(i, j) - FLAGS_learning_rate * tmp; 
	  }
	}

	delete cur_gradient;
  }

  // broadcast total number.
  void CountSample() {
	int worker_num = 0, master_num = 0;

	if (model->taskid == 0) {
	  MPI_Reduce(&worker_num, &master_num, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	  total_num = master_num;
	}
	else {
	  worker_num = datapoints->GetSize();
	  MPI_Reduce(&worker_num, &master_num, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	}
	MPI_Bcast(&total_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }

 public:
  DFSDCAUpdater(Model *model, Datapoint *datapoints) : Updater(model, datapoints) {
	alpha.zeros(datapoints->GetSize(), model->NumParameters());
  }

  virtual void EpochBegin() override {
	Updater::EpochBegin();
	CountSample();
  }

  ~DFSDCAUpdater() {
  }
};

#endif
