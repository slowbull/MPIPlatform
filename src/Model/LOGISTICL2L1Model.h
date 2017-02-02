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

#ifndef _LOGISTICL2L1MODEL_
#define _LOGISTICL2L1MODEL_

#include <sstream>
#include <math.h>
#include "Model.h"
#include "../Layer/Layer.h"
#include "../Tools/Tools.h"

class LOGISTICL2L1Model : public Model {
 private:
  int n_coords;
  std::vector<double> model;
  std::vector<int> dims;
  int postive=0;
  int negative=0;

  void Initialize() {
    // linear model
    n_coords = FLAGS_d1;

    // init dims.
	dims.push_back(FLAGS_d1);
	dims.push_back(1);

    // Initialize model.
    model.resize(n_coords, 0);
    }
 public:

  LOGISTICL2L1Model(int taskid) : Model(taskid) {
    Initialize();
  }

  // setup postive and negative numbers.
  void SetUp(Datapoint *datapoints) override {
    for(int i=0; i<datapoints->GetSize(); i++){
	  if(datapoints->GetLabelsRows(i,i)[0] == 1){
	    postive += 1;
	  }
	  else{
	    negative += 1;
	  }
    }
  }


  double ComputeLoss(Datapoint *datapoints, double& auc) override {
    double loss = 0;
	int size = datapoints->GetSize();
	mat w = vec_2_mat(model, 0, dims[0], dims[1]);
	mat o(size, dims[1]);

	affine_forward(datapoints->GetFeaturesCols(0, size-1).t(), w, o);
	loss = logistic_forward(o, datapoints->GetLabelsRows(0, size-1));

	std::vector<double> probs = mat_2_vec(o);
	std::vector<double> labels = mat_2_vec(datapoints->GetLabelsRows(0,size-1));
	auc = EvaluateAUC(labels, probs, postive, negative);

	return loss + ComputeRegularization(); 
  }

  virtual double ComputeRegularization() override {
	double regloss = 0;
	for(int i=0; i<model.size(); i++){
	  regloss += std::abs(model[i]) * FLAGS_l1_lambda + 0.5 * pow(model[i], 2) * FLAGS_l2_lambda;	
	}
	return regloss;
  }

  virtual void ProximalOperator(std::vector<double> &local_model, double gamma) override{
    for (int i=0; i<local_model.size(); i++) { 
	  double val = local_model[i];
	  double sign = val > 0 ? 1: -1;	
	  local_model[i] = sign * fmax( std::abs(val) - gamma, 0);
	}
  }

  virtual int NumParameters() override {
	return n_coords;
  }

  std::vector<double> & ModelData() override {
	return model;
  }

  void PrecomputeCoefficients(Datapoint *datapoints, Gradient *g, std::vector<double> &local_model) override {
	// use layers 
	int size = datapoints->GetSize();
	if (g->coeffs.size() != n_coords) g->coeffs.resize(n_coords);
	mat w = vec_2_mat(local_model, 0, dims[0], dims[1]);
	sp_mat grad(dims[0], dims[1]);
	mat o(size, dims[1]);
	mat dldo(size, dims[1]);
	mat dx(size, dims[0]);
	
	affine_forward(datapoints->GetFeaturesCols(0, size-1).t(), w, o);
	logistic_backward(o, datapoints->GetLabelsRows(0, size-1), dldo);
	affine_backward(datapoints->GetFeaturesCols(0, size-1).t(), w, dldo, dx, grad);

	g->coeffs = mat_2_vec(grad);
  }

  // l2 norm.
  virtual void ComputeL2Gradient(Gradient *g, std::vector<double> &local_model) override {
	for (int i=0;  i<NumParameters(); i++) {
	  g->coeffs[i] += FLAGS_l2_lambda * local_model[i];
	}
  }	


  virtual void StoreModel(){
	std::ofstream out_file("model.out");
	for(int i=0; i< NumParameters(); i++){
	  out_file << model[i] << " ";
	}
	out_file.close();
  }

  ~LOGISTICL2L1Model() {
  }
};

#endif
