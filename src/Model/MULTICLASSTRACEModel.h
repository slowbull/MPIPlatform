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

#ifndef _MULTICLASSTRACEMODEL_
#define _MULTICLASSTRACEMODEL_

#include <sstream>
#include <math.h>
#include "Model.h"
#include "../Layer/Layer.h"
#include "../Tools/Tools.h"

class MULTICLASSTRACEModel : public Model {
 private:
  int n_coords;
  std::vector<double> model;
  std::vector<int> dims;
  int postive=0;
  int negative=0;

  void Initialize() {
    // 1 layer fully connected network
    n_coords = FLAGS_d1*FLAGS_d2;

    // init dims.
	dims.push_back(FLAGS_d1);
	dims.push_back(FLAGS_d2);

    // Initialize model.
	InitWeight(model, dims);
    }

 public:
  MULTICLASSTRACEModel(int taskid) : Model(taskid) {
    Initialize();
  }

  // setup postive and negative numbers.
  void SetUp(Datapoint *datapoints) override {
  }

  double ComputeLoss(Datapoint *datapoints, double& accuracy) override {
    double loss = 0;
	int size = datapoints->GetSize();
	mat w_1 = vec_2_mat(model, 0, dims[0], dims[1]);
	mat o_1(size, dims[1]);


	affine_forward(datapoints->GetFeaturesCols(0, size-1).t(), w_1, o_1);
	loss = logistic_forward(o_1, datapoints->GetLabelsRows(0, size-1));

	accuracy = metric_acc_logistic(o_1, datapoints->GetLabelsRows(0, size-1));

	return loss + ComputeRegularization(); 
  }

  virtual double ComputeRegularization() override {
	double regloss = 0;
	// l2 norm
	for(int i=0; i<model.size(); i++){
	  regloss += 0.5 * pow(model[i], 2) * FLAGS_l2_lambda;	
	}
	// nuclear norm
	mat w_1 = vec_2_mat(model, 0, dims[0], dims[1]); 
	mat U, V;
	vec s;
	svd(U, s, V, w_1);
	regloss += accu(s) * FLAGS_trace_lambda;

	return regloss;
  }

  virtual void ProximalOperator(std::vector<double> &local_model, double gamma) override{
	// trace norm operator.
	mat w_1 = vec_2_mat(local_model, 0, dims[0], dims[1]); 
	mat U, V;
	vec s;
	svd(U, s, V, w_1);
	mat tmp_s(w_1);
	tmp_s.zeros();
	for(size_t i=0; i < s.n_elem; i++){
		if(s[i] - gamma > 0){
			tmp_s(i, i) = s[i] - gamma;
		}
		else{
			tmp_s(i, i) = 0;
		}
	}

	w_1 = U * tmp_s * V.t();

	local_model	= mat_2_vec(w_1);
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
	mat w_1 = vec_2_mat(local_model, 0, dims[0], dims[1]);
	mat grad_1(dims[0], dims[1]);
	mat dx(size, dims[0]);
	mat o_1(size, dims[1]);
	mat dldo_1(size, dims[1]);

	// forward
	affine_forward(datapoints->GetFeaturesCols(0, size-1).t(), w_1, o_1);

	// backward
	logistic_backward(o_1, datapoints->GetLabelsRows(0, size-1), dldo_1);
	affine_backward(datapoints->GetFeaturesCols(0, size-1).t(), w_1, dldo_1, dx, grad_1);

	g->coeffs = mat_2_vec(grad_1);
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

  ~MULTICLASSTRACEModel() {
  }
};

#endif
