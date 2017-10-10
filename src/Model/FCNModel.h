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

#ifndef _FCNMODEL_
#define _FCNMODEL_

#include <sstream>
#include <math.h>
#include "Model.h"
#include "../Layer/Layer.h"
#include "../Tools/Tools.h"

class FCNModel : public Model {
 private:
  int n_coords;
  std::vector<double> model;
  std::vector<int> dims;
  int postive=0;
  int negative=0;

  void Initialize() {
    // fully connected network
    n_coords = FLAGS_d1*FLAGS_d2 + FLAGS_d2*FLAGS_d3;

    // init dims.
	dims.push_back(FLAGS_d1);
	dims.push_back(FLAGS_d2);
	dims.push_back(FLAGS_d3);

    // Initialize model.
	InitWeight(model, dims);
    }

 public:
  FCNModel(int taskid) : Model(taskid) {
    Initialize();
  }

  // setup postive and negative numbers.
  void SetUp(Datapoint *datapoints) override {
  }

  double ComputeLoss(Datapoint *datapoints, double& accuracy) override {
    double loss = 0;
	int size = datapoints->GetSize();
	int begin=0;
	mat w_1 = vec_2_mat(model, begin, dims[0], dims[1]);
	begin += dims[0] * dims[1];
	mat w_2 = vec_2_mat(model, begin, dims[1], dims[2]);
	mat o_1(size, dims[1]);
	mat a_1(size, dims[1]);
	mat o_2(size, dims[2]);
	mat probs(size, dims[2]);

	affine_forward(datapoints->GetFeaturesCols(0, size-1).t(), w_1, o_1);
	relu_forward(o_1, a_1);
	affine_forward(a_1, w_2, o_2);
	loss = softmax_forward(o_2, datapoints->GetLabelsRows(0, size-1), probs);

	accuracy = EvaluateAccuracy(probs, datapoints->GetLabelsRows(0, size-1));

	return loss + ComputeRegularization(); 
  }

  virtual double ComputeRegularization() override {
	double regloss = 0;
	for(int i=0; i<model.size(); i++){
	  regloss += 0.5 * pow(model[i], 2) * FLAGS_l2_lambda;	
	}
	return regloss;
  }

  virtual void ProximalOperator(std::vector<double> &local_model, double gamma) override{
  }

  virtual int NumParameters() override {
	return n_coords;
  }

  std::vector<double> & ModelData() override {
	return model;
  }

  void PrecomputeCoefficients(Datapoint *datapoints, Gradient *g, std::vector<double> &local_model, const std::vector<int> &left_right ) override {
	// use layers 
	int size = left_right[1] - left_right[0];
	if (g->coeffs.size() != n_coords) g->coeffs.resize(n_coords);
	int begin=0;
	mat w_1 = vec_2_mat(local_model, begin, dims[0], dims[1]);
	begin += dims[0] * dims[1];
	mat w_2 = vec_2_mat(local_model, begin, dims[1], dims[2]);
	mat grad_1(dims[0], dims[1]);
    mat grad_2(dims[1], dims[2]);
	mat dx(size, dims[0]);
	mat o_1(size, dims[1]);
	mat a_1(size, dims[1]);
	mat dldo_1(size, dims[1]);
	mat dlda_1(size, dims[1]);
	mat o_2(size, dims[2]);
	mat dldo_2(size, dims[2]);
	mat probs(size, dims[2]);

	
	affine_forward(datapoints->GetFeaturesCols(left_right[0], left_right[1]-1).t(), w_1, o_1);
	relu_forward(o_1, a_1);
	affine_forward(a_1, w_2, o_2);
	softmax_forward(o_2, datapoints->GetLabelsRows(left_right[0], left_right[1]-1), probs);

	softmax_backward(o_2, datapoints->GetLabelsRows(left_right[0], left_right[1]-1), probs, dldo_2);
	affine_backward(a_1, w_2, dldo_2, dlda_1, grad_2);
	relu_backward(o_1, dlda_1, dldo_1);
	affine_backward(datapoints->GetFeaturesCols(left_right[0], left_right[1]-1).t(), w_1, dldo_1, dx, grad_1);

	g->coeffs = mat_2_vec(grad_1);
	std::vector<double> tmp;
	tmp = mat_2_vec(grad_2);
	g->coeffs.insert(g->coeffs.end(), tmp.begin(), tmp.end());
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

  ~FCNModel() {
  }
};

#endif
