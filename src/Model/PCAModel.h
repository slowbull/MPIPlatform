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

#ifndef _PCAMODEL_
#define _PCAMODEL_

#include <sstream>
#include <math.h>
#include "Model.h"
#include "../Layer/Layer.h"
#include "../Tools/Tools.h"

class PCAModel : public Model {
 private:
  int n_coords;
  std::vector<double> model;
  std::vector<int> dims;
  mat est_vector;

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

  PCAModel(int taskid) : Model(taskid) {
    Initialize();
  }

  void SetUp(Datapoint *datapoints) override {
	est_vector = datapoints->GetVector();
  }

  double ComputeLoss(Datapoint *datapoints, double& auc) override {
    double loss = 0;
	auc = 0;
	int size = datapoints->GetSize();
	mat w = vec_2_mat(model, 0, dims[0], dims[1]);
	
	mat eye_mat(dims[0], dims[0], fill::eye);
	mat loss_mat = 0.5 * w.t() *(datapoints->GetLambda() * eye_mat - datapoints->GetFeaturesCols(0, size-1)
		   	* datapoints->GetFeaturesCols(0, size-1).t() / size) * w - est_vector.t() * w;
	loss = loss_mat[0];
	return loss; 
  }

  virtual double ComputeRegularization() override {
	double regloss = 0;
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

  void PrecomputeCoefficients(Datapoint *datapoints, Gradient *g, std::vector<double> &local_model, const std::vector<int> &left_right) override {
	// use layers 
	int size = left_right[1] - left_right[0];
	if (g->coeffs.size() != n_coords) g->coeffs.resize(n_coords);
	mat w = vec_2_mat(local_model, 0, dims[0], dims[1]);
	mat grad;

	mat eye_mat(dims[0], dims[0], fill::eye);
	grad = (datapoints->GetLambda() * eye_mat - datapoints->GetFeaturesCols(left_right[0], left_right[1]-1) 
			* datapoints->GetFeaturesCols(left_right[0], left_right[1]-1).t() / size )*w - est_vector;
	g->coeffs = mat_2_vec(grad);
  }

  // l2 norm.
  virtual void ComputeL2Gradient(Gradient *g, std::vector<double> &local_model) override {
  }	


  virtual void StoreModel(){
	std::ofstream out_file("model.out");
	for(int i=0; i< NumParameters(); i++){
	  out_file << model[i] << " ";
	}
	out_file.close();
  }

  ~PCAModel() {
  }
};

#endif
