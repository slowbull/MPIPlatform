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

/*
 * file should be armadillo data saved in binary format.
 * features:   d x n  
 */

#ifndef _PCA_DATAPOINT_
#define _PCA_DATAPOINT_

#include <string.h>
#include "Datapoint.h"
#include "../defines.h"
#include "../Tools/Tools.h"

class PCADatapoint : public Datapoint {
 private:
  sp_mat features;
  mat est_vector;
  double lambda; // estimated largest singular value.

  void Initialize(const std::string &est_vector_filename, const std::string &feature_filename) {
    features.load(feature_filename, arma_binary);
	est_vector.load(est_vector_filename, arma_binary);
	lambda = FLAGS_lambda; // let lambda = 100 for now.
  }

 public:
  PCADatapoint(const std::string &data_dir, int taskid) : Datapoint(data_dir, taskid) {
    if(taskid != 0) {
	  std::string est_vector_filename;
	  std::string feature_filename;
	  if(!FLAGS_distribute) {
        est_vector_filename = data_dir + "est_vector.mat_" + std::to_string(taskid);	 
        feature_filename = data_dir + "features.mat_" + std::to_string(taskid);		
	  }
	  else {
        est_vector_filename = data_dir + "est_vector.mat";	 
        feature_filename = data_dir + "features.mat";
	  }
	  Initialize(est_vector_filename, feature_filename);
	}
	}

  virtual int GetSize() override {
	return features.n_cols;
  }	

  virtual sp_mat GetFeaturesCols(int left, int right) override {
    return  features.cols(left, right);
  }

  virtual mat GetLabelsRows(int left, int right) override {
  }

  virtual void OnehotEncoding(int num_class) override {
  }

  virtual mat GetVector() override {
  	return est_vector;
  }

  virtual double GetLambda() override {
  	return lambda;
  }

  PCADatapoint() {}
  ~PCADatapoint() {} 
};

#endif
