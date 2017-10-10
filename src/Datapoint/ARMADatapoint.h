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
 * labels:     n x c
 */

#ifndef _ARMA_DATAPOINT_
#define _ARMA_DATAPOINT_

#include <string.h>
#include "Datapoint.h"
#include "../defines.h"
#include "../Tools/Tools.h"

class ARMADatapoint : public Datapoint {
 private:
  sp_mat features;
  mat labels;

  void Initialize(const std::string &label_filename, const std::string &feature_filename) {
    features.load(feature_filename, arma_binary);
	sp_mat tmp_data;
	tmp_data.load(label_filename, arma_binary);
	labels = (mat)tmp_data;
	if (labels.min() > 0)
		labels -= labels.min();
  }

 public:
  ARMADatapoint(const std::string &data_dir, int taskid) : Datapoint(data_dir, taskid) {
    if(taskid != 0) {
	  std::string label_filename;
	  std::string feature_filename;
	  if(!FLAGS_distribute) {
        label_filename = data_dir + "labels.mat_" + std::to_string(taskid);	 
        feature_filename = data_dir + "features.mat_" + std::to_string(taskid);		
	  }
	  else {
        label_filename = data_dir + "labels.mat";	 
        feature_filename = data_dir + "features.mat";
	  }
	  Initialize(label_filename, feature_filename);
	}
	}

  virtual int GetSize() override {
	return labels.n_rows;
  }	

  virtual sp_mat GetFeaturesCols(int left, int right) override {
    return  features.cols(left, right);
  }

  virtual mat GetLabelsRows(int left, int right) override {
	return labels.rows(left, right);
  }

  virtual void OnehotEncoding(int num_class) override {
  	this->labels = one_hot_encoding(this->labels, num_class);
  }

  virtual mat GetVector() override {
  }

  virtual double GetLambda() override {
  }


  ARMADatapoint() {}
  ~ARMADatapoint() {} 
};

#endif
