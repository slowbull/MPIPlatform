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

#ifndef _MODEL_
#define _MODEL_

class Model {
 public:
  int taskid;
  Model() {}
  Model(int taskid) { 
    this->taskid = taskid;
  }
  virtual ~Model() {}

  // Computes loss on the model
  virtual double ComputeLoss(Datapoint *datapoints, double &evaluation) = 0;

  // Computes regularizatoin on the model
  virtual double ComputeRegularization() = 0;

  // Do some set up with the model and datapoints before running gradient descent.
  virtual void SetUp(Datapoint *datapoints) {}

  // Return data to actual model.
  virtual std::vector<double> & ModelData() = 0;

  // return closed form solution for proximal operator.
  virtual void ProximalOperator(std::vector<double> & val, double gamma) = 0;

  virtual void StoreModel(){}

  virtual int NumParameters() = 0;

  // The following are for updates of the form:
  virtual void PrecomputeCoefficients(Datapoint *datapoint, Gradient *g, std::vector<double> &local_model) = 0;

  virtual void ComputeL2Gradient(Gradient *g, std::vector<double> &local_model) = 0;
};

#endif
