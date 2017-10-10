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

#ifndef _DATAPOINT_
#define _DATAPOINT_

#include<armadillo>

using namespace arma;

class Datapoint {
 private:
  int taskid;
 public:
  Datapoint() {}
  Datapoint(const std::string &input_line, int taskid) {
    this->taskid = taskid;
  }
  virtual ~Datapoint() {}

  // Get the taskid of this datapoint (equivalent to id).
  virtual int GetOrder() {
    return taskid;
  }

  virtual sp_mat GetFeaturesCols(int left, int right) = 0;

  virtual mat GetLabelsRows(int left, int right) = 0;

  virtual int  GetSize() = 0;

  virtual void OnehotEncoding(int num_class) = 0;

  virtual mat GetVector() = 0;

  virtual double GetLambda() = 0;

};

#endif
