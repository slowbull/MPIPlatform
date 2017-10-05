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

#ifndef _LAYERS_
#define _LAYERS_

#include "../defines.h"

// least square loss
double least_forward(const mat& a, const mat& y){
  int dim1 = a.n_rows;
  double loss = 0;
  mat tmp(a);
	
  tmp = pow(a-y,2);
  loss = 0.5 * accu(tmp) / dim1;

  return loss;
}

// least square loss backward
void least_backward(const mat& a, const mat&  y, mat &grad){
  int dim1 = a.n_rows;

  grad = (a - y) / dim1;
}


double l2hingeloss_forward(const mat& a, const mat& y){
  int dim1 = a.n_rows;
  double loss = 0;
  mat tmp = 1 - a%y;
	
  loss = accu(pow((tmp%(tmp>0)),2)) / dim1;

  return loss;
}

void l2hingeloss_backward(const mat& a, const mat&  y, mat &grad){
  int dim1 = a.n_rows;
  mat tmp = 1 - a%y;

  grad =  (-y%tmp%(tmp>0)) / dim1 * 2;
}


double l1hingeloss_forward(const mat& a, const mat& y){
  int dim1 = a.n_rows;
  double loss = 0;
  mat tmp = 1 - a%y;
	
  loss = accu(tmp%(tmp>0)) / dim1;

  return loss;
}

void l1hingeloss_backward(const mat& a, const mat&  y, mat &grad){
  int dim1 = a.n_rows;
  mat tmp = 1 - a%y;

  grad = -y % (tmp>0) / dim1;
}


// softmax loss
double softmax_forward(const mat&  a, const mat&  y, mat &probs){
  int dim1 = a.n_rows;
  int dim2 = a.n_cols;
  double loss = 0;
	
  probs = exp(a);
  mat sums = repmat(sum(probs, 1), 1, dim2);
  probs = probs / sums;

  //compute loss
  int label = 0;
  for(int i=0; i<dim1; i++){
    label = (int)y(i,0);	
	loss += -log(probs(i,label))/dim1;
  }

  return loss;
}

void softmax_backward(const mat& a, const mat&  y, const mat&  probs, mat &grad){
  int dim1 = a.n_rows;
  int label = 0;
  grad = probs;

  for(int i=0; i<dim1; i++){
    label = y(i,0);
	grad(i,label) = grad(i,label)-1;
  }
  grad = grad / dim1;
}


void sigmoid_forward(const mat&  o, mat  &a){
  a = pow(1+ exp(-o),-1);
}


void sigmoid_backward(const mat& o, const mat& dl, mat &grad){
  grad =   dl % o % (1-o);
}

void relu_forward(const mat& o, mat &a){
  a = o;
  a.for_each([](mat::elem_type& val){ val=(val>0?val:0);});
}

void relu_backward(const mat& o, const mat& dl, mat &grad){
  grad = o;
  grad.for_each([](mat::elem_type& val){ val=(val>0?1:0);});
  grad = dl % grad;
}


double logistic_forward(const mat& a, const mat& y0){
  mat tmp = -a % y0;
  double loss;
  tmp = log(1+exp(tmp));
  loss = accu(tmp) / a.n_rows;
  return loss;
}

void logistic_backward(const mat& a, const mat&  y, mat &grad){
  int dim1 = a.n_rows;
  mat tmp = -a % y;
  grad = -pow(1+exp(tmp),-1) % exp(tmp) % y / dim1;
}

template<typename T>
void affine_forward(const T& x, const mat& w, mat &o){
  o = x*w;
}

template<typename T1, typename T2>
void affine_backward(const T1& x, const mat& w, const mat& dl,\
			   	    mat &dx, T2 &grad){
  grad = x.t()*dl;
  dx = dl * w.t();
}


#endif
