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

#ifndef _TOOLS_
#define _TOOLS_

#include<cstdlib>
#include<cstdio>
#include<cmath> 
#include<algorithm>
#include<armadillo>
#include "../Layer/Layer.h"

using namespace arma;

/*
// for different layers nn, we have to edit this file.
template<typename T>
std::vector<double> mat_2_vec(const T& Matrix){
  int row = Matrix.n_rows;
  int col = Matrix.n_cols;
  std::vector<double> out(row*col, 0);
  for(int i=0; i<row; i++)
	for(int j=0; j<col; j++)
	  out[i*col+j] = Matrix(i,j); 

  return out;
}

mat vec_2_mat(const std::vector<double>& w, int begin, int row, int col){
  mat out(row,col); int index = 0;
	for(int i=0; i<row; i++)
	  for(int j=0; j<col; j++){ 
		index = i*col + j; 
		out(i,j) = w[begin+index];
	  }
  return out;
}
*/

template<typename T> 
std::vector<double> mat_2_vec(const T& Matrix){
	std::vector<double> out;
	if (Matrix.n_rows==1 || Matrix.n_cols==1)
		 out = arma::conv_to<std::vector<double>>::from((mat)Matrix);
	else{
		mat MatrixT = (mat)Matrix.t();
		for (size_t i=0; i<MatrixT.n_cols; i++){
			std::vector<double> sub_w = arma::conv_to<std::vector<double>>::from(MatrixT.col(i));
			out.insert(out.end(), sub_w.begin(), sub_w.end());
		}
	}
	return out;
}

mat vec_2_mat(const std::vector<double> &w, int begin, int row, int col){
	int index = 0;
	std::vector<double> sub_w(w.begin()+begin, w.begin()+row*col+begin);
	mat out = arma::conv_to<mat>::from(sub_w);
	out = reshape(out, col, row);
	return out.t();
}


template<typename T>
T max_element(std::vector<T> vec) {
  T max_val = vec[0];
	for(int i = 1; i < vec.size(); i++) {
	  if (max_val < vec[i])	
		max_val = vec[i];
	}
  return max_val;
}

template<typename T>
std::vector<int> sort_indexes(const std::vector<T> & v){
  std::vector<int> idx(v.size());
  iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&v](int i1, int i2){return v[i1] < v[i2];});
  return idx;
}

double EvaluateAUC(const std::vector<double> & labels, std::vector<double> probability, int num_postive, int num_negative) {
  std::vector<int> idx = sort_indexes(probability);	
  int size = probability.size();
  double sum = 0;
  for(int i = 0; i < size; i++){
    double element = (labels[idx[i]] + 1) / 2;
	sum += element;
	probability[i] =  sum;
  }
  double auc = 0;
  double y2 = 1, y1 = 0, x2 = 1, x1 = 0;
  for(int i=0; i < size; i++){
	y1 = (num_postive - probability[i]) / num_postive;
	x1 = 1 - ( i + 1 - probability[i]) / num_negative;
	auc += 0.5 * (y2+y1)*(x2-x1);
	x2 = x1;
	y2 = y1;
  }
  return auc;
}

// compute accuracy 
double EvaluateAccuracy(const mat& probs, const mat& y){
  int num_sample = probs.n_rows;
  int num_class = probs.n_cols;

  uvec pred_y = index_max(probs, 1);
  double count = sum(pred_y == y);

  return count / num_sample;
}

// multiclass logistic accuracy.
double metric_acc_logistic(const mat& o, const mat& y){
  int num_sample = o.n_rows;
  uvec pred_y = index_max(o, 1);
  uvec truth_y = index_max(y, 1);

  double count = sum(pred_y == truth_y);
 
  return count / num_sample;
}


// vector to onehot encoding
mat one_hot_encoding(const mat& y, int num_class){
  mat onehot_y(y.n_rows, num_class);
  for(size_t i = 0; i < y.n_rows; i++){
	int label = y(i);
	onehot_y.row(i) = -1 * ones(1, num_class);
	onehot_y(i, label) = 1;
  }
  return  onehot_y;
}


// if seed !=0 initialize with seed random value. if flag == 0 initialize with  0.
void InitWeight(std::vector<double> &w, const std::vector<int> dims){
  w.resize(0);
  for (int k = 0; k < dims.size() - 1; k++) {
	srand(k);
	mat tmp(dims[k], dims[k+1]);
	for(int i = 0; i < dims[k]; i++)
	  for(int j = 0; j < dims[k+1]; j++)
		tmp(i, j) = (1.0*rand()/RAND_MAX - 0.5) * 2 / dims[k];
	std::vector<double> vec = mat_2_vec(tmp);	
	w.insert(w.end(), vec.begin(), vec.end());
  }
}
#endif
