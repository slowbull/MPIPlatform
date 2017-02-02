# Platform for distributed optimization expriments using OpenMPI

## Introduction
This project is a platform for distributed optimization expriments using OpenMPI. This platform is used to implement the expeirments of my recent papers [1][2]. Please cite them if this code helps you. :)

[1] Zhouyuan Huo , Heng Huang, Asynchronous Stochastic Gradient Descent with Variance Reduction for Non-Convex Optimization . AAAI, 2017

[2]Zhouyuan Huo , Bin Gu, Heng Huang, Decoupled Asynchronous Proximal Stochastic Gradient Descent with Variance Reduction . arXiv


## Installation
1. Clone this repository from github.   
2. run   ```./install```

 **Remark:** ```./install``` is only used for first time installation. After you modify the code and want to compile, just run ```make``` in ```build/``` directory. 
 
 
## How to use on localhost
### Run linear model.
1. **Generate distributed datasets**.  Go to build directory, and run ```./splitdata ../data/covtype_binary 4 0```. It will generate 4 data files in ```data/covtype_binary_split/``` directory. ```./splitdata``` will transform libsvm format data to armadillo format data, and distributes it to multiple files. run  ```./splitdata``` directly to see how to use it.
 
2. **Run**  To run do 
```
mpirun -np 5 ./mpiplatform -logistic_l2_l1 -num_workers=4 -data_file="absolutepathto/data/covtype_binary_split/" -print_loss_per_epoch -d1=54 -learning_rate=1e-1 -n_epochs=100 -mini_batch=100 -in_iters=1000 -svrg -max_delay=10
```

### Run fully connected nueral network model.
 
1. **Generate distributed datasets**.  Go to build directory, and run ```./splitdata ../data/covtype_multiclass 4 0```. It will generate 4 data files in ```data/covtype_multiclass_split/``` directory. 
  
2. **Run**  To run do 
```
mpirun -np 5 ./mpiplatform -fcn -num_workers=4 -data_file="absolutepathto/data/covtype_multiclass_split/" -print_loss_per_epoch -d1=54 -d2=20 -d3=7 -learning_rate=1e-3 -n_epochs=100 -mini_batch=10 -in_iters=1000 -svrg -max_delay=10
```

# How to use on AWS.

## Disclaimer
This repository uses code from [Cyclades](https://github.com/amplab/cyclades), we borrow the framework from this project.  And we use code from [LibSVM](https://github.com/cjlin1/libsvm) to read libsvm data and  transform it to distributed armadillo format.

