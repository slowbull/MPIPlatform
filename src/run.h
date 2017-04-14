#include <iostream>
#include "defines.h"

template<class MODEL_CLASS, class DATAPOINT_CLASS, class CUSTOM_UPDATER=SGDUpdater>
TrainStatistics RunOnce(int taskid) {
    // Initialize model and datapoints.
  Model *model = new MODEL_CLASS(taskid);
  Datapoint *datapoints = new DATAPOINT_CLASS(FLAGS_data_file, taskid);
  model->SetUp(datapoints);

  Updater *updater = NULL;
  if (FLAGS_svrg) {
	if (!FLAGS_elastic_average) 
	  updater = new SVRGUpdater(model, datapoints);
	else
	  updater = new DisSVRGUpdater(model, datapoints);
  }
  else if (FLAGS_sgd) {
	updater = new SGDUpdater(model, datapoints);
  }
  else {
	updater = new CUSTOM_UPDATER(model, datapoints);
  }

  // Create trainer depending on flag.
  Trainer *trainer = NULL;
  if (!FLAGS_elastic_average) {
	  if (taskid == 0) {
	  	trainer = new ServerTrainer(model, datapoints);
	  }
	  else {
	  	trainer = new WorkerTrainer(model, datapoints);
	  }
  }
  else {
	  if (taskid == 0) {
	  	trainer = new DistServerTrainer(model, datapoints);
	  }
	  else {
	  	trainer = new DistWorkerTrainer(model, datapoints);
	  }
  }

  TrainStatistics stats = trainer->Train(model, datapoints, updater);

  // Delete trainer.
  delete trainer;

  // Delete model and datapoints.
  delete model;
  delete datapoints;

  // Delete updater.
  delete updater;

  return stats;
}

template<class MODEL_CLASS, class DATAPOINT_CLASS, class CUSTOM_UPDATER=SGDUpdater>
void Run(int taskid) {
  TrainStatistics stats = RunOnce<MODEL_CLASS, DATAPOINT_CLASS, CUSTOM_UPDATER>(taskid);
}
