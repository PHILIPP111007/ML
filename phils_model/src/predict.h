#ifndef PREDICT_H
#define PREDICT_H

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>


typedef struct PredictTask {
    float **sample;
    float ***weights;
    float **biases;
    float *layer_sizes;
    float *activations;
    float *predictions;
    int layer_sizes_rows;
    int layer_sizes_cols;
    int dataset_index;
    int dataset_samples_cols;
    int dataset_samples_depth;
    int n_neurons_last_layer;
    int gpu;
} PredictTask;

typedef struct ThreadRange {
    PredictTask *tasks;
    int start;
    int end;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
} ThreadRange;

void *predict_thread(
    void *arg
);

#endif
