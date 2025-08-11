#ifndef LINEAR_H
#define LINEAR_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include "src/functions.h"
#include "src/activations.h"
#include "src/loss.h"
#include "src/init.h"
#include "src/json.h"
#include "src/adam.h"
#include "src/forward.h"
#include "src/backward.h"
#include "src/logger.h"
#include "src/predict.h"


void linear_fit(
    float ***samples,
    int dataset_samples_rows,
    int dataset_samples_cols,
    int dataset_samples_depth,
    float **targets,
    int dataset_targets_cols,
    float ***weights,
    float **biases,
    float *losses,
    int loss,
    float *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols,
    float *activations,
    float *dropouts,
    int epoch,
    int n_epoch,
    int verbose,
    int regression,
    int max_change,
    float learning_rate,
    int num_threads,
    int gpu,
    struct AdamOptimizer *opt,
    cl_context context,
    cl_command_queue queue,
    cl_program program_matmul_gpu,
    cl_program program_adam_step_gpu
);

void linear_predict_one(
    float **sample,
    int sample_rows,
    int sample_cols,
    float *prediction,
    float ***weights,
    float **biases,
    float *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols,
    float *activations,
    int gpu,
    cl_context context,
    cl_command_queue queue,
    cl_program program,
    cl_mem weights_vec_buf
);

void linear_predict(
    float ***weights,
    float **biases,
    float ***samples,
    int dataset_samples_rows,
    int dataset_samples_cols,
    int dataset_samples_depth,
    float *predictions,
    float *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols,
    float *activations,
    int num_cpu,
    int gpu,
    cl_context context,
    cl_command_queue queue,
    cl_program program,
    cl_mem weights_vec_buf
);

#endif
