#ifndef FORWARD_H
#define FORWARD_H

#define CL_TARGET_OPENCL_VERSION 220

#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif


typedef struct ForwardData{
    float ****X_list;
    float ****Y_list;
    float ***samples;
    float ***weights;
    float **biases;
    float *layer_sizes;
    float *activations;
    float *dropouts;
    int start_idx;
    int end_idx;
    int sample_rows;
    int sample_cols;
    int layer_sizes_rows;
    int layer_sizes_cols;
    int gpu;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
} ForwardData;

void forward(
    float **sample,
    int sample_rows,
    int sample_cols,
    float ***weights,
    float **biases,
    float ***Y,
    float *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols,
    float *activations,
    int is_predict_one_saple,
    int gpu,
    cl_context context,
    cl_command_queue queue,
    cl_program program
);

void *forward_worker(
    void *arg
);

void forward_threading(
    struct ForwardData forward_thread_data[],
    float ***samples,
    float ***weights,
    float **biases,
    float ****X_list,
    float ****Y_list,
    int dataset_samples_rows,
    int dataset_samples_cols,
    int dataset_samples_depth,
    float *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols,
    float *activations,
    float *dropouts,
    int num_threads,
    int gpu,
    cl_context context,
    cl_command_queue queue,
    cl_program program
);

#endif
