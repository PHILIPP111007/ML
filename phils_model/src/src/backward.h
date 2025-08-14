#ifndef BACKWARD_H
#define BACKWARD_H

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>


typedef struct BackwardData {
    float ***weights;
    float ****X_list;
    float ****Y_list;
    float ****grad_w_list;
    float ****grad_x_list;
    float ***grad_b_list;
    float **targets;
    float *layer_sizes;
    float *activations;
    float *epoch_losses;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_mem weights_transposed_vec_buf;
    int start_idx;
    int end_idx;
    int layer_sizes_rows;
    int layer_sizes_cols;
    int matrix_rows;
    int loss;
    int regression;
    int dataset_samples_rows;
    int dataset_samples_cols;
    int dataset_targets_cols;
    int gpu;
    int large_matrices;
} BackwardData;

void *backward_worker(
    void *arg
);

void backward_threading(
    struct BackwardData backward_thread_data[],
    float ***weights,
    float **targets,
    float **biases,
    float ****X_list,
    float ****Y_list,
    float ****grad_w_list,
    float ****grad_x_list,
    float ***grad_b_list,
    float *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols,
    int dataset_samples_rows,
    int dataset_samples_cols,
    int dataset_targets_cols,
    int matrix_rows,
    float *activations,
    int loss,
    float *epoch_losses,
    int regression,
    int num_threads,
    int gpu,
    int large_matrices,
    cl_context context,
    cl_command_queue queue,
    cl_program program,
    cl_mem weights_transposed_vec_buf
);

#endif
