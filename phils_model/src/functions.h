#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>


void matmul(
    float **A,
    float **B,
    float **C,
    int rows_A,
    int cols_A,
    int rows_B,
    int cols_B
);

void matmul_gpu(
    cl_context context,
    cl_command_queue queue,
    cl_program program,
    float *A,
    cl_mem d_B,
    float *C,
    int ROWS_A,
    int COLS_A,
    int ROWS_B,
    int COLS_B,
    int layer_index
);

float **transpose(
    float **original_matrix,
    int rows,
    int cols
);

float sum(
    float **matrix,
    int rows,
    int cols
);

float *sum_axis_0(
    float **matrix,
    int rows,
    int cols
);

float mean(
    float *arr,
    int len
);

int argmax(
    float *arr,
    int size
);

float safe_update(
    float number,
    float max_change
);

void apply_dropout(
    float **y,
    int matrix_rows,
    int n_neurons,
    float dropout
);

float **create_matrix(
    int rows,
    int cols
);

void free_matrix(
    float **matrix
);

float *get_weights_vec(
    float ***weights,
    int layer_sizes_rows,
    int layer_sizes_cols,
    float *layer_sizes
);

float *get_weights_transposed_vec(
    float ***weights_transposed,
    int layer_sizes_rows,
    int layer_sizes_cols,
    float *layer_sizes
);

cl_mem get_weights_vec_buf(
    float *weights_vec,
    int layer_sizes_rows,
    int layer_sizes_cols,
    float *layer_sizes,
    cl_context context
);

#endif
