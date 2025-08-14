#ifndef ADAM_H
#define ADAM_H

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>


typedef struct AdamOptimizer {
    float ***m;    // Накопленный средний градиент
    float ***v;    // Накопленный квадратный градиент
    float lr;      // Скорость обучения
    float b1;      // Коэффициент для среднего градиента
    float b2;      // Коэффициент для квадрата градиента
    float eps;     // Маленькое значение для избежания деления на ноль
    int epoch;          // Текущая эпоха
} AdamOptimize;


void destroy_adam(
    struct AdamOptimizer *opt,
    float *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols
);

struct AdamOptimizer *create_adam(
    float lr,
    float b1,
    float b2,
    float eps,
    float *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols
);

void adam_step(
    struct AdamOptimizer *optimizer,
    float ***weights,
    float ***grads,
    float *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols,
    float max_change
);

void adam_step_gpu(
    struct AdamOptimizer *optimizer,
    float ***weights,
    float *weights_vec,
    cl_mem weights_vec_buf,
    float ****grad_w_list,
    int dataset_samples_rows,
    float *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols,
    float max_change,
    cl_context context,
    cl_command_queue queue,
    cl_program program
);


#endif
