#ifndef ADAM_H
#define ADAM_H


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

float fast_pow(
    float a,
    int b
);

float fast_sqrt(
    float x
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

#endif
