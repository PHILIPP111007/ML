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

    float b1_minus_1;
    float b2_minus_1;
    float b1_pow;
    float b2_pow;
    float inv_1mb1;
    float inv_1mb2;
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
    int layer_sizes_cols
);

#endif
