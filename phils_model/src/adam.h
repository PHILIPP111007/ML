#ifndef ADAM_H
#define ADAM_H


typedef struct AdamOptimizer {
    double ***m;    // Накопленный средний градиент
    double ***v;    // Накопленный квадратный градиент
    double lr;      // Скорость обучения
    double b1;      // Коэффициент для среднего градиента
    double b2;      // Коэффициент для квадрата градиента
    double eps;     // Маленькое значение для избежания деления на ноль
    int epoch;          // Текущая эпоха
} AdamOptimize;


void destroy_adam(
    struct AdamOptimizer *opt,
    double *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols
);

struct AdamOptimizer *create_adam(
    double lr,
    double b1,
    double b2,
    double eps,
    double *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols
);

void adam_step(
    struct AdamOptimizer *optimizer,
    double ***weights,
    double ***grads,
    double *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols
);

#endif
