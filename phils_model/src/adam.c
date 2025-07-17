#include <stdlib.h>
#include <math.h>
#include "adam.h"

///////////////////////////////////////////////////////////////////////////////
// Adam
///////////////////////////////////////////////////////////////////////////////

typedef struct {
    double lr;      // Скорость обучения
    double b1;      // Коэффициент для среднего градиента
    double b2;      // Коэффициент для квадрата градиента
    double eps;     // Маленькое значение для избежания деления на ноль
    int epoch;          // Текущая эпоха
    double ***m;    // Накопленный средний градиент
    double ***v;    // Накопленный квадратный градиент
} AdamOptimizer;


// Освобождение ресурсов оптимизатора
void destroy_adam(struct AdamOptimizer *opt, double *layer_sizes, int layer_sizes_rows, int layer_sizes_cols) {
    for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {

        double n_inputs_double = layer_sizes[layer_index * layer_sizes_cols + 0];
        int n_inputs = (int)n_inputs_double;

        for (int i = 0; i < n_inputs; i++) {
            free(opt->m[layer_index][i]);
            free(opt->v[layer_index][i]);
        }
        free(opt->m[layer_index]);
        free(opt->v[layer_index]);
    }
    free(opt);
}

// Создание нового экземпляра оптимизатора
struct AdamOptimizer *create_adam(double lr, double b1, double b2, double eps, double *layer_sizes, int layer_sizes_rows, int layer_sizes_cols) {
    AdamOptimizer *optimizer = malloc(sizeof(AdamOptimizer));
    optimizer->lr = lr;
    optimizer->b1 = b1;
    optimizer->b2 = b2;
    optimizer->eps = eps;
    optimizer->epoch = 0;

    double ***m = malloc(layer_sizes_rows * sizeof(double**));
    double ***v = malloc(layer_sizes_rows * sizeof(double**));

    for (int layer_index = 0; layer_index < layer_sizes_rows; ++layer_index) {
        double n_inputs_double = layer_sizes[layer_index * layer_sizes_cols + 0];
        double n_neurons_double = layer_sizes[layer_index * layer_sizes_cols + 1];
        int n_inputs = (int)n_inputs_double;
        int n_neurons = (int)n_neurons_double;

        m[layer_index] =  malloc(n_inputs * sizeof(double*));
        v[layer_index] =  malloc(n_inputs * sizeof(double*));

        for (int i = 0; i < n_inputs; ++i) {
            m[layer_index][i] =  malloc(n_neurons * sizeof(double*));
            v[layer_index][i] =  malloc(n_neurons * sizeof(double*));
        }
    }
    optimizer->m = m;
    optimizer->v = v;

    return optimizer;
}

void adam_update(struct AdamOptimizer *optimizer, double ***weights, double ***grads, double *layer_sizes, int layer_sizes_rows, int layer_sizes_cols) {
    optimizer->epoch++;

    for(int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
        double n_inputs_double = layer_sizes[layer_index * layer_sizes_cols + 0];
        double n_neurons_double = layer_sizes[layer_index * layer_sizes_cols + 1];
        int n_inputs = (int)n_inputs_double;
        int n_neurons = (int)n_neurons_double;



        for (int i = 0; i < n_inputs; i++) {
            for (int j = 0; j < n_neurons; j++) {
                // Обновляем первый момент (среднее градиентов)
                optimizer->m[layer_index][i][j] = optimizer->b1 * optimizer->m[layer_index][i][j] + (1 - optimizer->b1) * grads[layer_index][i][j];
                
                // Обновляем второй момент (средний квадрат градиентов)
                optimizer->v[layer_index][i][j] = optimizer->b2 * optimizer->v[layer_index][i][j] + (1 - optimizer->b2) * pow(grads[layer_index][i][j], 2);
                
                // Исправляем смещение
                double m_hat = optimizer->m[layer_index][i][j] / (1 - pow(optimizer->b1, optimizer->epoch));
                double v_hat = optimizer->v[layer_index][i][j] / (1 - pow(optimizer->b2, optimizer->epoch));
                
                // Обновляем веса
                weights[layer_index][i][j] -= optimizer->lr * m_hat / (sqrt(v_hat) + optimizer->eps);

                if (isnan(weights[layer_index][i][j])) {
                    weights[layer_index][i][j] = 0.0;
                }
            }
        }
    }
}
