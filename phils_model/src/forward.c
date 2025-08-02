#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "forward.h"
#include "functions.h"
#include "loss.h"
#include "activations.h"


// Forward pass
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
    float *activations) {

    float n_inputs_float = layer_sizes[0 * layer_sizes_cols + 0];
    float n_neurons_float = layer_sizes[0 * layer_sizes_cols + 1];
    int n_inputs = (int)n_inputs_float;
    int n_neurons = (int)n_neurons_float;

    float **y = create_matrix(sample_rows, n_neurons);

    matmul(sample, weights[0], y, sample_rows, sample_cols, n_inputs, n_neurons);

    for (int i = 0; i < sample_rows; ++i) {
        for (int j = 0; j < n_neurons; ++j) {
            y[i][j] += biases[0][i];
        }
    }
    int activation = (int)activations[0];
    apply_activation_calc(y, sample_rows, n_neurons, activation);

    for (int i = 0; i < sample_rows; i++) {
        free(sample[i]);
    }
    free(sample);

    Y[0] = create_matrix(sample_rows, n_neurons);
    for (int i = 0; i < sample_rows; i++) {
        for (int j = 0; j < n_neurons; j++) {
            Y[0][i][j] = y[i][j];
        }
    }
    for (int i = 0; i < sample_rows; i++) {
        free(y[i]);
    }
    free(y);

    int matrix_rows = sample_rows;
    for (int layer_index = 1; layer_index < layer_sizes_rows; layer_index++) {
        float n_inputs_float = layer_sizes[layer_index * layer_sizes_cols + 0];
        float n_neurons_float = layer_sizes[layer_index * layer_sizes_cols + 1];
        int n_inputs = (int)n_inputs_float;
        int n_neurons = (int)n_neurons_float;

        float **x = create_matrix(matrix_rows, n_inputs);
        for (int i = 0; i < matrix_rows; i++) {
            for (int j = 0; j < n_inputs; j++) {
                x[i][j] = Y[layer_index - 1][i][j];
            }
        }

        float **y = create_matrix(matrix_rows, n_neurons);
        matmul(x, weights[layer_index], y, matrix_rows, n_inputs, n_inputs, n_neurons);

        for (int i = 0; i < matrix_rows; i++) {
            free(x[i]);
        }
        free(x);

        for (int i = 0; i < matrix_rows; ++i) {
            for (int j = 0; j < n_neurons; ++j) {
                y[i][j] += biases[layer_index][i];
            }
        }
        int activation = (int)activations[layer_index];
        apply_activation_calc(y, matrix_rows, n_neurons, activation);

        Y[layer_index] = create_matrix(matrix_rows, n_neurons);
        for (int i = 0; i < matrix_rows; i++) {
            for (int j = 0; j < n_neurons; j++) {
                Y[layer_index][i][j] = y[i][j];
            }
        }
        for (int i = 0; i < matrix_rows; i++) {
            free(y[i]);
        }
        free(y);
    }
}

// Forward pass
void *forward_worker(void *arg) {
    ForwardData *fd = (ForwardData *)arg;

    int start_idx = fd->start_idx;
    int end_idx = fd->end_idx;
    float ***samples = fd->samples;
    int sample_rows = fd->sample_rows;
    int sample_cols = fd->sample_cols;
    float ****X_list = fd->X_list;
    float ****Y_list = fd->Y_list;
    float ***weights = fd->weights;
    float **biases = fd->biases;
    float *layer_sizes = fd->layer_sizes;
    int layer_sizes_rows = fd->layer_sizes_rows;
    int layer_sizes_cols = fd->layer_sizes_cols;
    float *activations = fd->activations;
    float *dropouts = fd->dropouts;

    for (int dataset_index = start_idx; dataset_index < end_idx; ++dataset_index) {

        float **sample = malloc(sample_rows * sizeof(float*));
        for (int i = 0; i < sample_rows; i++) {
            sample[i] = malloc(sample_cols * sizeof(float));
            for (int j = 0; j < sample_cols; j++) {
                sample[i][j] = samples[dataset_index][i][j];
            }
        }

        float ***X = malloc(layer_sizes_rows * sizeof(float**));
        float ***Y = malloc(layer_sizes_rows * sizeof(float**));

        float n_inputs_float = layer_sizes[0 * layer_sizes_cols + 0];
        float n_neurons_float = layer_sizes[0 * layer_sizes_cols + 1];
        int n_inputs = (int)n_inputs_float;
        int n_neurons = (int)n_neurons_float;

        float **y = create_matrix(sample_rows, n_neurons);
        matmul(sample, weights[0], y, sample_rows, sample_cols, n_inputs, n_neurons);

        for (int i = 0; i < sample_rows; ++i) {
            for (int j = 0; j < n_neurons; ++j) {
                y[i][j] += biases[0][i];
            }
        }
        int activation = (int)activations[0];
        apply_activation_calc(y, sample_rows, n_neurons, activation);
        float dropout = dropouts[0];
        apply_dropout(y, sample_rows, n_neurons, dropout);

        X[0] = create_matrix(sample_rows, sample_cols);
        for (int i = 0; i < sample_rows; i++) {
            for (int j = 0; j < sample_cols; j++) {
                X[0][i][j] = sample[i][j];
            }
        }

        for (int i = 0; i < sample_rows; i++) {
            free(sample[i]);
        }
        free(sample);

        Y[0] = create_matrix(sample_rows, n_neurons);
        for (int i = 0; i < sample_rows; i++) {
            for (int j = 0; j < n_neurons; j++) {
                Y[0][i][j] = y[i][j];
            }
        }
        for (int i = 0; i < sample_rows; i++) {
            free(y[i]);
        }
        free(y);

        int matrix_rows = sample_rows;

        for (int layer_index = 1; layer_index < layer_sizes_rows; layer_index++) {
            float n_inputs_float = layer_sizes[layer_index * layer_sizes_cols + 0];
            float n_neurons_float = layer_sizes[layer_index * layer_sizes_cols + 1];
            int n_inputs = (int)n_inputs_float;
            int n_neurons = (int)n_neurons_float;

            float **x = create_matrix(matrix_rows, n_inputs);
            for (int i = 0; i < matrix_rows; i++) {
                for (int j = 0; j < n_inputs; j++) {
                    x[i][j] = Y[layer_index - 1][i][j];
                }
            }

            X[layer_index] = create_matrix(matrix_rows, n_inputs);
            for (int i = 0; i < matrix_rows; i++) {
                for (int j = 0; j < n_inputs; j++) {
                    X[layer_index][i][j] = x[i][j];
                }
            }

            float **y = create_matrix(matrix_rows, n_neurons);
            matmul(x, weights[layer_index], y, matrix_rows, n_inputs, n_inputs, n_neurons);
            for (int i = 0; i < matrix_rows; i++) {
                free(x[i]);
            }
            free(x);

            for (int i = 0; i < matrix_rows; ++i) {
                for (int j = 0; j < n_neurons; ++j) {
                    y[i][j] += biases[layer_index][i];
                }
            }
            int activation = (int)activations[layer_index];
            apply_activation_calc(y, matrix_rows, n_neurons, activation);
            float dropout = dropouts[layer_index];
            apply_dropout(y, matrix_rows, n_neurons, dropout);

            Y[layer_index] = create_matrix(matrix_rows, n_neurons);
            for (int i = 0; i < matrix_rows; i++) {
                for (int j = 0; j < n_neurons; j++) {
                    Y[layer_index][i][j] = y[i][j];
                }
            }
            for (int i = 0; i < matrix_rows; i++) {
                free(y[i]);
            }
            free(y);

            matrix_rows = matrix_rows;
        }

        X_list[dataset_index] = X;
        Y_list[dataset_index] = Y;

    }
    return NULL;
}

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
    int num_threads) {

    pthread_t forward_threads[num_threads];

    // Разбиение всего набора данных на части для каждого потока
    int block_size = dataset_samples_rows / num_threads;
    int remainder = dataset_samples_rows % num_threads;

    // Начинаем раздавать задания потокам
    int start_idx = 0;
    for (int t = 0; t < num_threads; ++t) {
        // Размер блока для текущего потока
        int end_idx = start_idx + block_size + (t < remainder ? 1 : 0);

        // Устанавливаем данные для текущего потока
        forward_thread_data[t].X_list = X_list;
        forward_thread_data[t].Y_list = Y_list;
        forward_thread_data[t].samples = samples;
        forward_thread_data[t].weights = weights;
        forward_thread_data[t].biases = biases;
        forward_thread_data[t].layer_sizes = layer_sizes;
        forward_thread_data[t].activations = activations;
        forward_thread_data[t].dropouts = dropouts;
        forward_thread_data[t].start_idx = start_idx;
        forward_thread_data[t].end_idx = end_idx;
        forward_thread_data[t].sample_rows = dataset_samples_cols;
        forward_thread_data[t].sample_cols = dataset_samples_depth;
        forward_thread_data[t].layer_sizes_rows = layer_sizes_rows;
        forward_thread_data[t].layer_sizes_cols = layer_sizes_cols;

        // Создание нового потока
        pthread_create(&forward_threads[t], NULL, forward_worker, &forward_thread_data[t]);

        // Следующий кусок данных начинается там, где закончился предыдущий
        start_idx = end_idx;
    }

    // Ожидаем завершения всех потоков
    for (int t = 0; t < num_threads; ++t) {
        pthread_join(forward_threads[t], NULL);
    }
}
