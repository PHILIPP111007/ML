#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include "functions.h"
#include "loss.h"
#include "activations.h"
#include "forward.h"


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
    float *activations,
    int is_predict_one_saple,
    int gpu,
    cl_context context,
    cl_command_queue queue,
    cl_program program,
    cl_mem weights_vec_buf) {

    const register int n_inputs = (int)layer_sizes[0 * layer_sizes_cols];
    const register int n_neurons = (int)layer_sizes[0 * layer_sizes_cols + 1];

    Y[0] = create_matrix(sample_rows, n_neurons);

    if (gpu) {
        float *sample_vec = malloc(sample_rows * sample_cols * sizeof(float));
        float *weights_vec = malloc(n_inputs * n_neurons * sizeof(float));
        float *y_vec = malloc(sample_rows * n_neurons * sizeof(float));

        for (int i = 0; i < sample_rows; i++) {
            #pragma omp simd
            for (int j = 0; j < sample_cols; j++) {
                sample_vec[i * sample_cols + j] = sample[i][j];
            }
        }
        for (int i = 0; i < n_inputs; i++) {
            #pragma omp simd
            for (int j = 0; j < n_neurons; j++) {
                weights_vec[i * n_neurons + j] = weights[0][i][j];
            }
        }

        matmul_gpu(context, queue, program, sample_vec, weights_vec_buf, y_vec, sample_rows, sample_cols, n_inputs, n_neurons, 0);

        for (int i = 0; i < sample_rows; i++) {
            #pragma omp simd
            for (int j = 0; j < n_neurons; j++) {
                Y[0][i][j] = y_vec[i * n_neurons + j];
            }
        }

        free(sample_vec);
        free(weights_vec);
        free(y_vec);
    } else {
        matmul(sample, weights[0], Y[0], sample_rows, sample_cols, n_inputs, n_neurons);
    }

    for (int i = 0; i < sample_rows; i++) {
        #pragma omp simd
        for (int j = 0; j < n_neurons; j++) {
            Y[0][i][j] += biases[0][i];
        }
    }
    register int activation = (int)activations[0];
    apply_activation_calc(Y[0], sample_rows, n_neurons, activation);

    int matrix_rows = sample_rows;
    for (int layer_index = 1; layer_index < layer_sizes_rows; layer_index++) {
        const register int n_inputs = (int)layer_sizes[layer_index * layer_sizes_cols];
        const register int n_neurons = (int)layer_sizes[layer_index * layer_sizes_cols + 1];

        Y[layer_index] = create_matrix(matrix_rows, n_neurons);

        if (gpu) {
            float *y_vec = malloc(matrix_rows * n_inputs * sizeof(float));
            float *weights_vec = malloc(n_inputs * n_neurons * sizeof(float));
            float *y_new_vec = malloc(matrix_rows * n_neurons * sizeof(float));

            for (int i = 0; i < matrix_rows; i++) {
                #pragma omp simd
                for (int j = 0; j < n_inputs; j++) {
                    y_vec[i * n_inputs + j] = Y[layer_index - 1][i][j];
                }
            }
            for (int i = 0; i < n_inputs; i++) {
                #pragma omp simd
                for (int j = 0; j < n_neurons; j++) {
                    weights_vec[i * n_neurons + j] = weights[layer_index][i][j];
                }
            }

            matmul_gpu(context, queue, program, y_vec, weights_vec_buf, y_new_vec, matrix_rows, n_inputs, n_inputs, n_neurons, layer_index);

            for (int i = 0; i < matrix_rows; i++) {
                #pragma omp simd
                for (int j = 0; j < n_neurons; j++) {
                    Y[layer_index][i][j] = y_new_vec[i * n_neurons + j];
                }
            }

            free(y_vec);
            free(weights_vec);
            free(y_new_vec);
        } else {
            matmul(Y[layer_index - 1], weights[layer_index], Y[layer_index], matrix_rows, n_inputs, n_inputs, n_neurons);
        }

        for (register int i = 0; i < matrix_rows; i++) {
            #pragma omp simd
            for (register int j = 0; j < n_neurons; j++) {
                Y[layer_index][i][j] += biases[layer_index][i];
            }
        }
        int activation = (int)activations[layer_index];
        apply_activation_calc(Y[layer_index], matrix_rows, n_neurons, activation);
    }

    for (register int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
        const register int n_neurons = (int)layer_sizes[layer_index * layer_sizes_cols + 1];

        for (register int i = 0; i < matrix_rows; i++) {
            for (register int j = 0; j < n_neurons; j++) {
                Y[layer_index][i][j] = check_if_isnan(Y[layer_index][i][j]);
            }
        }
    }
}

// Forward pass
void *forward_worker(void *arg) {
    ForwardData *fd = (ForwardData *)arg;

    const int start_idx = fd->start_idx;
    const int end_idx = fd->end_idx;
    float ***samples = fd->samples;
    int sample_rows = fd->sample_rows;
    const int sample_cols = fd->sample_cols;
    float ****X_list = fd->X_list;
    float ****Y_list = fd->Y_list;
    float ***weights = fd->weights;
    float **biases = fd->biases;
    float *layer_sizes = fd->layer_sizes;
    const int layer_sizes_rows = fd->layer_sizes_rows;
    const int layer_sizes_cols = fd->layer_sizes_cols;
    float *activations = fd->activations;
    float *dropouts = fd->dropouts;
    const int gpu = fd->gpu;
    cl_mem weights_vec_buf = fd->weights_vec_buf;

    for (register int dataset_index = start_idx; dataset_index < end_idx; dataset_index++) {
        float **__restrict sample = create_matrix(sample_rows, sample_cols);
        for (int i = 0; i < sample_rows; i++) {

            #pragma omp simd
            for (register int j = 0; j < sample_cols; j++) {
                sample[i][j] = samples[dataset_index][i][j];
            }
        }

        float ***__restrict X = malloc(layer_sizes_rows * sizeof(float**));
        float ***__restrict Y = malloc(layer_sizes_rows * sizeof(float**));

        const register int n_inputs = (int)layer_sizes[0 * layer_sizes_cols];
        const register int n_neurons = (int)layer_sizes[0 * layer_sizes_cols + 1];

        Y[0] = create_matrix(sample_rows, n_neurons);
        matmul(sample, weights[0], Y[0], sample_rows, sample_cols, n_inputs, n_neurons);
        for (register int i = 0; i < sample_rows; i++) {
            for (register int j = 0; j < n_neurons; j++) {
                Y[0][i][j] += biases[0][i];
                Y[0][i][j] = check_if_isnan(Y[0][i][j]);

            }
        }
        int activation = (int)activations[0];
        apply_activation_calc(Y[0], sample_rows, n_neurons, activation);
        float dropout = dropouts[0];
        apply_dropout(Y[0], sample_rows, n_neurons, dropout);

        X[0] = create_matrix(sample_rows, sample_cols);
        for (register int i = 0; i < sample_rows; i++) {
            #pragma omp simd
            for (register int j = 0; j < sample_cols; j++) {
                X[0][i][j] = sample[i][j];
            }
        }
        free_matrix(sample);

        int matrix_rows = sample_rows;

        for (int layer_index = 1; layer_index < layer_sizes_rows; layer_index++) {
            const register int n_inputs = (int)layer_sizes[layer_index * layer_sizes_cols];
            const register int n_neurons = (int)layer_sizes[layer_index * layer_sizes_cols + 1];

            X[layer_index] = create_matrix(matrix_rows, n_inputs);
            for (register int i = 0; i < matrix_rows; i++) {
                #pragma omp simd
                for (register int j = 0; j < n_inputs; j++) {
                    X[layer_index][i][j] = Y[layer_index - 1][i][j];
                }
            }

            Y[layer_index] = create_matrix(matrix_rows, n_neurons);

            if (gpu) {
                float *x_vec = malloc(matrix_rows * n_inputs * sizeof(float));
                float *weights_vec = malloc(n_inputs * n_neurons * sizeof(float));
                float *y_vec = malloc(matrix_rows * n_neurons * sizeof(float));

                for (int i = 0; i < matrix_rows; i++) {
                    #pragma omp simd
                    for (int j = 0; j < n_inputs; j++) {
                        x_vec[i * n_inputs + j] = X[layer_index][i][j];
                    }
                }

                matmul_gpu(fd->context, fd->queue, fd->program, x_vec, weights_vec_buf, y_vec, matrix_rows, n_inputs, n_inputs, n_neurons, layer_index);

                for (int i = 0; i < matrix_rows; i++) {
                    #pragma omp simd
                    for (int j = 0; j < n_neurons; j++) {
                        Y[layer_index][i][j] = y_vec[i * n_neurons + j];
                    }
                }

                free(x_vec);
                free(y_vec);
            } else {
                matmul(X[layer_index], weights[layer_index], Y[layer_index], matrix_rows, n_inputs, n_inputs, n_neurons);
            }

            for (register int i = 0; i < matrix_rows; i++) {
                for (register int j = 0; j < n_neurons; j++) {
                    Y[layer_index][i][j] += biases[layer_index][i];
                    Y[layer_index][i][j] = check_if_isnan(Y[layer_index][i][j]);
                }
            }
            int activation = (int)activations[layer_index];
            apply_activation_calc(Y[layer_index], matrix_rows, n_neurons, activation);
            float dropout = dropouts[layer_index];
            apply_dropout(Y[layer_index], matrix_rows, n_neurons, dropout);
        }

        X_list[dataset_index] = X;
        Y_list[dataset_index] = Y;
    }
    return NULL;
}

void forward_threading(
    ForwardData *forward_thread_data,
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
    cl_program program,
    cl_mem weights_vec_buf) {

    pthread_t forward_threads[num_threads];

    // Разбиение всего набора данных на части для каждого потока
    int block_size = dataset_samples_rows / num_threads;
    int remainder = dataset_samples_rows % num_threads;

    // Начинаем раздавать задания потокам
    int start_idx = 0;
    for (int t = 0; t < num_threads; t++) {
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
        forward_thread_data[t].gpu = gpu;
        forward_thread_data[t].context = context;
        forward_thread_data[t].queue = queue;
        forward_thread_data[t].program = program;
        forward_thread_data[t].weights_vec_buf = weights_vec_buf;

        // Создание нового потока
        pthread_create(&forward_threads[t], NULL, forward_worker, &forward_thread_data[t]);

        // Следующий кусок данных начинается там, где закончился предыдущий
        start_idx = end_idx;
    }

    // Ожидаем завершения всех потоков
    for (int t = 0; t < num_threads; t++) {
        pthread_join(forward_threads[t], NULL);
    }
}
