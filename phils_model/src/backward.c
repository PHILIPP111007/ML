#include <stdlib.h>
#include <pthread.h>
#include "backward.h"
#include "functions.h"
#include "loss.h"
#include "activations.h"
#include "functions.h"


// Backward pass
void *backward_worker(void *arg) {
    BackwardData *bd = (BackwardData *)arg;

    int start_idx = bd->start_idx;
    int end_idx = bd->end_idx;
    double ***weights = bd->weights;
    double **targets = bd->targets;
    double ****X_list = bd->X_list;
    double ****Y_list = bd->Y_list;
    double *layer_sizes = bd->layer_sizes;
    int layer_sizes_rows = bd->layer_sizes_rows;
    int layer_sizes_cols = bd->layer_sizes_cols;
    int matrix_rows = bd->matrix_rows;
    double *activations = bd->activations;
    int loss = bd->loss;
    double *epoch_losses = bd->epoch_losses;
    int regression = bd->regression;
    double ****grad_w_list = bd->grad_w_list;
    double ****grad_x_list = bd->grad_x_list;
    double ***grad_b_list = bd->grad_b_list;
    int dataset_samples_rows = bd->dataset_samples_rows;
    int dataset_samples_cols = bd->dataset_samples_cols;
    int dataset_targets_cols = bd->dataset_targets_cols;

    for (int dataset_index = start_idx; dataset_index < end_idx; ++dataset_index) {
        double ***X = X_list[dataset_index];
        double ***Y = Y_list[dataset_index];

        double *target = malloc(dataset_targets_cols * sizeof(double));
        for (int i = 0; i < dataset_targets_cols; i++) {
            target[i] = targets[dataset_index][i];
        }

        double ***grad_w = malloc(layer_sizes_rows * sizeof(double**));
        double ***grad_x = malloc(layer_sizes_rows * sizeof(double**));
        double **grad_b = malloc(layer_sizes_rows * sizeof(double*));

        double n_inputs_double = layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols + 0];
        double n_neurons_double = layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols + 1];
        int n_inputs = (int)n_inputs_double;
        int n_neurons = (int)n_neurons_double;

        double **delta = create_matrix(matrix_rows, n_neurons);

        calc_loss(loss, target, Y[layer_sizes_rows - 1], matrix_rows, n_neurons, delta, regression);
        free(target);
        double output_error = sum(delta, matrix_rows, n_neurons);
        epoch_losses[dataset_index] = output_error;

        grad_b[layer_sizes_rows - 1] = sum_axis_0(delta, matrix_rows, n_neurons);

        double **x = create_matrix(matrix_rows, n_inputs);
        for (int i = 0; i < matrix_rows; i++) {
            for (int j = 0; j < n_inputs; j++) {
                x[i][j] = X[layer_sizes_rows - 1][i][j];
            }
        }

        double **x_T = create_matrix(n_inputs, matrix_rows);
        x_T = transpose(x, matrix_rows, n_inputs);
        double **w = create_matrix(n_inputs, n_neurons);
        matmul(x_T, delta, w, n_inputs, matrix_rows, matrix_rows, n_neurons);

        grad_w[layer_sizes_rows - 1] = create_matrix(n_inputs, n_neurons);
        for (int i = 0; i < n_inputs; i++) {
            for (int j = 0; j < n_neurons; j++) {
                grad_w[layer_sizes_rows - 1][i][j] = w[i][j];
            }
        }
        for (int i = 0; i < n_inputs; i++) {
            free(x_T[i]);
            free(w[i]);
        }
        free(x_T);
        free(w);
        for (int i = 0; i < matrix_rows; i++) {
            free(x[i]);
        }
        free(x);

        double **weight = create_matrix(n_inputs, n_neurons);
        for (int i = 0; i < n_inputs; i++) {
            for (int j = 0; j < n_neurons; j++) {
                weight[i][j] = weights[layer_sizes_rows - 1][i][j];
            }
        }

        double **w_T = create_matrix(n_neurons, n_inputs);
        w_T = transpose(weight, n_inputs, n_neurons);
        double **result = create_matrix(matrix_rows, n_inputs);
        matmul(delta, w_T, result, matrix_rows, n_neurons, n_neurons, n_inputs);

        grad_x[layer_sizes_rows - 1] = create_matrix(matrix_rows, n_inputs);
        for (int i = 0; i < matrix_rows; i++) {
            for (int j = 0; j < n_inputs; j++) {
                grad_x[layer_sizes_rows - 1][i][j] = result[i][j];
            }
        }
        for (int i = 0; i < n_inputs; i++) {
            free(weight[i]);
        }
        free(weight);
        for (int i = 0; i < n_neurons; i++) {
            free(w_T[i]);
        }
        free(w_T);
        for (int i = 0; i < matrix_rows; i++) {
            free(result[i]);
            free(delta[i]);
        }
        free(result);
        free(delta);

        matrix_rows = matrix_rows;

        for (int layer_index = layer_sizes_rows - 2; layer_index >= 0; layer_index--) {
            double n_inputs_double = layer_sizes[layer_index * layer_sizes_cols + 0];
            double n_neurons_double = layer_sizes[layer_index * layer_sizes_cols + 1];
            int n_inputs = (int)n_inputs_double;
            int n_neurons = (int)n_neurons_double;

            double **y = create_matrix(matrix_rows, n_neurons);
            for (int i = 0; i < matrix_rows; i++) {
                for (int j = 0; j < n_neurons; j++) {
                    y[i][j] = Y[layer_index][i][j];
                }
            }
            int activation = (int)activations[layer_index];
            apply_activation_derivative(y, matrix_rows, n_neurons, activation);

            double **grad = create_matrix(matrix_rows, n_neurons);
            for (int i = 0; i < matrix_rows; i++) {
                for (int j = 0; j < n_neurons; j++) {
                    grad[i][j] = grad_x[layer_index + 1][i][j];
                }
            }

            double **delta = create_matrix(matrix_rows, n_neurons);
            for (int i = 0; i < matrix_rows; i++) {
                for (int j = 0; j < n_neurons; j++) {
                    delta[i][j] = grad[i][j] * y[i][j];
                }
            }
            for (int i = 0; i < matrix_rows; i++) {
                free(grad[i]);
                free(y[i]);
            }
            free(grad);
            free(y);

            grad_b[layer_index] = sum_axis_0(delta, matrix_rows, n_neurons);


            double **x = create_matrix(matrix_rows, n_inputs);
            for (int i = 0; i < matrix_rows; i++) {
                for (int j = 0; j < n_inputs; j++) {
                    x[i][j] = X[layer_index][i][j];
                }
            }

            double **x_T = create_matrix(n_inputs, matrix_rows);
            x_T = transpose(x, matrix_rows, n_inputs);
            for (int i = 0; i < matrix_rows; i++) {
                free(x[i]);
            }
            free(x);

            double **w = create_matrix(n_inputs, n_neurons);
            matmul(x_T, delta, w, n_inputs, matrix_rows, matrix_rows, n_neurons);

            grad_w[layer_index] = create_matrix(n_inputs, n_neurons);
            for (int i = 0; i < n_inputs; i++) {
                for (int j = 0; j < n_neurons; j++) {
                    grad_w[layer_index][i][j] = w[i][j];
                }
            }
            for (int i = 0; i < n_inputs; i++) {
                free(w[i]);
                free(x_T[i]);
            }
            free(w);
            free(x_T);

            double **weight = create_matrix(n_inputs, n_neurons);
            for (int i = 0; i < n_inputs; i++) {
                for (int j = 0; j < n_neurons; j++) {
                    weight[i][j] = weights[layer_index][i][j];
                }
            }

            double **w_T = create_matrix(n_neurons, n_inputs);
            w_T = transpose(weight, n_inputs, n_neurons);
            for (int i = 0; i < n_inputs; i++) {
                free(weight[i]);
            }
            free(weight);

            double **result_grad_x = create_matrix(matrix_rows, n_inputs);
            matmul(delta, w_T, result_grad_x, matrix_rows, n_neurons, n_neurons, n_inputs);
            for (int i = 0; i < n_neurons; i++) {
                free(w_T[i]);
            }
            free(w_T);

            grad_x[layer_index] = create_matrix(matrix_rows, n_inputs);
            for (int i = 0; i < matrix_rows; i++) {
                for (int j = 0; j < n_inputs; j++) {
                    grad_x[layer_index][i][j] = result_grad_x[i][j];
                }
            }
            for (int i = 0; i < matrix_rows; i++) {
                free(result_grad_x[i]);
                free(delta[i]);
            }
            free(result_grad_x);
            free(delta);

            matrix_rows = matrix_rows;
        }
        grad_w_list[dataset_index] = grad_w;
        grad_x_list[dataset_index] = grad_x;
        grad_b_list[dataset_index] = grad_b;

    }
    return NULL;
}

void backward_threading(
    struct BackwardData backward_thread_data[],
    double ***weights,
    double **targets,
    double **biases,
    double ****X_list,
    double ****Y_list,
    double ****grad_w_list,
    double ****grad_x_list,
    double ***grad_b_list,
    double *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols,
    int dataset_samples_rows,
    int dataset_samples_cols,
    int dataset_targets_cols,
    int matrix_rows,
    double *activations,
    int loss,
    double *epoch_losses,
    int regression,
    int num_threads) {

    pthread_t backward_threads[num_threads];

    // Разбиение всего набора данных на части для каждого потока
    int block_size = dataset_samples_rows / num_threads;
    int remainder = dataset_samples_rows % num_threads;

    // Начинаем раздавать задания потокам
    int start_idx = 0;
    for (int t = 0; t < num_threads; ++t) {
        // Размер блока для текущего потока
        int end_idx = start_idx + block_size + (t < remainder ? 1 : 0);

        // Устанавливаем данные для текущего потока
        backward_thread_data[t].start_idx = start_idx;
        backward_thread_data[t].end_idx = end_idx;
        backward_thread_data[t].weights = weights;
        backward_thread_data[t].targets = targets;
        backward_thread_data[t].X_list = X_list;
        backward_thread_data[t].Y_list = Y_list;
        backward_thread_data[t].layer_sizes = layer_sizes;
        backward_thread_data[t].layer_sizes_rows = layer_sizes_rows;
        backward_thread_data[t].layer_sizes_cols = layer_sizes_cols;
        backward_thread_data[t].matrix_rows = matrix_rows;
        backward_thread_data[t].activations = activations;
        backward_thread_data[t].loss = loss;
        backward_thread_data[t].epoch_losses = epoch_losses;
        backward_thread_data[t].regression = regression;
        backward_thread_data[t].grad_w_list = grad_w_list;
        backward_thread_data[t].grad_x_list = grad_x_list;
        backward_thread_data[t].grad_b_list = grad_b_list;
        backward_thread_data[t].dataset_samples_rows = dataset_samples_rows;
        backward_thread_data[t].dataset_samples_cols = dataset_samples_cols;
        backward_thread_data[t].dataset_targets_cols = dataset_targets_cols;


        // Создание нового потока
        pthread_create(&backward_threads[t], NULL, backward_worker, &backward_thread_data[t]);

        // Следующий кусок данных начинается там, где закончился предыдущий
        start_idx = end_idx;
    }

    // Ожидаем завершения всех потоков
    for (int t = 0; t < num_threads; ++t) {
        pthread_join(backward_threads[t], NULL);
    }
}
