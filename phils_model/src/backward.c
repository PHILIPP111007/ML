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
    float ***weights = bd->weights;
    float **targets = bd->targets;
    float ****X_list = bd->X_list;
    float ****Y_list = bd->Y_list;
    float *layer_sizes = bd->layer_sizes;
    int layer_sizes_rows = bd->layer_sizes_rows;
    int layer_sizes_cols = bd->layer_sizes_cols;
    int matrix_rows = bd->matrix_rows;
    float *activations = bd->activations;
    int loss = bd->loss;
    float *epoch_losses = bd->epoch_losses;
    int regression = bd->regression;
    float ****grad_w_list = bd->grad_w_list;
    float ****grad_x_list = bd->grad_x_list;
    float ***grad_b_list = bd->grad_b_list;
    int dataset_samples_rows = bd->dataset_samples_rows;
    int dataset_samples_cols = bd->dataset_samples_cols;
    int dataset_targets_cols = bd->dataset_targets_cols;

    for (int dataset_index = start_idx; dataset_index < end_idx; dataset_index++) {
        float ***X = X_list[dataset_index];
        float ***Y = Y_list[dataset_index];

        float *target = malloc(dataset_targets_cols * sizeof(float));
        for (int i = 0; i < dataset_targets_cols; i++) {
            target[i] = targets[dataset_index][i];
        }

        float ***grad_w = malloc(layer_sizes_rows * sizeof(float**));
        float ***grad_x = malloc(layer_sizes_rows * sizeof(float**));
        float **grad_b = malloc(layer_sizes_rows * sizeof(float*));

        const int n_inputs = (int)layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols];
        const int n_neurons = (int)layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols + 1];

        float **delta = create_matrix(matrix_rows, n_neurons);

        calc_loss(loss, target, Y[layer_sizes_rows - 1], matrix_rows, n_neurons, delta, regression);
        free(target);
        float output_error = sum(delta, matrix_rows, n_neurons);
        epoch_losses[dataset_index] = output_error;

        grad_b[layer_sizes_rows - 1] = sum_axis_0(delta, matrix_rows, n_neurons);

        float **x_T = create_matrix(n_inputs, matrix_rows);
        x_T = transpose(X[layer_sizes_rows - 1], matrix_rows, n_inputs);
        grad_w[layer_sizes_rows - 1] = create_matrix(n_inputs, n_neurons);
        matmul(x_T, delta, grad_w[layer_sizes_rows - 1], n_inputs, matrix_rows, matrix_rows, n_neurons);
        for (int i = 0; i < n_inputs; i++) {
            free(x_T[i]);
        }
        free(x_T);

        float **w_T = create_matrix(n_neurons, n_inputs);
        w_T = transpose(weights[layer_sizes_rows - 1], n_inputs, n_neurons);
        grad_x[layer_sizes_rows - 1] = create_matrix(matrix_rows, n_inputs);
        matmul(delta, w_T, grad_x[layer_sizes_rows - 1], matrix_rows, n_neurons, n_neurons, n_inputs);

        for (int i = 0; i < n_neurons; i++) {
            free(w_T[i]);
        }
        free(w_T);
        for (int i = 0; i < matrix_rows; i++) {
            free(delta[i]);
        }
        free(delta);

        for (int layer_index = layer_sizes_rows - 2; layer_index >= 0; layer_index--) {
            const int n_inputs = (int)layer_sizes[layer_index * layer_sizes_cols];
            const int n_neurons = (int)layer_sizes[layer_index * layer_sizes_cols + 1];

            int activation = (int)activations[layer_index];
            apply_activation_derivative(Y[layer_index], matrix_rows, n_neurons, activation);

            float **delta = create_matrix(matrix_rows, n_neurons);
            for (int i = 0; i < matrix_rows; i++) {

                #pragma omp simd
                for (int j = 0; j < n_neurons; j++) {
                    delta[i][j] = grad_x[layer_index + 1][i][j] * Y[layer_index][i][j];
                }
            }

            grad_b[layer_index] = sum_axis_0(delta, matrix_rows, n_neurons);

            float **x_T = create_matrix(n_inputs, matrix_rows);
            x_T = transpose(X[layer_index], matrix_rows, n_inputs);

            grad_w[layer_index] = create_matrix(n_inputs, n_neurons);
            matmul(x_T, delta, grad_w[layer_index], n_inputs, matrix_rows, matrix_rows, n_neurons);
            for (int i = 0; i < n_inputs; i++) {
                free(x_T[i]);
            }
            free(x_T);

            float **w_T = create_matrix(n_neurons, n_inputs);
            w_T = transpose(weights[layer_index], n_inputs, n_neurons);

            grad_x[layer_index] = create_matrix(matrix_rows, n_inputs);
            matmul(delta, w_T, grad_x[layer_index], matrix_rows, n_neurons, n_neurons, n_inputs);
            for (int i = 0; i < n_neurons; i++) {
                free(w_T[i]);
            }
            free(w_T);
            for (int i = 0; i < matrix_rows; i++) {
                free(delta[i]);
            }
            free(delta);
        }
        grad_w_list[dataset_index] = grad_w;
        grad_x_list[dataset_index] = grad_x;
        grad_b_list[dataset_index] = grad_b;
    }
    return NULL;
}

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
    int num_threads) {

    pthread_t backward_threads[num_threads];

    // Splitting the entire data set into parts for each thread
    int block_size = dataset_samples_rows / num_threads;
    int remainder = dataset_samples_rows % num_threads;

    int start_idx = 0;
    for (int t = 0; t < num_threads; t++) {
        // Block size for the current thread
        int end_idx = start_idx + block_size + (t < remainder ? 1 : 0);

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

        pthread_create(&backward_threads[t], NULL, backward_worker, &backward_thread_data[t]);

        // The next piece of data starts where the previous one ended.
        start_idx = end_idx;
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(backward_threads[t], NULL);
    }
}
