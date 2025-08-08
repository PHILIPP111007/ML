#include <stdlib.h>
#include <pthread.h>
#include "functions.h"
#include "loss.h"
#include "activations.h"
#include "functions.h"
#include "backward.h"


// Backward pass
void *backward_worker(void *arg) {
    BackwardData *bd = (BackwardData *)arg;

    // Local variables for better readability and scope control
    const register int start_idx = bd->start_idx;
    const register int end_idx = bd->end_idx;
    const register int layer_sizes_rows = bd->layer_sizes_rows;
    const register int layer_sizes_cols = bd->layer_sizes_cols;
    const register int matrix_rows = bd->matrix_rows;
    const register int regression = bd->regression;
    const register int dataset_samples_rows = bd->dataset_samples_rows;
    const register int dataset_samples_cols = bd->dataset_samples_cols;
    const register int dataset_targets_cols = bd->dataset_targets_cols;
    const register int gpu = bd->gpu;
    cl_context context = bd->context;
    cl_command_queue queue = bd->queue;
    cl_program program = bd->program;

    for (int dataset_index = start_idx; dataset_index < end_idx; dataset_index++) {
        float ***__restrict X = bd->X_list[dataset_index];
        float ***__restrict Y = bd->Y_list[dataset_index];

        // Allocate gradient arrays once per dataset
        float ***__restrict grad_w = malloc(layer_sizes_rows * sizeof(float**));
        float ***__restrict grad_x = malloc(layer_sizes_rows * sizeof(float**));
        float **__restrict grad_b = malloc(layer_sizes_rows * sizeof(float*));

        // Initialize last layer
        const register int n_inputs = (int)bd->layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols];
        const register int n_neurons = (int)bd->layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols + 1];

        float **__restrict delta = create_matrix(matrix_rows, n_neurons);

        // Calculate loss and initial delta
        calc_loss(bd->loss, bd->targets[dataset_index], Y[layer_sizes_rows - 1], matrix_rows, n_neurons, delta, regression);
        bd->epoch_losses[dataset_index] = sum(delta, matrix_rows, n_neurons);

        // Compute gradients for last layer
        grad_b[layer_sizes_rows - 1] = sum_axis_0(delta, matrix_rows, n_neurons);

        float **__restrict x_T = create_matrix(n_inputs, matrix_rows);
        x_T = transpose(X[layer_sizes_rows - 1], matrix_rows, n_inputs);
        grad_w[layer_sizes_rows - 1] = create_matrix(n_inputs, n_neurons);
        matmul(x_T, delta, grad_w[layer_sizes_rows - 1], n_inputs, matrix_rows, matrix_rows, n_neurons);
        free_matrix(x_T);

        float **__restrict w_T = create_matrix(n_neurons, n_inputs);
        w_T = transpose(bd->weights[layer_sizes_rows - 1], n_inputs, n_neurons);
        grad_x[layer_sizes_rows - 1] = create_matrix(matrix_rows, n_inputs);
        matmul(delta, w_T, grad_x[layer_sizes_rows - 1], matrix_rows, n_neurons, n_neurons, n_inputs);
        free_matrix(w_T);
        free_matrix(delta);

        // Backpropagate through hidden layers
        for (int layer_index = layer_sizes_rows - 2; layer_index >= 0; layer_index--) {
            const register int n_inputs = (int)bd->layer_sizes[layer_index * layer_sizes_cols];
            const register int n_neurons = (int)bd->layer_sizes[layer_index * layer_sizes_cols + 1];

            apply_activation_derivative(Y[layer_index], matrix_rows, n_neurons, (int)bd->activations[layer_index]);

            float **__restrict delta = create_matrix(matrix_rows, n_neurons);

            // Compute delta with SIMD optimization
            for (register int i = 0; i < matrix_rows; i++) {
                #pragma omp parallel for simd
                for (register int j = 0; j < n_neurons; j++) {
                    delta[i][j] = grad_x[layer_index + 1][i][j] * Y[layer_index][i][j];
                }
            }

            grad_b[layer_index] = sum_axis_0(delta, matrix_rows, n_neurons);
            grad_w[layer_index] = create_matrix(n_inputs, n_neurons);

            float **__restrict x_T = create_matrix(n_inputs, matrix_rows);
            x_T = transpose(X[layer_index], matrix_rows, n_inputs);

            if (gpu) {
                float *x_T_vec = malloc(n_inputs * matrix_rows * sizeof(float));
                float *delta_vec = malloc(matrix_rows * n_neurons * sizeof(float));
                float *w_vec = malloc(n_inputs * n_neurons * sizeof(float));

                for (int i = 0; i < n_inputs; i++) {
                    #pragma omp simd
                    for (int j = 0; j < matrix_rows; j++) {
                        x_T_vec[i * matrix_rows + j] = x_T[i][j];
                    }
                }
                for (int i = 0; i < matrix_rows; i++) {
                    #pragma omp simd
                    for (int j = 0; j < n_neurons; j++) {
                        delta_vec[i * n_neurons + j] = delta[i][j];
                    }
                }

                matmul_gpu(context, queue, program, x_T_vec, delta_vec, w_vec, n_inputs, matrix_rows, matrix_rows, n_neurons);

                for (int i = 0; i < matrix_rows; i++) {
                    #pragma omp simd
                    for (int j = 0; j < n_neurons; j++) {
                        grad_w[layer_index][i][j] = w_vec[i * n_neurons + j];
                    }
                }

                free(x_T_vec);
                free(delta_vec);
                free(w_vec);
            } else {
                matmul(x_T, delta, grad_w[layer_index], n_inputs, matrix_rows, matrix_rows, n_neurons);
            }
            free_matrix(x_T);

            float **__restrict w_T = create_matrix(n_neurons, n_inputs);
            w_T = transpose(bd->weights[layer_index], n_inputs, n_neurons);
            grad_x[layer_index] = create_matrix(matrix_rows, n_inputs);

            if (gpu) {
                float *delta_vec = malloc(matrix_rows * n_neurons * sizeof(float));
                float *w_T_vec = malloc(n_neurons * n_inputs * sizeof(float));
                float *x_vec = malloc(matrix_rows * n_inputs * sizeof(float));

                for (int i = 0; i < matrix_rows; i++) {
                    #pragma omp simd
                    for (int j = 0; j < n_neurons; j++) {
                        delta_vec[i * n_neurons + j] = delta[i][j];
                    }
                }
                for (int i = 0; i < n_neurons; i++) {
                    #pragma omp simd
                    for (int j = 0; j < n_inputs; j++) {
                        w_T_vec[i * n_inputs + j] = w_T[i][j];
                    }
                }

                matmul_gpu(context, queue, program, delta_vec, w_T_vec, x_vec, matrix_rows, n_neurons, n_neurons, n_inputs);

                for (int i = 0; i < matrix_rows; i++) {
                    #pragma omp simd
                    for (int j = 0; j < n_inputs; j++) {
                        grad_x[layer_index][i][j] = x_vec[i * n_inputs + j];
                    }
                }
                free(delta_vec);
                free(w_T_vec);
                free(x_vec);
            } else {
                matmul(delta, w_T, grad_x[layer_index], matrix_rows, n_neurons, n_neurons, n_inputs);
            }
            free_matrix(w_T);

            free_matrix(delta);
        }

        // Store gradients
        bd->grad_w_list[dataset_index] = grad_w;
        bd->grad_x_list[dataset_index] = grad_x;
        bd->grad_b_list[dataset_index] = grad_b;
    }
    return NULL;
}

void backward_threading(
    BackwardData *backward_thread_data,
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
    int num_threads,
    int gpu,
    cl_context context,
    cl_command_queue queue,
    cl_program program) {

    pthread_t backward_threads[num_threads];

    // Splitting the entire data set into parts for each thread
    int block_size = dataset_samples_rows / num_threads;
    int remainder = dataset_samples_rows % num_threads;

    int start_idx = 0;
    for (int t = 0; t < num_threads; t++) {
        // Block size for the current thread
        int end_idx = start_idx + block_size + (t < remainder ? 1 : 0);

        backward_thread_data[t].weights = weights;
        backward_thread_data[t].X_list = X_list;
        backward_thread_data[t].Y_list = Y_list;
        backward_thread_data[t].grad_w_list = grad_w_list;
        backward_thread_data[t].grad_x_list = grad_x_list;
        backward_thread_data[t].grad_b_list = grad_b_list;
        backward_thread_data[t].targets = targets;
        backward_thread_data[t].layer_sizes = layer_sizes;
        backward_thread_data[t].activations = activations;
        backward_thread_data[t].epoch_losses = epoch_losses;
        backward_thread_data[t].start_idx = start_idx;
        backward_thread_data[t].end_idx = end_idx;
        backward_thread_data[t].layer_sizes_rows = layer_sizes_rows;
        backward_thread_data[t].layer_sizes_cols = layer_sizes_cols;
        backward_thread_data[t].matrix_rows = matrix_rows;
        backward_thread_data[t].loss = loss;
        backward_thread_data[t].regression = regression;
        backward_thread_data[t].dataset_samples_rows = dataset_samples_rows;
        backward_thread_data[t].dataset_samples_cols = dataset_samples_cols;
        backward_thread_data[t].dataset_targets_cols = dataset_targets_cols;
        backward_thread_data[t].gpu = gpu;
        backward_thread_data[t].context = context;
        backward_thread_data[t].queue = queue;
        backward_thread_data[t].program = program;

        pthread_create(&backward_threads[t], NULL, backward_worker, &backward_thread_data[t]);

        // The next piece of data starts where the previous one ended.
        start_idx = end_idx;
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(backward_threads[t], NULL);
    }
}
