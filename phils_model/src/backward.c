#include <stdlib.h>
#include "backward.h"
#include "functions.h"
#include "loss.h"
#include "activations.h"
#include "functions.h"


// Backward pass
void *backward(void *arg) {
    BackwardData *bd = (BackwardData *)arg;

    int dataset_index = bd->dataset_index;
    double ***weights = bd->weights;
    double ***X = bd->X;
    double ***Y = bd->Y;
    double *target = bd->target;
    double ***grad_w = bd->grad_w;
    double ***grad_x = bd->grad_x;
    double **grad_b = bd->grad_b;
    double *layer_sizes = bd->layer_sizes;
    int layer_sizes_rows = bd->layer_sizes_rows;
    int layer_sizes_cols = bd->layer_sizes_cols;
    int matrix_rows = bd->matrix_rows;
    double *activations = bd->activations;
    int loss = bd->loss;
    int threading = bd->threading;
    int num_cpu = bd->num_cpu;
    double *epoch_losses = bd->epoch_losses;
    int regression = bd->regression;

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
    matmul(x_T, delta, w, n_inputs, matrix_rows, matrix_rows, n_neurons, threading, num_cpu);

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
    matmul(delta, w_T, result, matrix_rows, n_neurons, n_neurons, n_inputs, threading, num_cpu);

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
        matmul(x_T, delta, w, n_inputs, matrix_rows, matrix_rows, n_neurons, threading, num_cpu);

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
        matmul(delta, w_T, result_grad_x, matrix_rows, n_neurons, n_neurons, n_inputs, threading, num_cpu);
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

    return NULL;
}

void delete_backward_thread_data(BackwardData *backward_thread_data) {
    double ***X = backward_thread_data->X;
    double ***Y = backward_thread_data->Y;

    double ***grad_w = backward_thread_data->grad_w;
    double ***grad_x = backward_thread_data->grad_x;
    double **grad_b = backward_thread_data->grad_b;

    for (int layer_index = 0; layer_index < backward_thread_data->layer_sizes_rows; ++layer_index) {
        double n_inputs_double = backward_thread_data->layer_sizes[layer_index * backward_thread_data->layer_sizes_cols + 0];
        double n_neurons_double = backward_thread_data->layer_sizes[layer_index * backward_thread_data->layer_sizes_cols + 1];
        int n_inputs = (int)n_inputs_double;
        int n_neurons = (int)n_neurons_double;

        for (int i = 0; i < n_inputs; i++) {
            free(grad_w[layer_index][i]);
        }
        for (int i = 0; i < backward_thread_data->matrix_rows; i++) {
            free(grad_x[layer_index][i]);
        }
        free(grad_w[layer_index]);
        free(grad_x[layer_index]);
        free(grad_b[layer_index]);

        for (int i = 0; i < backward_thread_data->matrix_rows; i++) {
            free(X[layer_index][i]);
            free(Y[layer_index][i]);
        }
        free(X[layer_index]);
        free(Y[layer_index]);
    }
    free(X);
    free(Y);

    free(grad_w);
    free(grad_x);
    free(grad_b);
}
