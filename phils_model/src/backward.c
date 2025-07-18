#include <stdlib.h>
#include "functions.h"
#include "loss.h"


void backward(
    double ***weights,
    double ***Y,
    double ***X,
    double *target,
    double ***grad_w,
    double ***grad_x,
    double **grad_b,
    double *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols,
    int matrix_rows,
    int loss,
    double *activations,
    int threading,
    int num_cpu,
    double *epoch_losses,
    int dataset_index,
    int regression) {

    // Backward pass

    double n_inputs_double = layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols + 0];
    double n_neurons_double = layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols + 1];
    int n_inputs = (int)n_inputs_double;
    int n_neurons = (int)n_neurons_double;

    double **delta = malloc(matrix_rows * sizeof(double*));
    for (int i = 0; i < matrix_rows; i++) {
        delta[i] = malloc(n_neurons * sizeof(double));
    }
    calc_loss(loss, target, Y[layer_sizes_rows - 1], matrix_rows, n_neurons, delta, regression);
    free(target);
    double output_error = sum(delta, matrix_rows, n_neurons);
    epoch_losses[dataset_index] = output_error;

    grad_b[layer_sizes_rows - 1] = sum_axis_0(delta, matrix_rows, n_neurons);

    double **x = malloc(matrix_rows * sizeof(double*));
    for (int i = 0; i < matrix_rows; i++) {
        x[i] = malloc(n_inputs * sizeof(double));
        for (int j = 0; j < n_inputs; j++) {
            x[i][j] = X[layer_sizes_rows - 1][i][j];
        }
    }
    double **x_T = malloc(n_inputs * sizeof(double*));
    for (int i = 0; i < n_inputs; i++) {
        x_T[i] = malloc(matrix_rows * sizeof(double));
    }
    x_T = transpose(x, matrix_rows, n_inputs);
    double **w = malloc(n_inputs * sizeof(double*));
    for (int i = 0; i < n_inputs; i++) {
        w[i] = malloc(n_neurons * sizeof(double));
    }
    matmul(x_T, delta, w, n_inputs, matrix_rows, matrix_rows, n_neurons, threading, num_cpu);
    grad_w[layer_sizes_rows - 1] = malloc(n_inputs * sizeof(double*));
    for (int i = 0; i < n_inputs; i++) {
        grad_w[layer_sizes_rows - 1][i] = malloc(n_neurons * sizeof(double));
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

    double **weight = malloc(n_inputs * sizeof(double*));
    for (int i = 0; i < n_inputs; i++) {
        weight[i] = malloc(n_neurons * sizeof(double));
        for (int j = 0; j < n_neurons; j++) {
            weight[i][j] = weights[layer_sizes_rows - 1][i][j];
        }
    }

    double **w_T = malloc(n_neurons * sizeof(double*));
    for (int i = 0; i < n_neurons; i++) {
        w_T[i] = malloc(n_inputs * sizeof(double));
    }
    w_T = transpose(weight, n_inputs, n_neurons);
    double **result = malloc(matrix_rows * sizeof(double*));
    for (int i = 0; i < matrix_rows; i++) {
        result[i] = malloc(n_inputs * sizeof(double));
    }
    matmul(delta, w_T, result, matrix_rows, n_neurons, n_neurons, n_inputs, threading, num_cpu);
    grad_x[layer_sizes_rows - 1] = malloc(matrix_rows * sizeof(double*));
    for (int i = 0; i < matrix_rows; i++) {
        grad_x[layer_sizes_rows - 1][i] = malloc(n_inputs * sizeof(double));
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

        double **y = malloc(matrix_rows * sizeof(double*));
        for (int i = 0; i < matrix_rows; i++) {
            y[i] = malloc(n_neurons * sizeof(double));
            for (int j = 0; j < n_neurons; j++) {
                y[i][j] = Y[layer_index][i][j];
            }
        }
        int activation = (int)activations[layer_index];
        apply_activation_derivative(y, matrix_rows, n_neurons, activation);

        double **grad = malloc(matrix_rows * sizeof(double*));
        for (int i = 0; i < matrix_rows; i++) {
            grad[i] = malloc(n_neurons * sizeof(double));
            for (int j = 0; j < n_neurons; j++) {
                grad[i][j] = grad_x[layer_index + 1][i][j];
            }
        }

        double **delta = malloc(matrix_rows * sizeof(double*));
        for (int i = 0; i < matrix_rows; i++) {
            delta[i] = malloc(n_neurons * sizeof(double));
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

        double **x = malloc(matrix_rows * sizeof(double*));
        for (int i = 0; i < matrix_rows; i++) {
            x[i] = malloc(n_inputs * sizeof(double));
            for (int j = 0; j < n_inputs; j++) {
                x[i][j] = X[layer_index][i][j];
            }
        }
        double **x_T = malloc(n_inputs * sizeof(double*));
        for (int i = 0; i < n_inputs; i++) {
            x_T[i] = malloc(matrix_rows * sizeof(double));
        }
        x_T = transpose(x, matrix_rows, n_inputs);
        for (int i = 0; i < matrix_rows; i++) {
            free(x[i]);
        }
        free(x);
        double **w = malloc(n_inputs * sizeof(double*));
        for (int i = 0; i < n_inputs; i++) {
            w[i] = malloc(n_neurons * sizeof(double));
        }
        matmul(x_T, delta, w, n_inputs, matrix_rows, matrix_rows, n_neurons, threading, num_cpu);
        grad_w[layer_index] = malloc(n_inputs * sizeof(double*));
        for (int i = 0; i < n_inputs; i++) {
            grad_w[layer_index][i] = malloc(n_neurons * sizeof(double));
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

        double **weight = malloc(n_inputs * sizeof(double*));
        for (int i = 0; i < n_inputs; i++) {
            weight[i] = malloc(n_neurons * sizeof(double));
            for (int j = 0; j < n_neurons; j++) {
                weight[i][j] = weights[layer_index][i][j];
            }
        }
        double **w_T = malloc(n_neurons * sizeof(double*));
        for (int i = 0; i < n_neurons; i++) {
            w_T[i] = malloc(n_inputs * sizeof(double));
        }
        w_T = transpose(weight, n_inputs, n_neurons);
        for (int i = 0; i < n_inputs; i++) {
            free(weight[i]);
        }
        free(weight);
        double **result_grad_x = malloc(matrix_rows * sizeof(double*));
        for (int i = 0; i < matrix_rows; i++) {
            result_grad_x[i] = malloc(n_inputs * sizeof(double));
        }
        matmul(delta, w_T, result_grad_x, matrix_rows, n_neurons, n_neurons, n_inputs, threading, num_cpu);
        for (int i = 0; i < n_neurons; i++) {
            free(w_T[i]);
        }
        free(w_T);
        grad_x[layer_index] = malloc(matrix_rows * sizeof(double*));
        for (int i = 0; i < matrix_rows; i++) {
            grad_x[layer_index][i] = malloc(n_inputs * sizeof(double));
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
}
