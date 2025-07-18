#include <stdio.h>
#include <stdlib.h>
#include "forward.h"
#include "functions.h"
#include "loss.h"
#include "activations.h"


void forward(
    double **sample,
    int sample_rows,
    int sample_cols,
    double ***weights,
    double **biases,
    double ***Y,
    double *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols,
    double *activations,
    int threading,
    int num_cpu) {

    // Forward pass

    double n_inputs_double = layer_sizes[0 * layer_sizes_cols + 0];
    double n_neurons_double = layer_sizes[0 * layer_sizes_cols + 1];
    int n_inputs = (int)n_inputs_double;
    int n_neurons = (int)n_neurons_double;

    double **y = create_matrix(sample_rows, n_neurons);

    matmul(sample, weights[0], y, sample_rows, sample_cols, n_inputs, n_neurons, threading, num_cpu);

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
        double n_inputs_double = layer_sizes[layer_index * layer_sizes_cols + 0];
        double n_neurons_double = layer_sizes[layer_index * layer_sizes_cols + 1];
        int n_inputs = (int)n_inputs_double;
        int n_neurons = (int)n_neurons_double;

        double **x = create_matrix(matrix_rows, n_inputs);
        for (int i = 0; i < matrix_rows; i++) {
            for (int j = 0; j < n_inputs; j++) {
                x[i][j] = Y[layer_index - 1][i][j];
            }
        }

        double **y = create_matrix(matrix_rows, n_neurons);
        matmul(x, weights[layer_index], y, matrix_rows, n_inputs, n_inputs, n_neurons, threading, num_cpu);

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

void forward_train(
    double **sample,
    int sample_rows,
    int sample_cols,
    double ***weights,
    double **biases,
    double ***X,
    double ***Y,
    double *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols,
    double *activations,
    double keep_prob,
    int threading,
    int num_cpu) {

    // Forward pass

    double n_inputs_double = layer_sizes[0 * layer_sizes_cols + 0];
    double n_neurons_double = layer_sizes[0 * layer_sizes_cols + 1];
    int n_inputs = (int)n_inputs_double;
    int n_neurons = (int)n_neurons_double;

    double **y = create_matrix(sample_rows, n_neurons);
    matmul(sample, weights[0], y, sample_rows, sample_cols, n_inputs, n_neurons, threading, num_cpu);

    for (int i = 0; i < sample_rows; ++i) {
        for (int j = 0; j < n_neurons; ++j) {
            y[i][j] += biases[0][i];
        }
    }
    int activation = (int)activations[0];
    apply_activation_calc(y, sample_rows, n_neurons, activation);
    dropout(y, sample_rows, n_neurons, keep_prob);

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
        double n_inputs_double = layer_sizes[layer_index * layer_sizes_cols + 0];
        double n_neurons_double = layer_sizes[layer_index * layer_sizes_cols + 1];
        int n_inputs = (int)n_inputs_double;
        int n_neurons = (int)n_neurons_double;

        double **x = create_matrix(matrix_rows, n_inputs);
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

        double **y = create_matrix(matrix_rows, n_neurons);
        matmul(x, weights[layer_index], y, matrix_rows, n_inputs, n_inputs, n_neurons, threading, num_cpu);
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
        dropout(y, matrix_rows, n_neurons, keep_prob);

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
}
