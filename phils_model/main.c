// clang -shared -o main.so -fPIC -O3 main.c src/functions.c src/activations.c src/loss.c src/init.c src/json.c src/adam.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "src/functions.h"
#include "src/activations.h"
#include "src/loss.h"
#include "src/init.h"
#include "src/json.h"
#include "src/adam.h"


void fit(
    double *dataset_samples,
    double *dataset_targets,
    int dataset_samples_rows,
    int dataset_samples_cols,
    int dataset_samples_depth,
    int dataset_targets_rows,
    int dataset_targets_cols,
    double *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols,
    double *activations,
    int activations_len,
    int loss,
    int n_epoch,
    double learning_rate,
    int verbose,
    double max_change,
    int random_state,
    double keep_prob,
    int regression,
    int threading,
    int num_cpu) {
    
    if (random_state != -1) {
        srand(random_state); // устанавливаем начальное состояние генератора
    }

    // Загрузка датасета
    double*** samples = malloc(dataset_samples_rows * sizeof(double**));
    double** targets = malloc(dataset_targets_rows * sizeof(double*)); 

    for (int dataset_index = 0; dataset_index < dataset_samples_rows; ++dataset_index) {
        samples[dataset_index] = malloc(dataset_samples_cols * sizeof(double*));
        for (int i = 0; i < dataset_samples_cols; ++i) {
            samples[dataset_index][i] = malloc(dataset_samples_depth * sizeof(double));
            for (int j = 0; j < dataset_samples_depth; ++j) {
                int index = dataset_index * dataset_samples_cols * dataset_samples_depth + i * dataset_samples_depth + j;
                samples[dataset_index][i][j] = (double)dataset_samples[index];
            }
        }
    }

    for (int i = 0; i < dataset_targets_rows; ++i) {
        targets[i] = malloc(dataset_targets_cols * sizeof(double));
        for (int j = 0; j < dataset_targets_cols; ++j) {
            targets[i][j] = (double)dataset_targets[i * dataset_targets_cols + j];
        }
    }

    // Инициализация смещений и весов
    double **biases = malloc(layer_sizes_rows * sizeof(double*));
    double ***weights = malloc(layer_sizes_rows * sizeof(double**));

    for (int layer_index = 0; layer_index < layer_sizes_rows; ++layer_index) {
        double n_inputs_double = layer_sizes[layer_index * layer_sizes_cols + 0];
        double n_neurons_double = layer_sizes[layer_index * layer_sizes_cols + 1];
        int n_inputs = (int)n_inputs_double;
        int n_neurons = (int)n_neurons_double;

        // Инициализировать смещение
        biases[layer_index] = malloc(n_neurons * sizeof(double));
        double *biases_arr = malloc(n_neurons * sizeof(double));
        biases_arr = init_bias(n_neurons, n_inputs);
        for (int i = 0; i < n_neurons; ++i) {
            biases[layer_index][i] = biases_arr[i];
        }
        free(biases_arr);

        // Инициализировать веса
        double **weights_arr = init_weights(n_neurons, n_inputs);
        weights[layer_index] = malloc(n_inputs * sizeof(double*));
        for (int i = 0; i < n_inputs; ++i) {
            weights[layer_index][i] = malloc(n_neurons * sizeof(double));
            for (int j = 0; j < n_neurons; ++j) {
                weights[layer_index][i][j] = weights_arr[i][j];
            }
        }
        for (int i = 0; i < n_inputs; ++i) {
            free(weights_arr[i]);
        }
        free(weights_arr);
    }

    // Обучение

    // Create Adam
    struct AdamOptimizer *opt = create_adam(learning_rate, 0.9, 0.999, 1e-8, layer_sizes, layer_sizes_rows, layer_sizes_cols);

    double losses_by_epoch[n_epoch];
    for (int epoch = 0; epoch < n_epoch; ++epoch) {
        double *epoch_losses = malloc(dataset_samples_rows * sizeof(double));
        for (int dataset_index = 0; dataset_index < dataset_samples_rows; ++dataset_index) {
            double **sample = malloc(dataset_samples_cols * sizeof(double*));
            for (int i = 0; i < dataset_samples_cols; i++) {
                sample[i] = malloc(dataset_samples_depth * sizeof(double));
                for (int j = 0; j < dataset_samples_depth; j++) {
                    sample[i][j] = samples[dataset_index][i][j];
                }
            }
            double *target = malloc(dataset_targets_cols * sizeof(double));
            for (int i = 0; i < dataset_targets_cols; i++) {
                target[i] = targets[dataset_index][i];
            }

            double n_inputs_double = layer_sizes[0 * layer_sizes_cols + 0];
            double n_neurons_double = layer_sizes[0 * layer_sizes_cols + 1];
            int n_inputs = (int)n_inputs_double;
            int n_neurons = (int)n_neurons_double;

            // Forward pass
            double ***X = malloc(layer_sizes_rows * sizeof(double**));
            double ***Y = malloc(layer_sizes_rows * sizeof(double**));

            double **y = malloc(dataset_samples_cols * sizeof(double*));
            for (int i = 0; i < dataset_samples_cols; i++) {
                y[i] = malloc(n_neurons * sizeof(double));
            }

            matmul(sample, weights[0], y, dataset_samples_cols, dataset_samples_depth, n_inputs, n_neurons, threading, num_cpu);

            for (int i = 0; i < dataset_samples_cols; ++i) {
                for (int j = 0; j < n_neurons; ++j) {
                    y[i][j] += biases[0][i];
                }
            }
            int activation = (int)activations[0];
            apply_activation_calc(y, dataset_samples_cols, n_neurons, activation);
            dropout(y, dataset_samples_cols, n_neurons, keep_prob);

            X[0] = malloc(dataset_samples_cols * sizeof(double*));
            for (int i = 0; i < dataset_samples_cols; i++) {
                X[0][i] = malloc(dataset_samples_depth * sizeof(double));
                for (int j = 0; j < dataset_samples_depth; j++) {
                    X[0][i][j] = sample[i][j];
                }
            }

            for (int i = 0; i < dataset_samples_cols; i++) {
                free(sample[i]);
            }
            free(sample);

            Y[0] = malloc(dataset_samples_cols * sizeof(double*));
            for (int i = 0; i < dataset_samples_cols; i++) {
                Y[0][i] = malloc(n_neurons * sizeof(double));
                for (int j = 0; j < n_neurons; j++) {
                    Y[0][i][j] = y[i][j];
                }
            }
            for (int i = 0; i < dataset_samples_cols; i++) {
                free(y[i]);
            }
            free(y);

            int matrix_rows = dataset_samples_cols;

            for (int layer_index = 1; layer_index < layer_sizes_rows; layer_index++) {
                double n_inputs_double = layer_sizes[layer_index * layer_sizes_cols + 0];
                double n_neurons_double = layer_sizes[layer_index * layer_sizes_cols + 1];
                int n_inputs = (int)n_inputs_double;
                int n_neurons = (int)n_neurons_double;

                double **x = malloc(matrix_rows * sizeof(double*));
                for (int i = 0; i < matrix_rows; i++) {
                    x[i] = malloc(n_inputs * sizeof(double));
                    for (int j = 0; j < n_inputs; j++) {
                        x[i][j] = Y[layer_index - 1][i][j];
                    }
                }

                X[layer_index] = malloc(matrix_rows * sizeof(double*));
                for (int i = 0; i < matrix_rows; i++) {
                    X[layer_index][i] = malloc(n_inputs * sizeof(double));
                    for (int j = 0; j < n_inputs; j++) {
                        X[layer_index][i][j] = x[i][j];
                    }
                }

                double **y = malloc(matrix_rows * sizeof(double*));
                for (int i = 0; i < matrix_rows; i++) {
                    y[i] = malloc(n_neurons * sizeof(double));
                }
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

                Y[layer_index] = malloc(matrix_rows * sizeof(double*));
                for (int i = 0; i < matrix_rows; i++) {
                    Y[layer_index][i] = malloc(n_neurons * sizeof(double));
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

            // Backward pass
            double ***grad_w = malloc(layer_sizes_rows * sizeof(double**));
            double ***grad_x = malloc(layer_sizes_rows * sizeof(double**));
            double **grad_b = malloc(layer_sizes_rows * sizeof(double*));

            n_inputs_double = layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols + 0];
            n_neurons_double = layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols + 1];
            n_inputs = (int)n_inputs_double;
            n_neurons = (int)n_neurons_double;

            double **delta = malloc(matrix_rows * sizeof(double*));
            for (int i = 0; i < matrix_rows; i++) {
                delta[i] = malloc(n_neurons * sizeof(double));
            }
            calc_loss(loss, target, Y[layer_sizes_rows - 1], matrix_rows, n_neurons, delta, regression);
            free(target);
            double output_error = sum(delta, matrix_rows, n_neurons);
            output_error /= matrix_rows + n_neurons;
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

            // Update weights and biases
            
            adam_step(opt, weights, grad_w, layer_sizes, layer_sizes_rows, layer_sizes_cols);

            for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
                double n_inputs_double = layer_sizes[layer_index * layer_sizes_cols + 0];
                double n_neurons_double = layer_sizes[layer_index * layer_sizes_cols + 1];
                int n_inputs = (int)n_inputs_double;
                int n_neurons = (int)n_neurons_double;

                for (int i = 0; i < n_neurons; ++i) {
                    biases[layer_index][i] -= safe_update(grad_b[layer_index][i], learning_rate, max_change);

                    if (isnan(biases[layer_index][i])) {
                        biases[layer_index][i] = 0.0;
                    }
                }
            }

            for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
                double n_inputs_double = layer_sizes[layer_index * layer_sizes_cols + 0];
                int n_inputs = (int)n_inputs_double;

                for (int i = 0; i < n_inputs; i++) {
                    free(grad_w[layer_index][i]);
                }
                for (int i = 0; i < matrix_rows; i++) {
                    free(grad_x[layer_index][i]);
                }
                free(grad_w[layer_index]);
                free(grad_x[layer_index]);
                free(grad_b[layer_index]);
            }
            free(grad_w);
            free(grad_x);
            free(grad_b);
            for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
                for (int i = 0; i < matrix_rows; i++) {
                    free(X[layer_index][i]);
                    free(Y[layer_index][i]);
                }
                free(X[layer_index]);
                free(Y[layer_index]);
            }
            free(X);
            free(Y);
        }
        double mean_loss = mean(epoch_losses, dataset_samples_rows);
        losses_by_epoch[epoch] = mean_loss;
        if (verbose) {
            printf("Epoch %d / %d. Loss: %f\n", epoch + 1, n_epoch, mean_loss);
        }
    }
    char *file_weights = "weights.json";
    save_weights_as_json(file_weights, weights, layer_sizes, layer_sizes_rows, layer_sizes_cols);
    char *file_biases = "biases.json";
    save_biases_as_json(file_biases, biases, layer_sizes, layer_sizes_rows, layer_sizes_cols);

    // Очищаем память

    destroy_adam(opt, layer_sizes, layer_sizes_rows, layer_sizes_cols);

    for (int layer_index = 0; layer_index < layer_sizes_rows; ++layer_index) {
        double n_inputs_double = layer_sizes[layer_index * layer_sizes_cols + 0];
        int n_inputs = (int)n_inputs_double;

        for (int i = 0; i < n_inputs; i++) {
            free(weights[layer_index][i]);
        }
        free(weights[layer_index]);
        free(biases[layer_index]);
    }
    free(biases);
    free(weights);
    for (int dataset_index = 0; dataset_index < dataset_samples_rows; ++dataset_index) {
        for (int i = 0; i < dataset_samples_cols; ++i) {
            free(samples[dataset_index][i]);
        }
        free(samples[dataset_index]);
    }
    free(samples);
    for (int i = 0; i < dataset_targets_rows; ++i) {
        free(targets[i]);
    }
    free(targets);
}

void predict_one(
    double *sample_input,
    int sample_rows,
    int sample_cols,
    double *weights_input,
    double *biases_input,
    double *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols,
    double *activations,
    int activations_len,
    double *prediction,
    int threading,
    int num_cpu) {

    double** sample = malloc(sample_rows * sizeof(double*));
    for (int i = 0; i < sample_rows; ++i) {
        sample[i] = malloc(sample_cols * sizeof(double));
        for (int j = 0; j < sample_cols; ++j) {
            sample[i][j] = sample_input[i + j];
        }
    }

    double ***weights = malloc(layer_sizes_rows * sizeof(double**));
    double **biases = malloc(layer_sizes_rows * sizeof(double*));

    int current_weight_offset = 0;
    int total_bias_count = 0;
    for (int layer_index = 0; layer_index < layer_sizes_rows; ++layer_index) {
        double n_inputs_double = layer_sizes[layer_index * layer_sizes_cols + 0];
        double n_neurons_double = layer_sizes[layer_index * layer_sizes_cols + 1];
        int n_inputs = (int)n_inputs_double;
        int n_neurons = (int)n_neurons_double;

        weights[layer_index] = malloc(n_inputs * sizeof(double*));
        for (int i = 0; i < n_inputs; ++i) {
            weights[layer_index][i] = malloc(n_neurons * sizeof(double));
            for (int j = 0; j < n_neurons; ++j) {
                int index = current_weight_offset + i * n_neurons + j;
                weights[layer_index][i][j] = weights_input[index];
            }
        }

        biases[layer_index] = malloc(n_neurons * sizeof(double));
        for (int i = 0; i < n_neurons; ++i) {
            int index = total_bias_count + i;
            biases[layer_index][i] = biases_input[index];
        }
        
        current_weight_offset += n_inputs * n_neurons;
        total_bias_count += n_neurons;
    }

    double n_inputs_double = layer_sizes[0 * layer_sizes_cols + 0];
    double n_neurons_double = layer_sizes[0 * layer_sizes_cols + 1];
    int n_inputs = (int)n_inputs_double;
    int n_neurons = (int)n_neurons_double;

    // Forward pass
    double ***Y = malloc(layer_sizes_rows * sizeof(double**));

    double **y = malloc(sample_rows * sizeof(double*));
    for (int i = 0; i < sample_rows; i++) {
        y[i] = malloc(n_neurons * sizeof(double));
    }

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

    Y[0] = malloc(sample_rows * sizeof(double*));
    for (int i = 0; i < sample_rows; i++) {
        Y[0][i] = malloc(n_neurons * sizeof(double));
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

        double **x = malloc(matrix_rows * sizeof(double*));
        for (int i = 0; i < matrix_rows; i++) {
            x[i] = malloc(n_inputs * sizeof(double));
            for (int j = 0; j < n_inputs; j++) {
                x[i][j] = Y[layer_index - 1][i][j];
            }
        }

        double **y = malloc(matrix_rows * sizeof(double*));
        for (int i = 0; i < matrix_rows; i++) {
            y[i] = malloc(n_neurons * sizeof(double));
        }
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

        Y[layer_index] = malloc(matrix_rows * sizeof(double*));
        for (int i = 0; i < matrix_rows; i++) {
            Y[layer_index][i] = malloc(n_neurons * sizeof(double));
            for (int j = 0; j < n_neurons; j++) {
                Y[layer_index][i][j] = y[i][j];
            }
        }
        for (int i = 0; i < matrix_rows; i++) {
            free(y[i]);
        }
        free(y);
    }

    n_inputs_double = layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols + 0];
    n_neurons_double = layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols + 1];
    n_inputs = (int)n_inputs_double;
    n_neurons = (int)n_neurons_double;

    y = malloc(matrix_rows * sizeof(double*));
    for (int i = 0; i < matrix_rows; i++) {
        y[i] = malloc(n_neurons * sizeof(double));
        for (int j = 0; j < n_neurons; j++) {
            y[i][j] = Y[layer_sizes_rows - 1][i][j];
        }
    }

    // Return predict
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < n_neurons; j++) {
            prediction[j] = y[i][j];
        }
    }
    for (int i = 0; i < matrix_rows; i++) {
        free(y[i]);
    }
    free(y);
    
    // Очищаем память
    for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
        for (int i = 0; i < matrix_rows; i++) {
            free(Y[layer_index][i]);
        }
        free(Y[layer_index]);
    }
    free(Y);
    for (int layer_index = 0; layer_index < layer_sizes_rows; ++layer_index) {
        double n_inputs_double = layer_sizes[layer_index * layer_sizes_cols + 0];
        int n_inputs = (int)n_inputs_double;

        for (int i = 0; i < n_inputs; i++) {
            free(weights[layer_index][i]);
        }
        free(weights[layer_index]);
        free(biases[layer_index]);
    }
    free(weights);
    free(biases);
}

void predict(
    double *dataset_samples,
    int dataset_samples_rows,
    int dataset_samples_cols,
    int dataset_samples_depth,
    double *weights_input,
    double *biases_input,
    double *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols,
    double *activations,
    int activations_len,
    double *predictions,
    int threading,
    int num_cpu) {

    // Загрузка датасета
    double*** samples = malloc(dataset_samples_rows * sizeof(double**));
    for (int dataset_index = 0; dataset_index < dataset_samples_rows; ++dataset_index) {
        samples[dataset_index] = malloc(dataset_samples_cols * sizeof(double*));
        for (int i = 0; i < dataset_samples_cols; ++i) {
            samples[dataset_index][i] = malloc(dataset_samples_depth * sizeof(double));
            for (int j = 0; j < dataset_samples_depth; ++j) {
                samples[dataset_index][i][j] = (double)dataset_samples[dataset_index * dataset_samples_cols * dataset_samples_depth + i * dataset_samples_depth + j];
            }
        }
    }

    double ***weights = malloc(layer_sizes_rows * sizeof(double**));
    double **biases = malloc(layer_sizes_rows * sizeof(double*));

    int current_weight_offset = 0;
    int total_bias_count = 0;
    for (int layer_index = 0; layer_index < layer_sizes_rows; ++layer_index) {
        double n_inputs_double = layer_sizes[layer_index * layer_sizes_cols + 0];
        double n_neurons_double = layer_sizes[layer_index * layer_sizes_cols + 1];
        int n_inputs = (int)n_inputs_double;
        int n_neurons = (int)n_neurons_double;

        weights[layer_index] = malloc(n_inputs * sizeof(double*));
        for (int i = 0; i < n_inputs; ++i) {
            weights[layer_index][i] = malloc(n_neurons * sizeof(double));
            for (int j = 0; j < n_neurons; ++j) {
                int index = current_weight_offset + i * n_neurons + j;
                weights[layer_index][i][j] = weights_input[index];
            }
        }

        biases[layer_index] = malloc(n_neurons * sizeof(double));
        for (int i = 0; i < n_neurons; ++i) {
            int index = total_bias_count + i;
            biases[layer_index][i] = biases_input[index];
        }
        
        current_weight_offset += n_inputs * n_neurons;
        total_bias_count += n_neurons;
    }

    for (int dataset_index = 0; dataset_index < dataset_samples_rows; ++dataset_index) {
        double **sample = malloc(dataset_samples_cols * sizeof(double*));
        for (int i = 0; i < dataset_samples_cols; i++) {
            sample[i] = malloc(dataset_samples_depth * sizeof(double));
            for (int j = 0; j < dataset_samples_depth; j++) {
                sample[i][j] = samples[dataset_index][i][j];
            }
        }

        double n_inputs_double = layer_sizes[0 * layer_sizes_cols + 0];
        double n_neurons_double = layer_sizes[0 * layer_sizes_cols + 1];
        int n_inputs = (int)n_inputs_double;
        int n_neurons = (int)n_neurons_double;

        // Forward pass
        double ***Y = malloc(layer_sizes_rows * sizeof(double**));

        double **y = malloc(dataset_samples_cols * sizeof(double*));
        for (int i = 0; i < dataset_samples_cols; i++) {
            y[i] = malloc(n_neurons * sizeof(double));
        }

        matmul(sample, weights[0], y, dataset_samples_cols, dataset_samples_depth, n_inputs, n_neurons, threading, num_cpu);

        for (int i = 0; i < dataset_samples_cols; ++i) {
            for (int j = 0; j < n_neurons; ++j) {
                y[i][j] += biases[0][i];
            }
        }
        int activation = (int)activations[0];
        apply_activation_calc(y, dataset_samples_cols, n_neurons, activation);

        for (int i = 0; i < dataset_samples_cols; i++) {
            free(sample[i]);
        }
        free(sample);

        Y[0] = malloc(dataset_samples_cols * sizeof(double*));
        for (int i = 0; i < dataset_samples_cols; i++) {
            Y[0][i] = malloc(n_neurons * sizeof(double));
            for (int j = 0; j < n_neurons; j++) {
                Y[0][i][j] = y[i][j];
            }
        }
        for (int i = 0; i < dataset_samples_cols; i++) {
            free(y[i]);
        }
        free(y);

        int matrix_rows = dataset_samples_cols;
        for (int layer_index = 1; layer_index < layer_sizes_rows; layer_index++) {
            double n_inputs_double = layer_sizes[layer_index * layer_sizes_cols + 0];
            double n_neurons_double = layer_sizes[layer_index * layer_sizes_cols + 1];
            int n_inputs = (int)n_inputs_double;
            int n_neurons = (int)n_neurons_double;

            double **x = malloc(matrix_rows * sizeof(double*));
            for (int i = 0; i < matrix_rows; i++) {
                x[i] = malloc(n_inputs * sizeof(double));
                for (int j = 0; j < n_inputs; j++) {
                    x[i][j] = Y[layer_index - 1][i][j];
                }
            }

            double **y = malloc(matrix_rows * sizeof(double*));
            for (int i = 0; i < matrix_rows; i++) {
                y[i] = malloc(n_neurons * sizeof(double));
            }
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

            Y[layer_index] = malloc(matrix_rows * sizeof(double*));
            for (int i = 0; i < matrix_rows; i++) {
                Y[layer_index][i] = malloc(n_neurons * sizeof(double));
                for (int j = 0; j < n_neurons; j++) {
                    Y[layer_index][i][j] = y[i][j];
                }
            }
            for (int i = 0; i < matrix_rows; i++) {
                free(y[i]);
            }
            free(y);
        }

        n_inputs_double = layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols + 0];
        n_neurons_double = layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols + 1];
        n_inputs = (int)n_inputs_double;
        n_neurons = (int)n_neurons_double;

        y = malloc(matrix_rows * sizeof(double*));
        for (int i = 0; i < matrix_rows; i++) {
            y[i] = malloc(n_neurons * sizeof(double));
            for (int j = 0; j < n_neurons; j++) {
                y[i][j] = Y[layer_sizes_rows - 1][i][j];
            }
        }

        for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
            for (int i = 0; i < matrix_rows; i++) {
                free(Y[layer_index][i]);
            }
            free(Y[layer_index]);
        }
        free(Y);

        // Return predict
        for (int i = 0; i < n_neurons; i++) {
            predictions[dataset_index * n_neurons + i] = y[0][i];
        }

        for (int i = 0; i < matrix_rows; i++) {
            free(y[i]);
        }
        free(y);
    }

    // Очищаем память
    for (int layer_index = 0; layer_index < layer_sizes_rows; ++layer_index) {
        double n_inputs_double = layer_sizes[layer_index * layer_sizes_cols + 0];
        int n_inputs = (int)n_inputs_double;

        for (int i = 0; i < n_inputs; i++) {
            free(weights[layer_index][i]);
        }
        free(weights[layer_index]);
        free(biases[layer_index]);
    }
    free(weights);
    free(biases);
    for (int dataset_index = 0; dataset_index < dataset_samples_rows; ++dataset_index) {
        for (int i = 0; i < dataset_samples_cols; ++i) {
            free(samples[dataset_index][i]);
        }
        free(samples[dataset_index]);
    }
    free(samples);
}
