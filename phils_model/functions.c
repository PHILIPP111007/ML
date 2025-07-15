// gcc -shared -o functions.so -fPIC -O3 functions.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

///////////////////////////////////////////////////////////////////////////////

void matmul(double **first, double **second, double **result_matrix, int rows_first, int cols_first, int rows_second, int cols_second) {
    if (cols_first != rows_second) {
        printf("cols first != rows second!\n");
        return;
    }

    for (int i = 0; i < rows_first; i++) {
        result_matrix[i] = malloc(cols_second * sizeof(double));
        for (int j = 0; j < cols_second; j++) {
            result_matrix[i][j] = 0.0;
            double *arr = malloc(cols_first * sizeof(double));
            for (int k = 0; k < cols_first; k++) {
                result_matrix[i][j] += first[i][k] * second[k][j];
            }
        }
    }
}

double **transpose(double **original_matrix, int rows, int cols) {
    double **transposed_matrix = (double**)malloc(cols * sizeof(double*));
    for (int i = 0; i < cols; i++) {
        transposed_matrix[i] = (double*)malloc(rows * sizeof(double));
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            transposed_matrix[j][i] = original_matrix[i][j];
        }
    }
    return transposed_matrix;
}

double sum(double **matrix, int rows, int cols) {
    double n = 0.0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            n += matrix[i][j];
        }
    }
    return n;
}

double *sum_axis_0(double **matrix, int rows, int cols) {
    double *result = malloc(cols * sizeof(double));

    for (int j = 0; j < cols; ++j) {
        result[j] = 0;
        for (int i = 0; i < rows; ++i) {
            result[j] += matrix[i][j];
        }
    }
    return result;
}

double mean(double *arr, int len) {
    if (len == 0) {
        return 0.0;
    }
    double sum = 0.0;
    for (int i = 0; i < len; ++i) {
        sum += arr[i];
    }
    return sum / len;
}

double safe_weight_update(double delta, double learning_rate, double max_change) {
    double change = delta * learning_rate;
    if (change > max_change) {
        change = max_change;
    }
    else if (change < -max_change) {
        change = -max_change;
    }

    return change;
}

///////////////////////////////////////////////////////////////////////////////
// Activation functions
///////////////////////////////////////////////////////////////////////////////

void relu_calc(double **y, int matrix_rows, int matrix_columns) {
    for (int i = 0; i < matrix_rows; ++i) {
        for (int j = 0; j < matrix_columns; ++j) {
            if (y[i] > 0) {
                y[i][j] = y[i][j];
            }
            else {
                y[i][j] = 0.0;
            }
        }
    }
}

void relu_derivative(double **y, int matrix_rows, int matrix_columns) {
    for (int i = 0; i < matrix_rows; ++i) {
        for (int j = 0; j < matrix_columns; ++j) {
            if (y[i] > 0) {
                y[i][j] = y[i][j];
            }
            else {
                y[i][j] = 0.0;
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

double sigmoid(double x) {
    double n = exp(x);
    if (x >= 0) {
        return 1.0 / (1.0 + n);
    } else {
        return n / (1.0 + n);
    }
}

void sigmoid_calc(double **y, int matrix_rows, int matrix_columns) {
    for (int i = 0; i < matrix_rows; ++i) {
        for (int j = 0; j < matrix_columns; ++j) {
            y[i][j] = sigmoid(y[i][j]);
        }
    }
}

void sigmoid_derivative(double **y, int matrix_rows, int matrix_columns) {
    double **f = malloc(matrix_rows * sizeof(double*));

    for (int i = 0; i < matrix_rows; ++i) {
        f[i] = malloc(matrix_columns * sizeof(double));
        for (int j = 0; j < matrix_columns; ++j) {
            f[i][j] = sigmoid(y[i][j]);
        }
    }

    for (int i = 0; i < matrix_rows; ++i) {
        for (int j = 0; j < matrix_columns; ++j) {
            y[i][j] = f[i][j] * (1.0 - f[i][j]);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

// Метод softmax (возвращает нормализованные вероятности классов).
void softmax_calc(double **y, int matrix_rows, int matrix_columns) {
    double max_val = y[0][0];

    for (int i = 1; i < matrix_rows; ++i) {
        for (int j = 1; j < matrix_columns; ++j) {
            if (y[i][j] > max_val) {
                max_val = y[i][j];
            }
        }
    }

    // Отнимем максимум от каждого элемента для стабилизации экспоненты.
    double sum_exp = 0.0;
    for (int i = 0; i < matrix_rows; ++i) {
        for (int j = 0; j < matrix_columns; ++j) {
            y[i][j] = exp(y[i][j] - max_val);
            sum_exp += y[i][j];
        }
    }

    // Нормализация путем деления каждого элемента на сумму экспонент.
    for (int i = 0; i < matrix_rows; ++i) {
        for (int j = 0; j < matrix_columns; ++j) {
            y[i][j] /= sum_exp;
        }
    }
}

void softmax_derivative(double **y, int matrix_rows, int matrix_columns) {
    for (int i = 0; i < matrix_rows; ++i) {
        for (int j = 0; j < matrix_columns; ++j) {
            y[i][j] = y[i][j];
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// Loss functions
///////////////////////////////////////////////////////////////////////////////

void mse_loss(double **prediction, int prediction_rows, int prediction_cols, double *target, double **output_error) {
    double **loss = malloc(prediction_rows * sizeof(double*));

    for (int i = 0; i < prediction_rows; ++i) {
        loss[i] =  malloc(prediction_cols * sizeof(double));
        for (int j = 0; j < prediction_cols; ++j) {
            loss[i][j] = target[j] - prediction[i][j];
        }
    }

    for (int i = 0; i < prediction_rows; ++i) {
        for (int j = 0; j < prediction_cols; ++j) {
            output_error[i][j] = loss[i][j];
        }
        free(loss[i]);
    }
    free(loss);
}

void cross_entropy_loss(double **prediction, int prediction_rows, int prediction_cols, double *target, double **output_error) {
    double **loss = malloc(prediction_rows * sizeof(double*));

    for (int i = 0; i < prediction_rows; ++i) {
        loss[i] =  malloc(prediction_cols * sizeof(double));
        for (int j = 0; j < prediction_cols; ++j) {
            double p = prediction[i][j] > 1e-15 ? prediction[i][j] : 1e-15;
            loss[i][j] = target[j] * log(p);
        }
    }

    for (int i = 0; i < prediction_rows; ++i) {
        double *arr = malloc(prediction_cols * sizeof(double));
        for (int j = 0; j < prediction_cols; ++j) {
            output_error[i][j] = loss[i][j];
        }
    }
    
    for (int i = 0; i < prediction_rows; ++i) {
        free(loss[i]);
    }
    free(loss);
}

///////////////////////////////////////////////////////////////////////////////

double *init_bias(int n_neurons) {
    double* bias = malloc(n_neurons * sizeof(double));

    for (int i = 0; i < n_neurons; ++i) {
        bias[i] = (double)rand() / RAND_MAX;
    }
    return bias;
}

double **init_weights(int n_neurons, int n_inputs) {
    double** weights = malloc(n_inputs * sizeof(double*));

    for (int i = 0; i < n_inputs; i++) {
        weights[i] = malloc(n_neurons * sizeof(double));
    }

    // Генерация случайных чисел с нормальным распределением
    for (int i = 0; i < n_inputs; i++) {
        for (int j = 0; j < n_neurons; j++) {
            double u1 = (double)rand() / RAND_MAX;
            double u2 = (double)rand() / RAND_MAX;
            double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
            weights[i][j] = (double)z;
        }
    }
    return weights;
}

///////////////////////////////////////////////////////////////////////////////

void apply_activation_calc(double **y, int matrix_rows, int matrix_columns, int activation) {
    if (activation == 0) {
        relu_calc(y, matrix_rows, matrix_columns);
    } else if (activation == 1) {
        sigmoid_calc(y, matrix_rows, matrix_columns);
    } else if (activation == 2) {
        softmax_calc(y, matrix_rows, matrix_columns);
    } else if (activation == 3) {
        return;
    }
}

void apply_activation_derivative(double **y, int matrix_rows, int matrix_columns, int activation) {
    if (activation == 0) {
        relu_derivative(y, matrix_rows, matrix_columns);
    } else if (activation == 1) {
        sigmoid_derivative(y, matrix_rows, matrix_columns);
    } else if (activation == 2) {
        softmax_derivative(y, matrix_rows, matrix_columns);
    } else if (activation == 3) {
        return;
    }
}

void calc_loss(int loss, double *target, double **prediction, int prediction_rows, int prediction_cols, double **output_error) {
    if (loss == 0) {
        return mse_loss(prediction, prediction_rows, prediction_cols, target, output_error);
    } else if (loss == 1) {
        return cross_entropy_loss(prediction, prediction_rows, prediction_cols, target, output_error);
    }
}

///////////////////////////////////////////////////////////////////////////////

void save_weights_as_json(char *fname, double ***weights_result, double *layer_sizes, int layer_sizes_rows, int layer_sizes_cols) {
    FILE *fp = fopen(fname, "w");

    if (!fp) {
        fprintf(stderr, "Ошибка открытия файла '%s'\\n", fname);
        return;
    }

    fprintf(fp, "[\n");

    for (int layer_size = 0; layer_size < layer_sizes_rows; ++layer_size) { 
        double n_inputs_double = layer_sizes[layer_size * layer_sizes_cols + 0];
        double n_neurons_double = layer_sizes[layer_size * layer_sizes_cols + 1];
        int n_inputs = (int)n_inputs_double;
        int n_neurons = (int)n_neurons_double;

        fprintf(fp, "\t[\n");
        for (int i = 0; i < n_inputs; i++) {
            fprintf(fp, "\t\t[\n");

            fprintf(fp, "\t\t\t");
            for (int j = 0; j < n_neurons; j++) {
                if (j != 0) fprintf(fp, "\t\t\t");
                if (j == n_neurons - 1) {
                    fprintf(fp, "%f\n", weights_result[layer_size][i][j]);
                } else {
                    fprintf(fp, "%f,\n", weights_result[layer_size][i][j]);
                }
            }

            fprintf(fp, "\t\t]");
            if (i != n_inputs - 1) fprintf(fp, ",");
            fprintf(fp, "\n");
        }

        if (layer_size != layer_sizes_rows - 1) {
            fprintf(fp, "\t],\n");
        } else {
            fprintf(fp, "\t]\n");
        }
    }
    fprintf(fp, "]");
    fclose(fp);
}

void save_biases_as_json(char *fname, double **biases, double *layer_sizes, int layer_sizes_rows, int layer_sizes_cols) {
    FILE *fp = fopen(fname, "w");

    if (!fp) {
        fprintf(stderr, "Ошибка открытия файла '%s'\\n", fname);
        return;
    }

    fprintf(fp, "[");

    for (int layer_size = 0; layer_size < layer_sizes_rows; ++layer_size) { 
        double n_neurons_double = layer_sizes[layer_size * layer_sizes_cols + 1];
        int n_neurons = (int)n_neurons_double;

        fprintf(fp, "[");
        for (size_t neuron = 0; neuron < n_neurons; neuron++) {
            fprintf(fp, "[");
            fprintf(fp, "%f", biases[layer_size][neuron]);
            fprintf(fp, "]");
            if (neuron != n_neurons - 1) fprintf(fp, ",");
        }

        if (layer_size != layer_sizes_rows - 1) {
            fprintf(fp, "],");
        } else {
            fprintf(fp, "]");
        }
    }

    fprintf(fp, "]");
    fclose(fp);
}

///////////////////////////////////////////////////////////////////////////////

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
    double max_change) {

    // Загрузка датасета
    double*** samples = malloc(dataset_samples_rows * sizeof(double**));
    double** targets = malloc(dataset_targets_rows * sizeof(double*));

    for (int dataset_index = 0; dataset_index < dataset_samples_rows; ++dataset_index) {
        samples[dataset_index] = malloc(dataset_samples_cols * sizeof(double*));
        for (int i = 0; i < dataset_samples_cols; ++i) {
            samples[dataset_index][i] = malloc(dataset_samples_depth * sizeof(double));
            for (int j = 0; j < dataset_samples_depth; ++j) {
                samples[dataset_index][i][j] = (double)dataset_samples[dataset_index * dataset_samples_cols * dataset_samples_depth + i * dataset_samples_depth + j];
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
        biases_arr = init_bias(n_neurons);
        for (int i = 0; i < n_neurons; ++i) {
            biases[layer_index][i] = biases_arr[i];
        }

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
        free(biases_arr);
    }

    // Обучение
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

            matmul(sample, weights[0], y, dataset_samples_cols, dataset_samples_depth, n_inputs, n_neurons);

            for (int i = 0; i < dataset_samples_cols; ++i) {
                for (int j = 0; j < n_neurons; ++j) {
                    y[i][j] += biases[0][i];
                }
            }
            int activation = (int)activations[0];
            apply_activation_calc(y, dataset_samples_cols, n_neurons, activation);

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

            int matrix_rows = dataset_samples_cols;
            int matrix_columns = n_inputs;

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
                matmul(x, weights[layer_index], y, matrix_rows, n_inputs, n_inputs, n_neurons);
                for (int i = 0; i < matrix_rows; i++) {
                    free(x[i]);
                }

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

                matrix_rows = matrix_rows;
                matrix_columns = n_inputs;
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
            calc_loss(loss, target, Y[layer_sizes_rows - 1], matrix_rows, n_neurons, delta);
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
            matmul(x_T, delta, w, n_inputs, matrix_rows, matrix_rows, n_neurons);
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
            free(w);
            for (int i = 0; i < matrix_rows; i++) {
                free(x[i]);
            }

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
            matmul(delta, w_T, result, matrix_rows, n_neurons, n_neurons, n_inputs);
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
            for (int i = 0; i < n_neurons; i++) {
                free(w_T[i]);
            }
            for (int i = 0; i < matrix_rows; i++) {
                free(result[i]);
                free(delta[i]);
            }

            matrix_rows = matrix_rows;
            matrix_columns = n_inputs;

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
                double **result = malloc(matrix_rows * sizeof(double*));
                for (int i = 0; i < matrix_rows; i++) {
                    result[i] = malloc(n_neurons * sizeof(double));
                    for (int j = 0; j < n_neurons; j++) {
                        result[i][j] = grad[i][j] * y[i][j];
                    }
                }
                for (int i = 0; i < matrix_rows; i++) {
                    free(grad[i]);
                    free(y[i]);
                }

                double **delta = malloc(matrix_rows * sizeof(double*));
                for (int i = 0; i < matrix_rows; i++) {
                    delta[i] = malloc(n_neurons * sizeof(double));
                    for (int j = 0; j < n_neurons; j++) {
                        delta[i][j] = result[i][j];
                    }
                }
                for (int i = 0; i < matrix_rows; i++) {
                    free(result[i]);
                }

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
                double **w = malloc(n_inputs * sizeof(double*));
                for (int i = 0; i < n_inputs; i++) {
                    w[i] = malloc(n_neurons * sizeof(double));
                }
                matmul(x_T, delta, w, n_inputs, matrix_rows, matrix_rows, n_neurons);
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
                double **result_grad_x = malloc(matrix_rows * sizeof(double*));
                for (int i = 0; i < matrix_rows; i++) {
                    result_grad_x[i] = malloc(n_inputs * sizeof(double));
                }
                matmul(delta, w_T, result_grad_x, matrix_rows, n_neurons, n_neurons, n_inputs);
                for (int i = 0; i < n_neurons; i++) {
                    free(w_T[i]);
                }
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

                matrix_rows = matrix_rows;
                matrix_columns = n_inputs;
            }

            // Update weights
            for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
                double n_inputs_double = layer_sizes[layer_index * layer_sizes_cols + 0];
                double n_neurons_double = layer_sizes[layer_index * layer_sizes_cols + 1];
                int n_inputs = (int)n_inputs_double;
                int n_neurons = (int)n_neurons_double;

                for (int i = 0; i < n_inputs; i++) {
                    for (int j = 0; j < n_neurons; j++) {
                        double change = safe_weight_update(grad_w[layer_index][i][j], learning_rate, max_change);
                        weights[layer_index][i][j] -= change;
                        // printf("%f\n", change);
                    }
                }
                for (int i = 0; i < n_neurons; ++i) {
                    biases[layer_index][i] -= grad_b[layer_index][i] * learning_rate;
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
            }
            free(grad_w);
            free(grad_x);
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
            free(target);
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

void predict(
    double *sample_input,
    int sample_rows,
    int sample_cols,
    double *biases_input,
    int biases_rows,
    double *weights_input,
    int weights_rows,
    int weights_cols,
    double *layer_size,
    int layer_size_rows,
    int activation,
    double *prediction) {

    double n_inputs_double = layer_size[0];
    double n_neurons_double = layer_size[1];
    int n_inputs = (int)n_inputs_double;
    int n_neurons = (int)n_neurons_double;

    double **sample = malloc(sample_rows * sizeof(double*));
    for (int i = 0; i < sample_rows; ++i) {
        sample[i] = malloc(sample_cols * sizeof(double));
        for (int j = 0; j < sample_cols; ++j) {
            sample[i][j] = sample_input[i * sample_cols + j];
        }
    }

    double* biases = malloc(biases_rows * sizeof(double));
    double** weights = malloc(weights_rows * sizeof(double*));

    for (int i = 0; i < weights_rows; ++i) {
        weights[i] = malloc(weights_cols * sizeof(double));
        for (int j = 0; j < weights_cols; ++j) {
            weights[i][j] = (double)weights_input[i * weights_cols + j];
        }
    }

    for (int i = 0; i < biases_rows; ++i) {
        biases[i] = (double)biases_input[i];
    }

    // Forward pass
    double **result = malloc(sample_rows * sizeof(double*));
    matmul(sample, weights, result, sample_rows, sample_cols, n_inputs, n_neurons);

    // Add bias
    for (int i = 0; i < sample_rows; ++i) {
        for (int j = 0; j < n_neurons; ++j) {
            result[i][j] += biases[i];
        }
    }

    // Apply activation function
    apply_activation_calc(result, sample_rows, n_neurons, activation);

    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < n_neurons; j++) {
            prediction[j] = result[i][j];
        }
    }

    // Очищаем память
    for (int i = 0; i < sample_rows; ++i) {
        free(sample[i]);
    }
    free(sample);
    for (int i = 0; i < weights_rows; i++) {
        free(weights[i]);
    }
    free(weights);
    for (int i = 0; i < sample_rows; ++i) {
        free(result[i]);
    }
    free(result);
    free(biases);
}
