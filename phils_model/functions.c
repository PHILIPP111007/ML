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
            free(arr);
        }
    }
}

double **transpose(double **original_matrix, int rows, int cols) {
    // Создаем новую матрицу с размерами равными числу столбцов и строк оригинальной матрицы
    double **transposed_matrix = (double**)malloc(cols * sizeof(double*));
    for (int i = 0; i < cols; i++) {
        transposed_matrix[i] = (double*)malloc(rows * sizeof(double)); // Строки новой матрицы будут иметь длину old_cols
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            transposed_matrix[j][i] = original_matrix[i][j];
        }
    }

    return transposed_matrix;
}

// ///////////////////////////////////////////////////////////////////////////////

double calculate_mean(double *arr, int len) {
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

    if (change > max_change)
        change = max_change;
    else if (change < -max_change)
        change = -max_change;

    return change;
}

int argmax(double* output, int size) {
    if (size <= 0)
        return -1; // Если размер массива некорректный

    double max_value = output[0];
    int index_max = 0;
    
    for (int i = 1; i < size; ++i) {
        if (output[i] > max_value) {
            max_value = output[i];
            index_max = i;
        }
    }
    return index_max;
}

///////////////////////////////////////////////////////////////////////////////
// Activation functions
///////////////////////////////////////////////////////////////////////////////

void relu_calc(double **y, int matrix_rows, int matrix_columns, double **output_data) {
    for (int i = 0; i < matrix_rows; ++i) {
        // Применяем max(0, x), аналог np.maximum(0, x) из Python.
        for (int j = 0; j < matrix_columns; ++j) {
            if (y[i] > 0) {
                output_data[i][j] = y[i][j];
            }
            else {
                output_data[i][j] = 0.0f;
            }
        }
    }
}

void relu_derivative(double **y, int matrix_rows, int matrix_columns, double **output_data) {
    for (int i = 0; i < matrix_rows; ++i) {
        // Применяем max(0, x), аналог np.maximum(0, x) из Python.
        for (int j = 0; j < matrix_columns; ++j) {
            if (y[i] > 0) {
                output_data[i][j] = y[i][j];
            }
            else {
                output_data[i][j] = 0.0f;
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

// Функция сигмоида (Sigmoid activation function).
double sigmoid(double x) {
    // Используем оптимизированную формулу расчета сигмоиды.
    double n = exp(x);
    if (x >= 0) {
        return 1.0 / (1.0 + n);
    } else {
        return n / (1.0 + n);
    }
}

// Расчет сигмоидной активации для всех элементов выборки.
void sigmoid_calc(double **y, int matrix_rows, int matrix_columns, double **output_data) {
    for (int i = 0; i < matrix_rows; ++i) {
        for (int j = 0; j < matrix_columns; ++j) {
            output_data[i][j] = sigmoid(y[i][j]);
        }
    }
}

// Расчёт производной сигмоиды.
void sigmoid_derivative(double **y, int matrix_rows, int matrix_columns, double **output_data) {
    for (int i = 0; i < matrix_rows; ++i) {
        for (int j = 0; j < matrix_columns; ++j) {
            // Формула производной сигмоиды: f'(x) = f(x)*(1-f(x)).
            output_data[i][j] = y[i][j] * (1.0 - y[i][j]);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

// Метод softmax (возвращает нормализованные вероятности классов).
void softmax_calc(double **y, int matrix_rows, int matrix_columns, double **output_data) {
    // Найдем максимальное значение среди элементов массива.
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
            output_data[i][j] = exp(y[i][j] - max_val);
            sum_exp += output_data[i][j];
        }
    }

    // Нормализация путем деления каждого элемента на сумму экспонент.
    for (int i = 0; i < matrix_rows; ++i) {
        for (int j = 0; j < matrix_columns; ++j) {
            output_data[i][j] /= sum_exp;
        }
    }
}

void softmax_derivative(double **y, int matrix_rows, int matrix_columns, double **output_data) {
    for (int i = 0; i < matrix_rows; ++i) {
        for (int j = 0; j < matrix_columns; ++j) {
            output_data[i][j] = y[i][j];
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// Loss functions
///////////////////////////////////////////////////////////////////////////////

void mse_loss(double **prediction, int prediction_rows, int prediction_cols, double *target, double **output_error) {
    double **loss = malloc(prediction_rows * sizeof(double*));

    for (int i = 0; i < prediction_rows; ++i) {
        double *arr = malloc(prediction_cols * sizeof(double));
        int max_prediction_index = argmax(prediction[i], prediction_cols);

        for (int j = 0; j < prediction_cols; ++j) {
            arr[j] = target[j] - prediction[i][j];
        }
        loss[i] = arr;
    }

    for (int i = 0; i < prediction_rows; ++i) {
        double *arr = malloc(prediction_cols * sizeof(double));
        for (int j = 0; j < prediction_cols; ++j) {
            arr[j] = loss[i][j];
        }
        output_error[i] = arr;
    }

    for (int i = 0; i < prediction_rows; ++i) {
        free(loss[i]);
    }
    free(loss);
}

void cross_entropy_loss(double **prediction, int prediction_rows, int prediction_cols, double *target, double **output_error) {
    double **loss = malloc(prediction_rows * sizeof(double*));

    for (int i = 0; i < prediction_rows; ++i) {
        double *arr = malloc(prediction_cols * sizeof(double));
        int max_prediction_index = argmax(prediction[i], prediction_cols);

        for (int j = 0; j < prediction_cols; ++j) {
            double p = prediction[i][j] > 1e-15 ? prediction[i][j] : 1e-15;
            arr[j] = target[j] * log(p);
        }
        loss[i] = arr;
    }

    for (int i = 0; i < prediction_rows; ++i) {
        double *arr = malloc(prediction_cols * sizeof(double));
        for (int j = 0; j < prediction_cols; ++j) {
            arr[j] = loss[i][j];
        }
        output_error[i] = arr;
    }
    
    for (int i = 0; i < prediction_rows; ++i) {
        free(loss[i]);
    }
    free(loss);
}

///////////////////////////////////////////////////////////////////////////////

// Функция инициализации смещений
double* init_bias(int n_neurons) {
    double* bias = malloc(n_neurons * sizeof(double)); // Выделяем память для строк

    for (int i = 0; i < n_neurons; ++i) {
        bias[i] = (double)rand() / RAND_MAX;
    }
    return bias;
}

// Инициализация матриц весов
double** init_weights(int n_neurons, int n_inputs) {
    // Создание динамического двумерного массива
    double** weights = malloc(n_inputs * sizeof(double*)); // Выделяем память для строк

    for (int i = 0; i < n_inputs; i++) {
        weights[i] = malloc(n_neurons * sizeof(double)); // Выделяем память для каждого столбца строки
    }

    // Генерация случайных чисел с нормальным распределением
    for (int i = 0; i < n_inputs; i++) {
        for (int j = 0; j < n_neurons; j++) {
            // Генерируем случайное нормальное распределение (среднее=0, стандартное отклонение=1)
            // Простое приближённое решение с использованием Box-Muller преобразования
            double u1 = (double)rand() / RAND_MAX;
            double u2 = (double)rand() / RAND_MAX;
            double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
            weights[i][j] = (double)z * 10.0;
        }
    }
    return weights;
}

///////////////////////////////////////////////////////////////////////////////

void apply_activation_calc(double **y, int matrix_rows, int matrix_columns, int activation) {
    if (activation == 0) {
        relu_calc(y, matrix_rows, matrix_columns, y);
    } else if (activation == 1) {
        sigmoid_calc(y, matrix_rows, matrix_columns, y);
    } else if (activation == 2) {
        softmax_calc(y, matrix_rows, matrix_columns, y);
    } else if (activation == 3) {
        return;
    }
}

void apply_activation_derivative(double **y, int matrix_rows, int matrix_columns, int activation) {
    if (activation == 0) {
        relu_derivative(y, matrix_rows, matrix_columns, y);
    } else if (activation == 1) {
        sigmoid_derivative(y, matrix_rows, matrix_columns, y);
    } else if (activation == 2) {
        softmax_derivative(y, matrix_rows, matrix_columns, y);
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
        double *biases_arr = malloc(n_neurons * sizeof(double));  // TODO
        biases_arr = init_bias(n_neurons);
        biases[layer_index] = biases_arr;

        // Инициализировать веса
        double **weights_arr = malloc(n_inputs * sizeof(double*));
        weights_arr = init_weights(n_neurons, n_inputs);
        weights[layer_index] = weights_arr;
    }

    // Обучение
    double losses_by_epoch[n_epoch];
    for (int epoch = 0; epoch < n_epoch; ++epoch) {
        double* epoch_losses = malloc(dataset_samples_rows * sizeof(double));
        for (int dataset_index = 0; dataset_index < dataset_samples_rows; ++dataset_index) {
            double **sample = malloc(dataset_samples_cols * sizeof(double*));
            sample = samples[dataset_index];
            double *target = malloc(dataset_targets_cols * sizeof(double));
            target = targets[dataset_index];
            
            double n_inputs_double = layer_sizes[0 * layer_sizes_cols + 0];
            double n_neurons_double = layer_sizes[0 * layer_sizes_cols + 1];
            int n_inputs = (int)n_inputs_double;
            int n_neurons = (int)n_neurons_double;

            // Forward pass
            double ***X =  malloc(layer_sizes_rows * sizeof(double**));
            double ***Y =  malloc(layer_sizes_rows * sizeof(double**));

            double **x = malloc(dataset_samples_cols * sizeof(double*));
            x = sample;
            double **y = malloc(dataset_samples_cols * sizeof(double*));

            matmul(x, weights[0], y, dataset_samples_cols, dataset_samples_depth, n_inputs, n_neurons);

            // TODO add bias

            X[0] = x;
            Y[0] = y;

            int matrix_rows = dataset_samples_cols;
            int matrix_columns = n_neurons;

            for (int layer_index = 1; layer_index < layer_sizes_rows; layer_index++) {
                double n_inputs_double = layer_sizes[layer_index * layer_sizes_cols + 0];
                double n_neurons_double = layer_sizes[layer_index * layer_sizes_cols + 1];
                int n_inputs = (int)n_inputs_double;
                int n_neurons = (int)n_neurons_double;

                double **x = malloc(matrix_rows * sizeof(double*));
                x = Y[layer_index - 1];
                double **y = malloc(matrix_rows * sizeof(double*));

                matmul(x, weights[layer_index], y, matrix_rows, matrix_columns, n_inputs, n_neurons);

                int activation = (int)activations[layer_index];
                apply_activation_calc(y, matrix_rows, matrix_columns, activation);

                X[layer_index] = x;
                Y[layer_index] = y;

                matrix_rows = matrix_rows;
                matrix_columns = n_neurons;
            }

            // Backward pass
            double ***grad_w = malloc(layer_sizes_rows * sizeof(double**));
            double ***grad_x = malloc(layer_sizes_rows * sizeof(double**));

            n_inputs_double = layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols + 0];
            n_neurons_double = layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols + 1];
            n_inputs = (int)n_inputs_double;
            n_neurons = (int)n_neurons_double;

            double **delta = malloc(matrix_rows * sizeof(double*));
            calc_loss(loss, target, Y[layer_sizes_rows - 1], matrix_rows, n_neurons, delta);
            double output_error = 0.0;
            for (int i = 0; i < matrix_rows; i++) {
                for (int j = 0; j < n_neurons; j++) {
                    output_error += delta[i][j];
                }
            }
            output_error /= matrix_rows + matrix_columns;
            epoch_losses[dataset_index] = output_error;

            x = malloc(matrix_rows * sizeof(double*));  //TODO
            x = X[layer_sizes_rows - 1];
            double **x_T = malloc(n_inputs * sizeof(double*));
            x_T = transpose(x, matrix_rows, n_inputs);
            double **w = malloc(n_inputs * sizeof(double*));
            matmul(x_T, delta, w, n_inputs, matrix_rows, matrix_rows, n_neurons);
            grad_w[layer_sizes_rows - 1] = w;

            w = malloc(n_inputs * sizeof(double*));
            w = weights[layer_sizes_rows - 1];
            double **w_T = malloc(n_neurons * sizeof(double*));
            w_T = transpose(w, n_inputs, n_neurons);
            double **result = malloc(matrix_rows * sizeof(double*));
            matmul(delta, w_T, result, matrix_rows, n_neurons, n_neurons, n_inputs);
            grad_x[layer_sizes_rows - 1] = result;

            matrix_rows = matrix_rows;
            matrix_columns = n_neurons;

            for (int layer_index = layer_sizes_rows - 2; layer_index >= 0; layer_index--) {
                double n_inputs_double = layer_sizes[layer_index * layer_sizes_cols + 0];
                double n_neurons_double = layer_sizes[layer_index * layer_sizes_cols + 1];
                int n_inputs = (int)n_inputs_double;
                int n_neurons = (int)n_neurons_double;

                double **y =  malloc(matrix_rows * sizeof(double*));
                y = Y[layer_index];
                int activation = (int)activations[layer_index];
                apply_activation_derivative(y, matrix_rows, n_neurons, activation);
                double **x = malloc(matrix_rows * sizeof(double*));
                x = grad_x[layer_index + 1];

                double **result = malloc(matrix_rows * sizeof(double*));
                for (int i = 0; i < matrix_rows; i++) {
                    result[i] = malloc(n_neurons * sizeof(double));
                    for (int j = 0; j < n_neurons; j++) {
                        result[i][j] = x[i][j] + y[i][j];
                     }
                }

                double **delta = malloc(matrix_rows * sizeof(double*));
                delta = result;

                x = malloc(matrix_rows * sizeof(double*));
                x = X[layer_index];
                double **x_T = malloc(n_inputs * sizeof(double*));
                x_T = transpose(x, matrix_rows, n_inputs);
                double **w = malloc(n_inputs * sizeof(double*));
                matmul(x_T, delta, w, n_inputs, matrix_rows, matrix_rows, n_neurons);
                grad_w[layer_index] = w;

                w = weights[layer_index];
                double **w_T = malloc(n_neurons * sizeof(double*));
                w_T = transpose(w, n_inputs, n_neurons);
                result = malloc(matrix_rows * sizeof(double*));
                matmul(delta, w_T, result, matrix_rows, n_neurons, n_neurons, n_inputs);
                grad_x[layer_index] = result;

                matrix_rows = matrix_rows;
                matrix_columns = n_neurons;
            }

            // Update weights
            for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
                double n_inputs_double = layer_sizes[layer_index * layer_sizes_cols + 0];
                double n_neurons_double = layer_sizes[layer_index * layer_sizes_cols + 1];
                int n_inputs = (int)n_inputs_double;
                int n_neurons = (int)n_neurons_double;

                double **w = malloc(n_inputs * sizeof(double*));
                w = grad_w[layer_index];

                for (int i = 0; i < n_inputs; i++) {
                    for (int j = 0; j < n_neurons; j++) {
                        double change = safe_weight_update(w[i][j], learning_rate, max_change);
                        weights[layer_index][i][j] += change;
                        if (isnan(weights[layer_index][i][j])) {
                            weights[layer_index][i][j] = 0.0;
                        }
                    }
                }
                matrix_columns = n_neurons;
            }
        }
        double mean_loss = calculate_mean(epoch_losses, dataset_samples_rows);
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

        double **weights_arr = weights[layer_index];
        for (int i = 0; i < n_inputs; i++) {
            free(weights_arr[i]);
        }
        double *bias = biases[layer_index];
        free(bias);
        free(weights_arr);
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
        double *weights_arr = malloc(weights_cols * sizeof(double));
        for (int j = 0; j < weights_cols; ++j) {
            weights_arr[j] = (double)weights_input[i * weights_cols + j];
        }
        weights[i] = weights_arr;
    }

    for (int i = 0; i < biases_rows; ++i) {
        biases[i] = (double)biases_input[i];
    }

    // Forward pass
    double **result = malloc(sample_rows * sizeof(double*));
    matmul(sample, weights, result, sample_rows, sample_cols, n_inputs, n_neurons);

    // Add bias
    // for (int i = 0; i < n_neurons; ++i) {
    //     output_list[i] += biases[i];
    // }

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
