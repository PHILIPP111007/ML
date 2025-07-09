// gcc -shared -o functions.so -fPIC -O3 functions.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

///////////////////////////////////////////////////////////////////////////////

double calculate_mean(double *arr, int len) {
    if (len == 0) return 0.0;
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
    
    for(int i = 1; i < size; ++i) {
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

// Реализация активации ReLU.
void relu_calc(double *input_sample, int input_sample_len, double* output_data) {
    for(int i = 0; i < input_sample_len; ++i) {
        // Применяем max(0, x), аналог np.maximum(0, x) из Python.
        if(input_sample[i] > 0) {
            output_data[i] = input_sample[i];
        }
        else {
            output_data[i] = 0.0f;
        }
    }
}

// Вычисляем производную от ReLU (это бинарная маска неотрицательных элементов).
void relu_derivative(double *input_sample, int input_sample_len, double* output_data) {
    for(int i = 0; i < input_sample_len; ++i) {
        if(input_sample[i] > 0) {
            output_data[i] = input_sample[i];
        }
        else {
            output_data[i] = 0.0f;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

// Функция сигмоида (Sigmoid activation function).
double sigmoid(double x) {
    // Используем оптимизированную формулу расчета сигмоиды.
    if (x >= 0) {
        return 1.0 / (1.0 + exp(-x));
    } else {
        return exp(x) / (1.0 + exp(x));
    }
}

// Расчет сигмоидной активации для всех элементов выборки.
void sigmoid_calc(double *input_sample, int input_sample_len, double* output_data) {
    for(int i = 0; i < input_sample_len; ++i) {
        output_data[i] = sigmoid(input_sample[i]);
    }
}

// Расчёт производной сигмоиды.
void sigmoid_derivative(double *input_sample, int input_sample_len, double* output_data) {
    double temp_output[input_sample_len]; // Временный массив для хранения результатов активации.

    sigmoid_calc(input_sample, input_sample_len, temp_output); // Сначала рассчитываем саму активацию.

    for(int i = 0; i < input_sample_len; ++i) {
        // Формула производной сигмоиды: f'(x) = f(x)*(1-f(x)).
        output_data[i] = temp_output[i] * (1.0 - temp_output[i]);
    }
}

///////////////////////////////////////////////////////////////////////////////

// Метод softmax (возвращает нормализованные вероятности классов).
void softmax_calc(double *input_sample, int input_sample_len, double* output_data) {
    // Найдем максимальное значение среди элементов массива.
    double max_val = input_sample[0];
    for (int i = 1; i < input_sample_len; ++i) {
        if (input_sample[i] > max_val) {
            max_val = input_sample[i];
        }
    }

    // Отнимем максимум от каждого элемента для стабилизации экспоненты.
    double sum_exp = 0.0;
    for (int i = 0; i < input_sample_len; ++i) {
        output_data[i] = exp(input_sample[i] - max_val);
        sum_exp += output_data[i];
    }

    // Нормализация путем деления каждого элемента на сумму экспонент.
    for (int i = 0; i < input_sample_len; ++i) {
        output_data[i] /= sum_exp;
    }
}

void softmax_derivative(double *input_sample, int input_sample_len, double* output_data) {
    for(int i = 0; i < input_sample_len; ++i) {
        output_data[i] = input_sample[i];
    }
}

///////////////////////////////////////////////////////////////////////////////
// Loss functions
///////////////////////////////////////////////////////////////////////////////

double mse_loss(double *prediction, int prediction_len, double *target) {
    double *loss = malloc(prediction_len * sizeof(double));

    for(int i = 0; i < prediction_len; ++i) {
        loss[i] = target[i] - prediction[i];
    }
    return calculate_mean(loss, prediction_len);
}

///////////////////////////////////////////////////////////////////////////////

double cross_entropy_loss(double *prediction, int prediction_len, double *target) {
    double loss = 0.0;

    for(int i = 0; i < prediction_len; ++i) {
        // Если вероятность близка к нулю, используем минимальное положительное значение
        double p = prediction[i] > 1e-15 ? prediction[i] : 1e-15;

        // Добавляем вклад каждой пары "вероятность-метка" в общий убыток
        loss -= target[i] * log(p);
    }
    return loss;
}

///////////////////////////////////////////////////////////////////////////////

// Функция инициализации смещений
double* init_bias(int n_neurons) {
    double* bias = malloc(n_neurons * sizeof(double)); // Выделяем память для строк

    for(int i = 0; i < n_neurons; ++i) {
        bias[i] = (double)rand() / RAND_MAX;
    }
    return bias;
}

// Инициализация матриц весов
double** init_weights(int n_neurons, int n_inputs) {
    // Создание динамического двумерного массива
    double** weights = malloc(n_neurons * sizeof(double*)); // Выделяем память для строк

    for (int i = 0; i < n_neurons; i++) {
        weights[i] = malloc(n_inputs * sizeof(double)); // Выделяем память для каждого столбца строки
    }

    // Генерация случайных чисел с нормальным распределением
    for (int i = 0; i < n_neurons; i++) {
        for (int j = 0; j < n_inputs; j++) {
            // Генерируем случайное нормальное распределение (среднее=0, стандартное отклонение=1)
            // Простое приближённое решение с использованием Box-Muller преобразования
            double u1 = (double)rand() / RAND_MAX;
            double u2 = (double)rand() / RAND_MAX;
            double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
            weights[i][j] = (double)z;
        }
    }
    return weights;
}

///////////////////////////////////////////////////////////////////////////////

void apply_activation_calc(double *output_list, int n_neurons, int activation) {
    if (activation == 0) {
        relu_calc(output_list, n_neurons, output_list);
    } else if (activation == 1) {
        sigmoid_calc(output_list, n_neurons, output_list);
    } else if (activation == 2) {
        softmax_calc(output_list, n_neurons, output_list);
    } else if (activation == 3) {
        return;
    }
}

void apply_activation_derivative(double *output_list, int n_neurons, int activation) {
    if (activation == 0) {
        relu_derivative(output_list, n_neurons, output_list);
    } else if (activation == 1) {
        sigmoid_derivative(output_list, n_neurons, output_list);
    } else if (activation == 2) {
        softmax_derivative(output_list, n_neurons, output_list);
    } else if (activation == 3) {
        return;
    }
}

double calc_loss(int loss, double *target, double *prediction, int prediction_len) {
    if (loss == 0) {
        return mse_loss(prediction, prediction_len, target);
    } else if (loss == 1) {
        return cross_entropy_loss(prediction, prediction_len, target);
    }
    return 0.0;
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
        for (int i = 0; i < n_neurons; i++) {
            fprintf(fp, "\t\t[\n");

            fprintf(fp, "\t\t\t");
            for (int j = 0; j < n_inputs; j++) {
                if (j != 0) fprintf(fp, "\t\t\t");
                if (j == n_inputs - 1) {
                    fprintf(fp, "%f\n", weights_result[layer_size][i][j]);
                } else {
                    fprintf(fp, "%f,\n", weights_result[layer_size][i][j]);
                }
            }

            fprintf(fp, "\t\t]");
            if (i != n_neurons - 1) fprintf(fp, ",");
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
    double** samples = malloc(dataset_samples_rows * sizeof(double*));
    double** targets = malloc(dataset_targets_rows * sizeof(double*));

    for (int i = 0; i < dataset_samples_rows; ++i) {
        double *arr = malloc(dataset_samples_cols * sizeof(double));
        for (int j = 0; j < dataset_samples_cols; ++j) {
            arr[j] = (double)dataset_samples[i * dataset_samples_cols + j];
        }
        samples[i] = arr;
    }

    for (int i = 0; i < dataset_targets_rows; ++i) {
        double *arr = malloc(dataset_targets_cols * sizeof(double));
        for (int j = 0; j < dataset_targets_cols; ++j) {
            arr[j] = (double)dataset_targets[i * dataset_targets_cols + j];
        }
        targets[i] = arr;
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
        double *biases_arr = malloc(n_neurons * sizeof(double));
        biases_arr = init_bias(n_neurons);
        biases[layer_index] = biases_arr;

        // Инициализировать веса
        double **weights_arr = malloc(n_neurons * sizeof(double*));
        weights_arr = init_weights(n_neurons, n_inputs);
        weights[layer_index] = weights_arr;
    }

    // Обучение
    double losses_by_epoch[n_epoch];
    double **output_lists = malloc(layer_sizes_rows * sizeof(double*));

    for (int epoch = 0; epoch < n_epoch; ++epoch) {
        double* epoch_losses = malloc(dataset_samples_rows * sizeof(double));
        for (int dataset_index = 0; dataset_index < dataset_samples_rows; ++dataset_index) {
            double *sample = malloc(dataset_samples_cols * sizeof(double));
            for (int i = 0; i < dataset_samples_cols; ++i) {
                sample[i] = samples[dataset_index][i];
            }

            double *target = malloc(dataset_targets_cols * sizeof(double));
            for (int i = 0; i < dataset_targets_cols; ++i) {
                target[i] = targets[dataset_index][i];
            }

            double n_inputs_double = layer_sizes[0 * layer_sizes_cols + 0];
            double n_neurons_double = layer_sizes[0 * layer_sizes_cols + 1];
            int n_inputs = (int)n_inputs_double;
            int n_neurons = (int)n_neurons_double;

            double *output_list = malloc(n_neurons * sizeof(double));

            // Forward pass
            for (int i = 0; i < n_neurons; i++) {
                double output = 0.0;
                for (int j = 0; j < n_inputs; j++) {
                    output += weights[0][i][j] * sample[j];
                }
                output_list[i] = output;
            }

            // Add bias
            for (int i = 0; i < n_neurons; ++i) {
                output_list[i] += biases[0][i];
            }

            // Apply activation function
            int activation = (int)activations[0];
            apply_activation_calc(output_list, n_neurons, activation);
            output_lists[0] = output_list;

            for (int layer_index = 1; layer_index < layer_sizes_rows; layer_index++) {
                double n_inputs_double = layer_sizes[layer_index * layer_sizes_cols + 0];
                double n_neurons_double = layer_sizes[layer_index * layer_sizes_cols + 1];
                int n_inputs = (int)n_inputs_double;
                int n_neurons = (int)n_neurons_double;

                double *output_list = malloc(n_neurons * sizeof(double));

                for (int i = 0; i < n_neurons; i++) {
                    double output = 0.0;
                    for (int j = 0; j < n_inputs; j++) {
                        output += weights[layer_index][i][j] * output_lists[layer_index - 1][j];
                    }
                    output_list[i] = output;
                }

                // Add bias
                for (int i = 0; i < n_neurons; ++i) {
                    output_list[i] += biases[layer_index][i];
                }

                // Apply activation function
                int activation = (int)activations[layer_index];
                apply_activation_calc(output_list, n_neurons, activation);
                output_lists[layer_index] = output_list;
            }
            n_neurons_double = layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols + 1];
            n_neurons = (int)n_neurons_double;

            double *prediction = malloc(n_neurons * sizeof(double));
            prediction = output_lists[layer_sizes_rows - 1];

            // Calc loss
            double output_error = calc_loss(loss, target, prediction, n_neurons);
            epoch_losses[dataset_index] = output_error;

            // Backward pass
            double **delta_list = malloc(layer_sizes_rows * sizeof(double*));
            n_inputs_double = layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols + 0];
            n_neurons_double = layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols + 1];
            n_inputs = (int)n_inputs_double;
            n_neurons = (int)n_neurons_double;

            double *delta = malloc(n_neurons * sizeof(double));
            int max_value_index = argmax(prediction, n_neurons);
            for (int i = 0; i < n_neurons; ++i) {
                if (i == max_value_index) {
                    delta[i] = output_error;
                } else {
                    delta[i] = 0;
                }
            }

            activation = (int)activations[layer_sizes_rows - 1];
            apply_activation_derivative(prediction, n_neurons, activation);
            double *new_delta = malloc(n_inputs * sizeof(double));

            for (int j = 0; j < n_inputs; j++) {
                double num = 0.0;
                for (int i = 0; i < n_neurons; i++) {
                    double num_1 = prediction[i] * delta[i] * weights[layer_sizes_rows - 1][i][j];
                    num += num_1;
                }
                new_delta[j] = num / (double)n_neurons;
            }

            delta_list[layer_sizes_rows - 1] = new_delta;
            for (int layer_index = layer_sizes_rows - 2; layer_index >= 0; layer_index--) {
                double n_inputs_double = layer_sizes[layer_index * layer_sizes_cols + 0];
                double n_neurons_double = layer_sizes[layer_index * layer_sizes_cols + 1];
                int n_inputs = (int)n_inputs_double;
                int n_neurons = (int)n_neurons_double;

                activation = (int)activations[layer_index];

                double *prediction = malloc(n_neurons * sizeof(double));
                prediction = output_lists[layer_index];
                apply_activation_derivative(prediction, n_neurons, activation);

                double *new_delta = malloc(n_inputs * sizeof(double));
                for (int j = 0; j < n_inputs; j++) {
                    double num = 0.0;
                    for (int i = 0; i < n_neurons; i++) {
                        double num_1 = prediction[i] * delta_list[layer_index + 1][n_neurons] * weights[layer_index][i][j];
                        num += num_1;
                    }
                    new_delta[j] = num / (double)n_neurons;
                }
                delta_list[layer_index] = new_delta;
            }

            // Update weights
            for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
                double n_inputs_double = layer_sizes[layer_index * layer_sizes_cols + 0];
                double n_neurons_double = layer_sizes[layer_index * layer_sizes_cols + 1];
                int n_inputs = (int)n_inputs_double;
                int n_neurons = (int)n_neurons_double;

                for (int i = 0; i < n_neurons; i++) {
                    for (int j = 0; j < n_inputs; j++) {
                        double change = safe_weight_update(delta_list[layer_index][j], learning_rate, max_change);
                        weights[layer_index][i][j] += change;
                    }
                }
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
        double n_neurons_double = layer_sizes[layer_index * layer_sizes_cols + 1];
        int n_neurons = (int)n_neurons_double;

        double **weights_arr = weights[layer_index];
        double *bias = biases[layer_index];
        for (int i = 0; i < n_neurons; i++) {
            free(weights_arr[i]);
        }
        free(bias);
        free(weights_arr);
    }
    free(biases);
    free(weights);
}


void predict(
    double *sample_input,
    int sample_rows,
    double *biases_input,
    int biases_rows,
    double *weights_input,
    int weights_rows,
    int weights_cols,
    double *layer_size,
    int layer_size_rows,
    int activation,
    double *prediction) {

    double *sample = malloc(sample_rows * sizeof(double));
    for (int i = 0; i < sample_rows; ++i) {
        sample[i] = sample_input[i];
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

    double n_inputs_double = layer_size[0];
    double n_neurons_double = layer_size[1];
    int n_inputs = (int)n_inputs_double;
    int n_neurons = (int)n_neurons_double;

    double *output_list = malloc(n_neurons * sizeof(double));

    // Forward pass
    for (int i = 0; i < n_neurons; i++) {
        double output = 0.0;
        for (int j = 0; j < n_inputs; j++) {
            output += weights[i][j] * sample[j];
        }
        output_list[i] = output;
    }

    // Add bias
    for (int i = 0; i < n_neurons; ++i) {
        output_list[i] += biases[i];
    }

    // Apply activation function
    apply_activation_calc(output_list, n_neurons, activation);

    for (int i = 0; i < n_neurons; ++i) {
        prediction[i] = output_list[i];
    }
}


