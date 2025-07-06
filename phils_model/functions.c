#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h> // для memcpy()


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

void empty_calc(double *input_sample, int input_sample_len, double* output_data) {
    for(int i = 0; i < input_sample_len; ++i) {
        output_data[i] = input_sample[i];
    }
}

void empty_derivative(double *input_sample, int input_sample_len, double* output_data) {
    for(int i = 0; i < input_sample_len; ++i) {
        output_data[i] = input_sample[i];
    }
}

///////////////////////////////////////////////////////////////////////////////
// Loss functions
///////////////////////////////////////////////////////////////////////////////

// Функцию потерь MSE
double mse_loss(double *prediction, int prediction_len, double *target) {
    double total_loss = 0.0;

    // Подсчет суммы квадратичных ошибок
    for(int i = 0; i < prediction_len; ++i) {
        double diff = target[i] - prediction[i];
        total_loss += diff * diff;
    }

    // Возвращаем среднее значение квадрата разности
    return total_loss / (2.0 * prediction_len);
}

///////////////////////////////////////////////////////////////////////////////

// Функция подсчета кросс-энтропии
double cross_entropy_loss(double *prediction, int prediction_len, double *target) {
    double loss = 0.0;

    // Подсчет кросс-энтропии
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
        bias[i] = ((double)rand() / RAND_MAX) * 10.0;
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
            weights[i][j] = (double)(z * 10.0); // Умножаем каждый вес на 10
        }
    }
    return weights;
}

///////////////////////////////////////////////////////////////////////////////

void apply_activation_calc(double *output_list, int n_neurons, char *activation) {
    if (strcmp(activation, "ReLU") == 0) {
        relu_calc(output_list, n_neurons, output_list);
    } else if (strcmp(activation, "Sigmoid") == 0) {
        sigmoid_calc(output_list, n_neurons, output_list);
    } else if (strcmp(activation, "Softmax") == 0) {
        softmax_calc(output_list, n_neurons, output_list);
    } else if (strcmp(activation, "Empty") == 0) {
        empty_calc(output_list, n_neurons, output_list);
    }
}

void apply_activation_derivative(double *output_list, int n_neurons, char *activation) {
    if (strcmp(activation, "ReLU") == 0) {
        relu_derivative(output_list, n_neurons, output_list);
    } else if (strcmp(activation, "Sigmoid") == 0) {
        sigmoid_derivative(output_list, n_neurons, output_list);
    } else if (strcmp(activation, "Softmax") == 0) {
        softmax_derivative(output_list, n_neurons, output_list);
    } else if (strcmp(activation, "Empty") == 0) {
        empty_derivative(output_list, n_neurons, output_list);
    }
}

double calc_loss(const char *loss, double *target, double *prediction, int prediction_len) {
    if (strcmp(loss, "MSELoss") == 0) {
        return mse_loss(prediction, prediction_len, target);
    } else if (strcmp(loss, "CrossEntropy") == 0) {
        return cross_entropy_loss(prediction, prediction_len, target);
    }
    return 0.0;
}

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

    // printf("%f     %f      %f\n", delta, learning_rate, change);

    if (change > max_change)
        change = max_change;
    else if (change < -max_change)
        change = -max_change;

    // printf("safe_weight_update %f\n", change);

    return change;
}

///////////////////////////////////////////////////////////////////////////////

void save_as_json(char *fname, double ***weights_result, double *layer_sizes, int layer_sizes_rows, int layer_sizes_cols) {
    FILE *fp = fopen(fname, "w+");

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
        for (size_t neuron = 0; neuron < n_neurons; neuron++) {
            fprintf(fp, "\t\t[\n");

            fprintf(fp, "\t\t\t");
            for (size_t input = 0; input < n_inputs; input++) {
                if (input != 0) fprintf(fp, "\t\t\t");
                if (input == n_inputs - 1) {
                    fprintf(fp, "%f\n", weights_result[layer_size][neuron][input]);
                } else {
                    fprintf(fp, "%f,\n", weights_result[layer_size][neuron][input]);
                }
            }

            fprintf(fp, "\t\t]");
            if (neuron != n_neurons - 1) fprintf(fp, ",");
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
    char **activations,
    int activations_len,
    const char *loss,
    int n_epoch,
    double learning_rate,
    int verbose) {

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
    double** biases = malloc(layer_sizes_rows * sizeof(double*));
    double*** weights = malloc(layer_sizes_rows * sizeof(double**));

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
            char* activation = activations[0];
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
                        output += weights[layer_index][i][j] * output_lists[layer_index - 1][i];
                    }
                    output_list[i] = output;
                }

                // Add bias
                for (int i = 0; i < n_neurons; ++i) {
                    output_list[i] += biases[layer_index][i];
                }

                // Apply activation function
                char* activation = activations[layer_index];
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
            for (int i = 0; i < n_neurons; ++i) {
                delta[i] = output_error;
            }

            activation = activations[layer_sizes_rows - 1];
            apply_activation_derivative(prediction, n_neurons, activation);
            double *new_delta = malloc(n_inputs * sizeof(double));

            // for (int i = 0; i < n_neurons; i++) {
            //     double num = 0.0;
            //     for (int j = 0; j < n_inputs; j++) {
            //         double num_1 = prediction[i] * delta[i] * weights[layer_sizes_rows - 1][i][j];
            //         num += num_1;
            //         // printf("%f     %f       %f      %f привет\n", num_1, prediction[i], delta[i], weights[layer_sizes_rows - 1][i][j]);
            //         // printf("%f\n",weights[layer_sizes_rows - 1][i][j]);
            //     }
            //     new_delta[i] = num / (double)n_inputs;
            // }
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

                double n_inputs_double_delta = layer_sizes[(layer_index + 1) * layer_sizes_cols + 0];
                int n_inputs_delta = (int)n_inputs_double_delta;

                activation = activations[layer_index];
                prediction = output_lists[layer_index];
                apply_activation_derivative(prediction, n_neurons, activation);

                double *new_delta = malloc(n_inputs * sizeof(double));
                for (int j = 0; j < n_inputs; j++) {
                    double num = 0.0;
                    for (int i = 0; i < n_neurons; i++) {
                        double num_1 = prediction[i] * delta_list[layer_index + 1][n_inputs_delta - 1] * weights[layer_index][i][j];
                        num += num_1;
                    }
                    new_delta[j] = num / (double)n_neurons;
                }
                delta_list[layer_index] = new_delta;
            }







            // TODO





            char *file = "weights_1.json";
            save_as_json(file, weights, layer_sizes, layer_sizes_rows, layer_sizes_cols);

            // Update weights
            for (int layer_index = 0; layer_index < layer_sizes_rows; ++layer_index) {
                n_inputs_double = layer_sizes[layer_index * layer_sizes_cols + 0];
                n_neurons_double = layer_sizes[layer_index * layer_sizes_cols + 1];
                n_inputs = (int)n_inputs_double;
                n_neurons = (int)n_neurons_double;

                for (int i = 0; i < n_neurons; i++) {
                    for (int j = 0; j < n_inputs; j++) {
                        printf("До %f\n", weights[layer_index][i][j]);
                        weights[layer_index][i][j] += safe_weight_update(delta_list[layer_index][i], learning_rate, 10);
                        // printf("%f\n", safe_weight_update(delta_list[layer_index][i], learning_rate, 10));





                        // printf("%f\n", safe_weight_update(delta_list[layer_index][i + j], learning_rate, 10000));
                        // printf("%f\n", learning_rate);
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
    char *file = "weights_2.json";
    save_as_json(file, weights, layer_sizes, layer_sizes_rows, layer_sizes_cols);
}







