#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h> // для memcpy()


///////////////////////////////////////////////////////////////////////////////
// Activation functions
///////////////////////////////////////////////////////////////////////////////

// Реализация активации ReLU.
void relu_calc(float *input_sample, int input_sample_len, float* output_data) {
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
void relu_derivative(float *input_sample, int input_sample_len, float* output_data) {
    for(int i = 0; i < input_sample_len; ++i) {
        // Производная равна 1 там, где значение больше нуля, иначе — ноль.
        output_data[i] = (input_sample[i] > 0) ? 1.0f : 0.0f;
    }
}

///////////////////////////////////////////////////////////////////////////////

void empty_calc(float *input_sample, int input_sample_len, float* output_data) {
    for(int i = 0; i < input_sample_len; ++i) {
        output_data[i] = input_sample[i];
    }
}

void empty_derivative(float *input_sample, int input_sample_len, float* output_data) {
    for(int i = 0; i < input_sample_len; ++i) {
        output_data[i] = input_sample[i];
    }
}

///////////////////////////////////////////////////////////////////////////////

// Функция сигмоида (Sigmoid activation function).
float sigmoid(float x) {
    // Ограничиваем диапазон ввода, чтобы избежать переполнения экспоненты.
    if (x <= -500) return 0.0;
    if (x >= 500) return 1.0;

    // Используем оптимизированную формулу расчета сигмоиды.
    if (x >= 0) {
        return 1.0 / (1.0 + exp(-x));
    } else {
        return exp(x) / (1.0 + exp(x));
    }
}

// Расчет сигмоидной активации для всех элементов выборки.
void sigmoid_calc(float *input_sample, int input_sample_len, float* output_data) {
    for(int i = 0; i < input_sample_len; ++i) {
        output_data[i] = sigmoid(input_sample[i]);
    }
}

// Расчёт производной сигмоиды.
void sigmoid_derivative(float *input_sample, int input_sample_len, float* output_data) {
    float temp_output[input_sample_len]; // Временный массив для хранения результатов активации.

    sigmoid_calc(input_sample, input_sample_len, temp_output); // Сначала рассчитываем саму активацию.

    for(int i = 0; i < input_sample_len; ++i) {
        // Формула производной сигмоиды: f'(x) = f(x)*(1-f(x)).
        output_data[i] = temp_output[i] * (1.0 - temp_output[i]);
    }
}

///////////////////////////////////////////////////////////////////////////////


// Метод softmax (возвращает нормализованные вероятности классов).
void softmax_calc(float *input_sample, int input_sample_len, float* output_data) {
    // Найдем максимальное значение среди элементов массива.
    float max_val = input_sample[0];
    for (int i = 1; i < input_sample_len; ++i) {
        if (input_sample[i] > max_val) {
            max_val = input_sample[i];
        }
    }

    // Отнимем максимум от каждого элемента для стабилизации экспоненты.
    float sum_exp = 0.0;
    for (int i = 0; i < input_sample_len; ++i) {
        output_data[i] = exp(input_sample[i] - max_val);
        sum_exp += output_data[i];
    }

    // Нормализация путем деления каждого элемента на сумму экспонент.
    for (int i = 0; i < input_sample_len; ++i) {
        output_data[i] /= sum_exp;
    }
}

void softmax_derivative(float *input_sample, int input_sample_len, float* output_data) {
    for(int i = 0; i < input_sample_len; ++i) {
        output_data[i] = input_sample[i];
    }
}


///////////////////////////////////////////////////////////////////////////////
// Loss functions
///////////////////////////////////////////////////////////////////////////////

// Функцию потерь MSE
float mse_loss(float *prediction, int prediction_len, float target) {
    float total_loss = 0.0;

    // Подсчет суммы квадратичных ошибок
    for(int i = 0; i < prediction_len; ++i) {
        float diff = target - prediction[i];
        total_loss += diff * diff;
    }

    // Возвращаем среднее значение квадрата разности
    return total_loss / (2.0 * prediction_len);
}

///////////////////////////////////////////////////////////////////////////////

// Функция подсчета кросс-энтропии
float cross_entropy_loss(float *prediction, int prediction_len, float target) {
    float loss = 0.0;

    // Подсчет кросс-энтропии
    for(int i = 0; i < prediction_len; ++i) {
        // Если вероятность близка к нулю, используем минимальное положительное значение
        float p = prediction[i] > 1e-15 ? prediction[i] : 1e-15;
        
        // Добавляем вклад каждой пары "вероятность-метка" в общий убыток
        loss -= target * log(p);
    }
    return loss;
}

///////////////////////////////////////////////////////////////////////////////


// Функция инициализации смещений (bias)
float* init_bias(int n_neurons) {
    float* bias = malloc(n_neurons * sizeof(float)); // Выделяем память для строк

    // Установка случайного числа
    srand(time(NULL));
    
    for(int i = 0; i < n_neurons; ++i) {
        bias[i] = ((float) rand() / RAND_MAX); // Генерируем число между 0 и 1
    }
    return bias;
}


// Инициализация матриц весов
float** init_weights(int n_neurons, int n_inputs) {
    // Создание динамического двумерного массива
    float** weights = malloc(n_neurons * sizeof(float*)); // Выделяем память для строк

    for (int i = 0; i < n_neurons; i++) {
        weights[i] = malloc(n_inputs * sizeof(float)); // Выделяем память для каждого столбца строки
    }

    // Генерация случайных чисел с нормальным распределением
    srand(time(NULL)); // Устанавливаем начальное состояние генератора случайных чисел

    for (int i = 0; i < n_neurons; i++) {
        for (int j = 0; j < n_inputs; j++) {
            // Генерируем случайное нормальное распределение (среднее=0, стандартное отклонение=1)
            // Простое приближённое решение с использованием Box-Muller преобразования
            float u1 = (float)rand() / RAND_MAX;
            float u2 = (float)rand() / RAND_MAX;
            float z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
            weights[i][j] = (float)(z * 10); // Умножаем каждый вес на 10
        }
    }
    return weights;
}

///////////////////////////////////////////////////////////////////////////////


void apply_activation_calc(float *output_list, int n_neurons, char *activation) {
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

void apply_activation_derivative(float *output_list, int n_neurons, char *activation) {
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

float calc_loss(const char *loss, float target, float *prediction, int prediction_len) {
    if (strcmp(loss, "MSELoss") == 0) {
        return mse_loss(prediction, prediction_len, target);
    } else if (strcmp(loss, "CrossEntropy") == 0) {
        return cross_entropy_loss(prediction, prediction_len, target);
    }
    return 0;
}

///////////////////////////////////////////////////////////////////////////////

float calculate_mean(float *arr, int len) {
    if (len == 0) return 0.0;
    float sum = 0.0;
    for (int i = 0; i < len; ++i) {
        sum += arr[i];
    }
    return sum / len;
}

///////////////////////////////////////////////////////////////////////////////


void save_as_json(char *fname, float ***weights_result, double *layer_sizes, int layer_sizes_rows, int layer_sizes_cols) {
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
    double *dataset,
    int dataset_rows,
    int dataset_cols,
    double *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols,
    char **activations,
    int activations_len,
    const char *loss,
    int n_epoch,
    float learning_rate,
    int verbose) {
    
    // Загрузка датасета
    float dataset_samples[dataset_rows][dataset_cols - 1];
    float dataset_targets[dataset_rows];

    for(int i = 0; i < dataset_rows; ++i) {
        int j = 0;
        for(; j < dataset_cols; ++j) {
            dataset_samples[i][j] = dataset[i * dataset_cols + j];
        }
        dataset_targets[i] = dataset[i * dataset_cols + i + 1];
    }

    // Инициализация смещений и весов
    float** biases = malloc(layer_sizes_rows * sizeof(float*));
    float*** weights = malloc(layer_sizes_rows * sizeof(float**));

    for(int i = 0; i < layer_sizes_rows; ++i) {
        double n_inputs_double = layer_sizes[i * layer_sizes_cols + 0];
        double n_neurons_double = layer_sizes[i * layer_sizes_cols + 1];

        int n_inputs = (int)n_inputs_double;
        int n_neurons = (int)n_neurons_double;

        // Инициализировать смещение
        biases[i] = init_bias(n_neurons);

        // Инициализировать веса
        weights[i] = init_weights(n_neurons, n_inputs);
    }

    // Обучение
    float losses_by_epoch[n_epoch];
    float **output_lists = malloc(layer_sizes_rows * sizeof(float*));

    for (int epoch = 0; epoch < n_epoch; ++epoch) {
        float* epoch_losses = malloc(dataset_rows * sizeof(float));

        for (int dataset_index = 0; dataset_index < dataset_rows; ++dataset_index) {
            float sample[dataset_cols - 1];
            memcpy(sample, dataset_samples[dataset_index], (dataset_cols - 1) * sizeof(float));
            float target = dataset_targets[dataset_index];

            double n_inputs_double = layer_sizes[0 * layer_sizes_cols + 0];
            double n_neurons_double = layer_sizes[0 * layer_sizes_cols + 1];

            int n_inputs = (int)n_inputs_double;
            int n_neurons = (int)n_neurons_double;
            float output_list[n_neurons];

            // Forward pass
            for (int i = 0; i < n_neurons; i++) {
                float output = 0.0;
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

            for (int layer_index = 1; layer_index < layer_sizes_rows; ++layer_index) {
                n_inputs_double = layer_sizes[layer_index * layer_sizes_cols + 0];
                n_neurons_double = layer_sizes[layer_index * layer_sizes_cols + 1];

                n_inputs = (int)n_inputs_double;
                n_neurons = (int)n_neurons_double;

                float* output_list = calloc(n_neurons, sizeof(float));

                for (int i = 0; i < n_neurons; i++) {
                    float output = 0.0;
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
                char* activation = activations[layer_index];
                apply_activation_calc(output_list, n_neurons, activation);
                output_lists[layer_index] = output_list;
            }

            n_neurons_double = layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols + 1];
            n_neurons = (int)n_neurons_double;

            float *prediction = output_lists[layer_sizes_rows - 1];

            // Calc loss
            float output_error = calc_loss(loss, target, prediction, n_neurons);
            epoch_losses[dataset_index] = output_error;

            // Backward pass
            float **delta_list = malloc(layer_sizes_rows * sizeof(float*));
            n_inputs_double = layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols + 0];
            n_neurons_double = layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols + 1];
            n_inputs = (int)n_inputs_double;
            n_neurons = (int)n_neurons_double;

            float *delta = malloc(n_neurons * sizeof(float));
            for (int i = 0; i < n_neurons; ++i) {
                delta[i] = output_error;
            }

            activation = activations[layer_sizes_rows - 1];
            prediction = output_lists[layer_sizes_rows - 1];
            apply_activation_derivative(prediction, n_neurons, activation);
            for (int i = 0; i < n_neurons; i++) {
                delta[i] *= prediction[i];
            }

            delta_list[layer_sizes_rows - 1] = delta;

            for (int layer_index = layer_sizes_rows - 2; layer_index >= 0; layer_index--) {
                n_inputs_double = layer_sizes[layer_index * layer_sizes_cols + 0];
                n_neurons_double = layer_sizes[layer_index * layer_sizes_cols + 1];
                n_inputs = (int)n_inputs_double;
                n_neurons = (int)n_neurons_double;

                activation = activations[layer_index];
                prediction = output_lists[layer_index];
                apply_activation_derivative(prediction, n_neurons, activation);

                float *new_delta = malloc(n_neurons * n_inputs * sizeof(float));
                for (int i = 0; i < n_neurons; i++) {
                    for (int j = 0; j < n_inputs; j++) {
                        float num = weights[layer_index][i][j] * prediction[i] * delta_list[layer_index + 1][i + j];
                        new_delta[i + j] = num;
                    }
                }
                delta_list[layer_index] = new_delta;
            }

            char *file = "weights_1.json";
            save_as_json(file, weights, layer_sizes, layer_sizes_rows, layer_sizes_cols);

            // Update weights
            for (int layer_index = 1; layer_index < layer_sizes_rows; ++layer_index) {
                n_inputs_double = layer_sizes[layer_index * layer_sizes_cols + 0];
                n_neurons_double = layer_sizes[layer_index * layer_sizes_cols + 1];
                n_inputs = (int)n_inputs_double;
                n_neurons = (int)n_neurons_double;

                for (int i = 0; i < n_neurons; i++) {
                    for (int j = 0; j < n_inputs; j++) {
                        weights[layer_index][i][j] += delta_list[layer_index][i + j] * learning_rate;
                    }
                }
            }
        }
        float mean_loss = calculate_mean(epoch_losses, dataset_rows);
        losses_by_epoch[epoch] = mean_loss;
        if (verbose) {
            printf("Epoch %d / %d. Loss: %f\n", epoch + 1, n_epoch, mean_loss);
        }
    }
    char *file = "weights_2.json";
    save_as_json(file, weights, layer_sizes, layer_sizes_rows, layer_sizes_cols);
}
