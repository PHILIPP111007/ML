// clang -shared -fopenmp -o main.so -fPIC -O3 main.c src/functions.c src/activations.c src/loss.c src/init.c src/json.c src/adam.c src/forward.c src/backward.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include "src/functions.h"
#include "src/activations.h"
#include "src/loss.h"
#include "src/init.h"
#include "src/json.h"
#include "src/adam.h"
#include "src/forward.h"
#include "src/backward.h"


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
    int num_cpu) {
    
    if (random_state != -1) {
        srand(random_state); // set the initial state of the generator
    }

    // Loading a dataset
    double ***samples = malloc(dataset_samples_rows * sizeof(double**));
    double **targets = malloc(dataset_targets_rows * sizeof(double*)); 

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

    // Initialization of biases and weights
    double **biases = malloc(layer_sizes_rows * sizeof(double*));
    double ***weights = malloc(layer_sizes_rows * sizeof(double**));

    for (int layer_index = 0; layer_index < layer_sizes_rows; ++layer_index) {
        double n_inputs_double = layer_sizes[layer_index * layer_sizes_cols + 0];
        double n_neurons_double = layer_sizes[layer_index * layer_sizes_cols + 1];
        int n_inputs = (int)n_inputs_double;
        int n_neurons = (int)n_neurons_double;

        // Initialize biases
        biases[layer_index] = malloc(n_neurons * sizeof(double));
        double *biases_arr = malloc(n_neurons * sizeof(double));
        biases_arr = init_bias(n_neurons, n_inputs);
        for (int i = 0; i < n_neurons; ++i) {
            biases[layer_index][i] = biases_arr[i];
        }
        free(biases_arr);

        // Initialize weights
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

    // Training

    // Create Adam
    struct AdamOptimizer *opt = create_adam(learning_rate, 0.9, 0.999, 1e-8, layer_sizes, layer_sizes_rows, layer_sizes_cols);

    double losses_by_epoch[n_epoch];
    for (int epoch = 0; epoch < n_epoch; ++epoch) {
        double *epoch_losses = malloc(dataset_samples_rows * sizeof(double));
        int matrix_rows = dataset_samples_cols;

        const int num_threads = (dataset_samples_rows < num_cpu) ? dataset_samples_rows : num_cpu;
        struct ForwardData forward_thread_data[num_threads];
        struct BackwardData backward_thread_data[num_threads];
        
        // Forward pass

        double ****X_list_intermediate = malloc(dataset_samples_rows * sizeof(double***));
        double ****Y_list_intermediate = malloc(dataset_samples_rows * sizeof(double***));
        double ****X_list = malloc(dataset_samples_rows * sizeof(double***));
        double ****Y_list = malloc(dataset_samples_rows * sizeof(double***));

        forward_threading(
            forward_thread_data,
            samples,
            weights,
            biases,
            X_list_intermediate,
            Y_list_intermediate,
            dataset_samples_rows,
            dataset_samples_cols,
            dataset_samples_depth,
            layer_sizes,
            layer_sizes_rows,
            layer_sizes_cols,
            activations,
            keep_prob,
            num_threads
        );

        for (int dataset_index = 0; dataset_index < dataset_samples_rows; ++dataset_index) {
            for (int i = 0; i < dataset_samples_cols; ++i) {
                free(samples[dataset_index][i]);
            }
            free(samples[dataset_index]);
        }
        free(samples);

        for (int t = 0; t < num_threads; ++t) {
            for (int dataset_index = 0; dataset_index < dataset_samples_rows; ++dataset_index) {
                X_list[dataset_index] = forward_thread_data[t].X_list[dataset_index];
                Y_list[dataset_index] = forward_thread_data[t].Y_list[dataset_index];
            }
        }

        double ****grad_w_list = malloc(dataset_samples_rows * sizeof(double***));
        double ****grad_x_list = malloc(dataset_samples_rows * sizeof(double***));
        double ***grad_b_list = malloc(dataset_samples_rows * sizeof(double**));

        // Backward pass
        backward_threading(
            backward_thread_data,
            weights,
            targets,
            biases,
            X_list,
            Y_list,
            grad_w_list,
            grad_x_list,
            grad_b_list,
            layer_sizes,
            layer_sizes_rows,
            layer_sizes_cols,
            dataset_samples_rows,
            dataset_samples_cols,
            dataset_targets_cols,
            matrix_rows,
            activations,
            loss,
            epoch_losses,
            regression,
            num_threads
        );

        // Update weights and biases
        for (int dataset_index = 0; dataset_index < dataset_samples_rows; ++dataset_index) {
            double ***grad_w = grad_w_list[dataset_index];
            double **grad_b = grad_b_list[dataset_index];

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
        }
        for (int t = 0; t < num_threads; ++t) {
            delete_backward_thread_data(&backward_thread_data[t]);
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

    // Clearing memory

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
    double *prediction) {

    double **sample = malloc(sample_rows * sizeof(double*));
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

    double ***Y = malloc(layer_sizes_rows * sizeof(double**));
    
    // Forward pass
    forward(sample, sample_rows, sample_cols, weights, biases, Y, layer_sizes, layer_sizes_rows, layer_sizes_cols, activations);


    double n_inputs_double = layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols + 0];
    double n_neurons_double = layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols + 1];
    int n_inputs = (int)n_inputs_double;
    int n_neurons = (int)n_neurons_double;

    int matrix_rows = sample_rows;

    double **y = malloc(matrix_rows * sizeof(double*));
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
    
    // Clearing memory
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
    double *predictions) {

    // Loading a dataset
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

        double ***Y = malloc(layer_sizes_rows * sizeof(double**));
        
        // Forward pass
        forward(sample, dataset_samples_cols, dataset_samples_depth, weights, biases, Y, layer_sizes, layer_sizes_rows, layer_sizes_cols, activations);

        double n_inputs_double = layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols + 0];
        double n_neurons_double = layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols + 1];
        int n_inputs = (int)n_inputs_double;
        int n_neurons = (int)n_neurons_double;

        int matrix_rows = dataset_samples_cols;

        double **y = malloc(matrix_rows * sizeof(double*));
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

    // Clearing memory
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
