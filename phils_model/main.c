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
    int threading,
    int num_cpu) {
    
    if (random_state != -1) {
        srand(random_state); // set the initial state of the generator
    }

    // Loading a dataset
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

        const int num_threads = dataset_samples_rows;
        pthread_t forward_threads[num_threads];
        pthread_t backward_threads[num_threads];
        
        struct ForwardData forward_thread_data[num_threads];
        struct BackwardData backward_thread_data[num_threads];
        
        // Forward pass
        for (int dataset_index = 0; dataset_index < dataset_samples_rows; ++dataset_index) {
            double **sample = malloc(dataset_samples_cols * sizeof(double*));
            for (int i = 0; i < dataset_samples_cols; i++) {
                sample[i] = malloc(dataset_samples_depth * sizeof(double));
                for (int j = 0; j < dataset_samples_depth; j++) {
                    sample[i][j] = samples[dataset_index][i][j];
                }
            }

            double ***X = malloc(layer_sizes_rows * sizeof(double**));
            double ***Y = malloc(layer_sizes_rows * sizeof(double**));

            forward_thread_data[dataset_index].dataset_index = dataset_index;
            forward_thread_data[dataset_index].sample = sample;
            forward_thread_data[dataset_index].sample_rows = dataset_samples_cols;
            forward_thread_data[dataset_index].sample_cols = dataset_samples_depth;
            forward_thread_data[dataset_index].weights = weights;
            forward_thread_data[dataset_index].biases = biases;
            forward_thread_data[dataset_index].X = X;
            forward_thread_data[dataset_index].Y = Y;
            forward_thread_data[dataset_index].layer_sizes = layer_sizes;
            forward_thread_data[dataset_index].layer_sizes_rows = layer_sizes_rows;
            forward_thread_data[dataset_index].layer_sizes_cols = layer_sizes_cols;
            forward_thread_data[dataset_index].activations = activations;
            forward_thread_data[dataset_index].keep_prob = keep_prob;
            forward_thread_data[dataset_index].threading = threading;
            forward_thread_data[dataset_index].num_cpu = num_cpu;

            pthread_create(&forward_threads[dataset_index], NULL, forward_train, &forward_thread_data[dataset_index]);
        }
        for (int t = 0; t < num_threads; ++t) {
            pthread_join(forward_threads[t], NULL);
        }

        // Backward pass
        for (int dataset_index = 0; dataset_index < dataset_samples_rows; ++dataset_index) {
            double ***X = forward_thread_data[dataset_index].X;
            double ***Y = forward_thread_data[dataset_index].Y;

            double ***grad_w = malloc(layer_sizes_rows * sizeof(double**));
            double ***grad_x = malloc(layer_sizes_rows * sizeof(double**));
            double **grad_b = malloc(layer_sizes_rows * sizeof(double*));

            double n_inputs_double = layer_sizes[0 * layer_sizes_cols + 0];
            double n_neurons_double = layer_sizes[0 * layer_sizes_cols + 1];
            int n_inputs = (int)n_inputs_double;
            int n_neurons = (int)n_neurons_double;

            double *target = malloc(dataset_targets_cols * sizeof(double));
            for (int i = 0; i < dataset_targets_cols; i++) {
                target[i] = targets[dataset_index][i];
            }


            backward_thread_data[dataset_index].dataset_index = dataset_index;
            backward_thread_data[dataset_index].weights = weights;
            backward_thread_data[dataset_index].X = X;
            backward_thread_data[dataset_index].Y = Y;
            backward_thread_data[dataset_index].target = target;
            backward_thread_data[dataset_index].grad_w = grad_w;
            backward_thread_data[dataset_index].grad_x = grad_x;
            backward_thread_data[dataset_index].grad_b = grad_b;
            backward_thread_data[dataset_index].layer_sizes = layer_sizes;
            backward_thread_data[dataset_index].layer_sizes_rows = layer_sizes_rows;
            backward_thread_data[dataset_index].layer_sizes_cols = layer_sizes_cols;
            backward_thread_data[dataset_index].matrix_rows = matrix_rows;
            backward_thread_data[dataset_index].loss = loss;

            backward_thread_data[dataset_index].activations = activations;
            backward_thread_data[dataset_index].threading = threading;
            backward_thread_data[dataset_index].num_cpu = num_cpu;
            backward_thread_data[dataset_index].epoch_losses = epoch_losses;
            backward_thread_data[dataset_index].regression = regression;

            pthread_create(&backward_threads[dataset_index], NULL, backward, &backward_thread_data[dataset_index]);

        }
        for (int t = 0; t < num_threads; ++t) {
            pthread_join(backward_threads[t], NULL);
        }

        // Update weights and biases
        for (int dataset_index = 0; dataset_index < dataset_samples_rows; ++dataset_index) {
            double ***X = backward_thread_data[dataset_index].X;
            double ***Y = backward_thread_data[dataset_index].Y;
            double ***weights = backward_thread_data[dataset_index].weights;
            double ***grad_w = backward_thread_data[dataset_index].grad_w;
            double ***grad_x = backward_thread_data[dataset_index].grad_x;
            double **grad_b = backward_thread_data[dataset_index].grad_b;
            
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
    forward(sample, sample_rows, sample_cols, weights, biases, Y, layer_sizes, layer_sizes_rows, layer_sizes_cols, activations, threading, num_cpu);


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
    double *predictions,
    int threading,
    int num_cpu) {

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
        forward(sample, dataset_samples_cols, dataset_samples_depth, weights, biases, Y, layer_sizes, layer_sizes_rows, layer_sizes_cols, activations, threading, num_cpu);

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
