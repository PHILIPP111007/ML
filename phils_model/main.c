// clang -shared -o main.so -fPIC -O3 -fopenmp -ffast-math -march=native main.c src/functions.c src/activations.c src/loss.c src/init.c src/json.c src/adam.c src/forward.c src/backward.c src/logger.c src/predict.c
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
#include "src/logger.h"
#include "src/predict.h"


void fit(
    float *dataset_samples,
    float *dataset_targets,
    int dataset_samples_rows,
    int dataset_samples_cols,
    int dataset_samples_depth,
    int dataset_targets_rows,
    int dataset_targets_cols,
    float *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols,
    float *activations,
    int activations_len,
    int loss,
    int n_epoch,
    float learning_rate,
    int verbose,
    float max_change,
    int random_state,
    int regression,
    int num_cpu,
    float *dropouts,
    float *losses) {
    
    if (random_state != -1) {
        srand(random_state); // set the initial state of the generator
    }

    // Loading a dataset
    float ***__restrict samples = malloc(dataset_samples_rows * sizeof(float**));
    float **__restrict targets = malloc(dataset_targets_rows * sizeof(float*)); 

    for (int dataset_index = 0; dataset_index < dataset_samples_rows; ++dataset_index) {
        samples[dataset_index] = malloc(dataset_samples_cols * sizeof(float*));
        for (int i = 0; i < dataset_samples_cols; i++) {
            samples[dataset_index][i] = malloc(dataset_samples_depth * sizeof(float));
            for (int j = 0; j < dataset_samples_depth; j++) {
                int index = dataset_index * dataset_samples_cols * dataset_samples_depth + i * dataset_samples_depth + j;
                samples[dataset_index][i][j] = (float)dataset_samples[index];
            }
        }
    }

    for (int i = 0; i < dataset_targets_rows; i++) {
        targets[i] = malloc(dataset_targets_cols * sizeof(float));
        for (int j = 0; j < dataset_targets_cols; j++) {
            targets[i][j] = (float)dataset_targets[i * dataset_targets_cols + j];
        }
    }

    // Initialization of biases and weights
    float **__restrict biases = malloc(layer_sizes_rows * sizeof(float*));
    float ***__restrict weights = malloc(layer_sizes_rows * sizeof(float**));

    for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
        const int n_inputs = (int)layer_sizes[layer_index * layer_sizes_cols];
        const int n_neurons = (int)layer_sizes[layer_index * layer_sizes_cols + 1];

        // Initialize biases
        biases[layer_index] = init_bias(n_neurons, n_inputs);

        // Initialize weights
        weights[layer_index] = init_weights(n_neurons, n_inputs);
    }

    // Training

    // Create Adam
    struct AdamOptimizer *opt = create_adam(learning_rate, 0.9, 0.999, 1e-8, layer_sizes, layer_sizes_rows, layer_sizes_cols);

    for (int epoch = 0; epoch < n_epoch; epoch++) {
        float *epoch_losses = malloc(dataset_samples_rows * sizeof(float));
        int matrix_rows = dataset_samples_cols;

        const int num_threads = (dataset_samples_rows < num_cpu) ? dataset_samples_rows : num_cpu;
        ForwardData *forward_thread_data = malloc(num_threads * sizeof(ForwardData));
        BackwardData *backward_thread_data = malloc(num_threads * sizeof(BackwardData));

        
        // Forward pass

        if (verbose) {
            logger_info("Forward step\n");
        }

        float ****__restrict X_list_intermediate = malloc(dataset_samples_rows * sizeof(float***));
        float ****__restrict Y_list_intermediate = malloc(dataset_samples_rows * sizeof(float***));
        float ****__restrict X_list = malloc(dataset_samples_rows * sizeof(float***));
        float ****__restrict Y_list = malloc(dataset_samples_rows * sizeof(float***));

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
            dropouts,
            num_threads
        );

        for (int t = 0; t < num_threads; t++) {
            for (int dataset_index = 0; dataset_index < dataset_samples_rows; dataset_index++) {
                X_list[dataset_index] = forward_thread_data[t].X_list[dataset_index];
                Y_list[dataset_index] = forward_thread_data[t].Y_list[dataset_index];
            }
        }
        free(forward_thread_data);

        float ****__restrict grad_w_list = malloc(dataset_samples_rows * sizeof(float***));
        float ****__restrict grad_x_list = malloc(dataset_samples_rows * sizeof(float***));
        float ***__restrict grad_b_list = malloc(dataset_samples_rows * sizeof(float**));

        // Backward pass

        if (verbose) {
            logger_info("Backward step\n");
        }

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

        free(backward_thread_data);

        // Update weights and biases

        if (verbose) {
            logger_info("Update weights and biases step\n");
        }

        #pragma omp for schedule(static)
        for (int dataset_index = 0; dataset_index < dataset_samples_rows; dataset_index++) {
            float ***__restrict grad_w = grad_w_list[dataset_index];
            float **__restrict grad_b = grad_b_list[dataset_index];

            adam_step(opt, weights, grad_w, layer_sizes, layer_sizes_rows, layer_sizes_cols, max_change);

            for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
                int n_neurons = (int)layer_sizes[layer_index * layer_sizes_cols + 1];
                float *__restrict bias_layer = biases[layer_index];
                float *__restrict grad_b_layer = grad_b[layer_index];

                #pragma omp parallel for schedule(static)
                for (int i = 0; i < n_neurons; ++i) {
                    const float change = grad_b_layer[i] * learning_rate;
                    bias_layer[i] -= safe_update(change, max_change);
                }
            }
        }

        // Handle NaN

        for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
            const int n_inputs = (int)layer_sizes[layer_index * layer_sizes_cols];
            const int n_neurons = (int)layer_sizes[layer_index * layer_sizes_cols + 1];

            #pragma omp parallel for collapse(2) schedule(static)
            for (int i = 0; i < n_inputs; i++) {
                for (int j = 0; j < n_neurons; j++) {
                    weights[layer_index][i][j] = isnan(weights[layer_index][i][j]) ? 0.0f : weights[layer_index][i][j];
                }
            }
        }

        for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
            const int n_neurons = (int)layer_sizes[layer_index * layer_sizes_cols + 1];

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n_neurons; i++) {
                biases[layer_index][i] = isnan(biases[layer_index][i]) ? 0.0f : biases[layer_index][i];
            }
        }

        // Clearing memory

        for (int dataset_index = 0; dataset_index < dataset_samples_rows; dataset_index++) {
            for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
                const int n_inputs = (int)layer_sizes[layer_index * layer_sizes_cols];

                for (int i = 0; i < matrix_rows; i++) {
                    free(X_list[dataset_index][layer_index][i]);
                    free(Y_list[dataset_index][layer_index][i]);
                }
                free(grad_w_list[dataset_index][layer_index]);
                free(grad_x_list[dataset_index][layer_index]);
                free(grad_b_list[dataset_index][layer_index]);

                free(X_list[dataset_index][layer_index]);
                free(Y_list[dataset_index][layer_index]);
            }
            free(grad_w_list[dataset_index]);
            free(grad_x_list[dataset_index]);
            free(grad_b_list[dataset_index]);

            free(X_list[dataset_index]);
            free(Y_list[dataset_index]);
        }
        free(grad_w_list);
        free(grad_x_list);
        free(grad_b_list);

        free(X_list);
        free(Y_list);

        float mean_loss = mean(epoch_losses, dataset_samples_rows);
        losses[epoch] = mean_loss;
        if (verbose) {
            char *s = (char*)malloc(100 * sizeof(char));
            sprintf(s, "Epoch %d / %d. Loss: %f\n", epoch + 1, n_epoch, mean_loss);
            logger_info(s);
        }
    }
    char *file_weights = "weights.json";
    save_weights_as_json(file_weights, weights, layer_sizes, layer_sizes_rows, layer_sizes_cols);
    char *file_biases = "biases.json";
    save_biases_as_json(file_biases, biases, layer_sizes, layer_sizes_rows, layer_sizes_cols);

    // Clearing memory

    for (int dataset_index = 0; dataset_index < dataset_samples_rows; dataset_index++) {
        for (int i = 0; i < dataset_samples_cols; ++i) {
            free(samples[dataset_index][i]);
        }
        free(samples[dataset_index]);
    }
    free(samples);

    destroy_adam(opt, layer_sizes, layer_sizes_rows, layer_sizes_cols);

    for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
        free(weights[layer_index]);
        free(biases[layer_index]);
    }
    free(biases);
    free(weights);
}

void predict_one(
    float *sample_input,
    int sample_rows,
    int sample_cols,
    float *weights_input,
    float *biases_input,
    float *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols,
    float *activations,
    int activations_len,
    float *prediction) {

    float **__restrict sample = create_matrix(sample_rows, sample_cols);
    for (int i = 0; i < sample_rows; i++) {
        for (int j = 0; j < sample_cols; j++) {
            sample[i][j] = sample_input[i + j];
        }
    }

    float ***__restrict weights = malloc(layer_sizes_rows * sizeof(float**));
    float **__restrict biases = malloc(layer_sizes_rows * sizeof(float*));

    int current_weight_offset = 0;
    int total_bias_count = 0;
    for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
        const int n_inputs = (int)layer_sizes[layer_index * layer_sizes_cols];
        const int n_neurons = (int)layer_sizes[layer_index * layer_sizes_cols + 1];

        weights[layer_index] = malloc(n_inputs * sizeof(float*));
        for (int i = 0; i < n_inputs; i++) {
            weights[layer_index][i] = malloc(n_neurons * sizeof(float));
            for (int j = 0; j < n_neurons; j++) {
                int index = current_weight_offset + i * n_neurons + j;
                weights[layer_index][i][j] = weights_input[index];
            }
        }

        biases[layer_index] = malloc(n_neurons * sizeof(float));
        for (int i = 0; i < n_neurons; i++) {
            int index = total_bias_count + i;
            biases[layer_index][i] = biases_input[index];
        }

        current_weight_offset += n_inputs * n_neurons;
        total_bias_count += n_neurons;
    }

    float ***__restrict Y = malloc(layer_sizes_rows * sizeof(float**));

    // Forward pass
    forward(sample, sample_rows, sample_cols, weights, biases, Y, layer_sizes, layer_sizes_rows, layer_sizes_cols, activations);

    free_matrix(sample);

    const int n_inputs = (int)layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols];
    const int n_neurons = (int)layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols + 1];

    int matrix_rows = sample_rows;

    float **__restrict y = create_matrix(matrix_rows, n_neurons);
    for (int i = 0; i < matrix_rows; i++) {
        for (int j = 0; j < n_neurons; j++) {
            y[i][j] = Y[layer_sizes_rows - 1][i][j];
        }
    }

    // Return predict
    for (int j = 0; j < n_neurons; j++) {
        prediction[j] = y[0][j];
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
    for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
        const int n_inputs = (int)layer_sizes[layer_index * layer_sizes_cols];

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
    float *dataset_samples,
    int dataset_samples_rows,
    int dataset_samples_cols,
    int dataset_samples_depth,
    float *weights_input,
    float *biases_input,
    float *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols,
    float *activations,
    int activations_len,
    float *predictions,
    int num_cpu) {

    // Loading a dataset
    float ***__restrict samples = malloc(dataset_samples_rows * sizeof(float**));
    for (int dataset_index = 0; dataset_index < dataset_samples_rows; dataset_index++) {
        samples[dataset_index] = malloc(dataset_samples_cols * sizeof(float*));
        for (int i = 0; i < dataset_samples_cols; i++) {
            samples[dataset_index][i] = malloc(dataset_samples_depth * sizeof(float));
            for (int j = 0; j < dataset_samples_depth; j++) {
                samples[dataset_index][i][j] = (float)dataset_samples[dataset_index * dataset_samples_cols * dataset_samples_depth + i * dataset_samples_depth + j];
            }
        }
    }

    float ***__restrict weights = malloc(layer_sizes_rows * sizeof(float**));
    float **__restrict biases = malloc(layer_sizes_rows * sizeof(float*));

    int current_weight_offset = 0;
    int total_bias_count = 0;
    for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
        const int n_inputs = (int)layer_sizes[layer_index * layer_sizes_cols];
        const int n_neurons = (int)layer_sizes[layer_index * layer_sizes_cols + 1];

        weights[layer_index] = create_matrix(n_inputs, n_neurons);
        for (int i = 0; i < n_inputs; ++i) {
            for (int j = 0; j < n_neurons; ++j) {
                int index = current_weight_offset + i * n_neurons + j;
                weights[layer_index][i][j] = weights_input[index];
            }
        }

        biases[layer_index] = malloc(n_neurons * sizeof(float));
        for (int i = 0; i < n_neurons; ++i) {
            int index = total_bias_count + i;
            biases[layer_index][i] = biases_input[index];
        }

        current_weight_offset += n_inputs * n_neurons;
        total_bias_count += n_neurons;
    }

    if (num_cpu > dataset_samples_rows) {
        num_cpu = dataset_samples_rows;
    }

    pthread_t *threads = malloc(num_cpu * sizeof(pthread_t));
    PredictTask *tasks = malloc(dataset_samples_rows * sizeof(PredictTask));

    // Get neurons count in last layer
    int n_neurons_last_layer = (int)layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols + 1];

    // Process samples in parallel
    for (int dataset_index = 0; dataset_index < dataset_samples_rows; dataset_index++) {
        tasks[dataset_index].sample = samples[dataset_index];
        tasks[dataset_index].weights = weights;
        tasks[dataset_index].biases = biases;
        tasks[dataset_index].layer_sizes = layer_sizes;
        tasks[dataset_index].layer_sizes_rows = layer_sizes_rows;
        tasks[dataset_index].layer_sizes_cols = layer_sizes_cols;
        tasks[dataset_index].activations = activations;
        tasks[dataset_index].predictions = predictions;
        tasks[dataset_index].dataset_index = dataset_index;
        tasks[dataset_index].dataset_samples_cols = dataset_samples_cols;
        tasks[dataset_index].dataset_samples_depth = dataset_samples_depth;
        tasks[dataset_index].n_neurons_last_layer = n_neurons_last_layer;
    }

    // Create threads
    int samples_per_thread = dataset_samples_rows / num_cpu;
    int remaining_samples = dataset_samples_rows % num_cpu;

    for (int i = 0; i < num_cpu; i++) {
        int start_idx = i * samples_per_thread + (i < remaining_samples ? i : remaining_samples);
        int end_idx = start_idx + samples_per_thread + (i < remaining_samples ? 1 : 0);

        ThreadRange *range = malloc(sizeof(ThreadRange));
        range->start = start_idx;
        range->end = end_idx;
        range->tasks = tasks;

        pthread_create(&threads[i], NULL, predict_thread, range);
    }

    // Wait for all threads to complete
    for (int i = 0; i < num_cpu; i++) {
        pthread_join(threads[i], NULL);
    }

    // Free memory
    free(threads);
    free(tasks);

    for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
        const int n_inputs = (int)layer_sizes[layer_index * layer_sizes_cols];

        free(weights[layer_index]);
        free(biases[layer_index]);
    }
    free(weights);
    free(biases);

    for (int dataset_index = 0; dataset_index < dataset_samples_rows; dataset_index++) {
        for (int i = 0; i < dataset_samples_cols; i++) {
            free(samples[dataset_index][i]);
        }
        free(samples[dataset_index]);
    }
    free(samples);
}
