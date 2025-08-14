#include "linear.h"


void linear_fit(
    float ***samples,
    int dataset_samples_rows,
    int dataset_samples_cols,
    int dataset_samples_depth,
    float **targets,
    int dataset_targets_cols,
    float ***weights,
    float **biases,
    float *losses,
    int loss,
    float *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols,
    float *activations,
    float *dropouts,
    int epoch,
    int n_epoch,
    int verbose,
    int regression,
    int max_change,
    float learning_rate,
    int num_threads,
    int gpu,
    struct AdamOptimizer *opt,
    cl_context context,
    cl_command_queue queue,
    cl_program program_matmul_gpu,
    cl_program program_adam_step_gpu
) {
    float *epoch_losses = malloc(dataset_samples_rows * sizeof(float));
    check_if_null((float *)epoch_losses, "epoch_losses");
    int matrix_rows = dataset_samples_cols;

    // Forward pass

    if (verbose) {
        logger_info("Forward step.\n");
    }

    ForwardData *forward_thread_data = malloc(num_threads * sizeof(ForwardData));
    BackwardData *backward_thread_data = malloc(num_threads * sizeof(BackwardData));

    float ****__restrict X_list_intermediate = malloc(dataset_samples_rows * sizeof(float***));
    check_if_null((float *)X_list_intermediate, "X_list_intermediate");
    float ****__restrict Y_list_intermediate = malloc(dataset_samples_rows * sizeof(float***));
    check_if_null((float *)Y_list_intermediate, "Y_list_intermediate");
    float ****__restrict X_list = malloc(dataset_samples_rows * sizeof(float***));
    check_if_null((float *)X_list, "X_list");
    float ****__restrict Y_list = malloc(dataset_samples_rows * sizeof(float***));
    check_if_null((float *)Y_list, "Y_list");

    // Create weights_vec_buffer and weights_transposed_vec_buffer

    float ***weights_transposed = malloc(layer_sizes_rows * sizeof(float**));
    check_if_null((float *)weights_transposed, "weights_transposed");
    for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
        const int n_inputs = (int)layer_sizes[layer_index * layer_sizes_cols];
        const int n_neurons = (int)layer_sizes[layer_index * layer_sizes_cols + 1];
        weights_transposed[layer_index] = transpose(weights[layer_index], n_inputs, n_neurons);
    }

    float *weights_vec;
    float *weights_transposed_vec;
    cl_mem weights_vec_buf;
    cl_mem weights_transposed_vec_buf;

    if (gpu) {
        weights_vec = get_weights_vec(weights, layer_sizes_rows, layer_sizes_cols, layer_sizes);
        weights_transposed_vec = get_weights_transposed_vec(weights_transposed, layer_sizes_rows, layer_sizes_cols, layer_sizes);
        weights_vec_buf = get_weights_vec_buf(weights_vec, layer_sizes_rows, layer_sizes_cols, layer_sizes, context);
        weights_transposed_vec_buf = get_weights_vec_buf(weights_transposed_vec, layer_sizes_rows, layer_sizes_cols, layer_sizes, context);

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
            num_threads,
            gpu,
            context,
            queue,
            program_matmul_gpu,
            weights_vec_buf
        );

        free(weights_transposed_vec);
    } else {
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
            num_threads,
            gpu,
            NULL,
            NULL,
            NULL,
            NULL
        );
    }

    for (int t = 0; t < num_threads; t++) {
        #pragma omp simd
        for (int dataset_index = 0; dataset_index < dataset_samples_rows; dataset_index++) {
            X_list[dataset_index] = forward_thread_data[t].X_list[dataset_index];
            Y_list[dataset_index] = forward_thread_data[t].Y_list[dataset_index];
        }
    }
    free(X_list_intermediate);
    free(Y_list_intermediate);
    free(forward_thread_data);

    float ****__restrict grad_w_list = malloc(dataset_samples_rows * sizeof(float***));
    check_if_null((float *)grad_w_list, "grad_w_list");
    float ****__restrict grad_x_list = malloc(dataset_samples_rows * sizeof(float***));
    check_if_null((float *)grad_x_list, "grad_x_list");
    float ***__restrict grad_b_list = malloc(dataset_samples_rows * sizeof(float**));
    check_if_null((float *)grad_b_list, "grad_b_list");

    // Backward pass

    if (verbose) {
        logger_info("Backward step.\n");
    }

    if (gpu) {
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
            num_threads,
            gpu,
            context,
            queue,
            program_matmul_gpu,
            weights_transposed_vec_buf
        );
    } else {
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
            num_threads,
            gpu,
            NULL,
            NULL,
            NULL,
            NULL
        );
    }

    // Update weights and biases

    if (verbose) {
        logger_info("Update weights and biases step.\n");
    }

    if (gpu) {
        adam_step_gpu(opt, weights, weights_vec, weights_vec_buf, grad_w_list, dataset_samples_rows, layer_sizes, layer_sizes_rows, layer_sizes_cols, max_change, context, queue, program_adam_step_gpu);
    } else {
        for (int dataset_index = 0; dataset_index < dataset_samples_rows; dataset_index++) {
            float ***grad_w = grad_w_list[dataset_index];
            adam_step(opt, weights, grad_w, layer_sizes, layer_sizes_rows, layer_sizes_cols, max_change);
        }
    }

    if (gpu) {
        free(weights_vec);
        clReleaseMemObject(weights_vec_buf);
        clReleaseMemObject(weights_transposed_vec_buf);
    }

    for (int dataset_index = 0; dataset_index < dataset_samples_rows; dataset_index++) {
        float **grad_b = grad_b_list[dataset_index];

        for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
            const int n_neurons = (int)layer_sizes[layer_index * layer_sizes_cols + 1];
            float *bias_layer = biases[layer_index];
            float *grad_b_layer = grad_b[layer_index];

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

        for (int i = 0; i < n_inputs; i++) {
            for (int j = 0; j < n_neurons; j++) {
                weights[layer_index][i][j] = check_if_isnan(weights[layer_index][i][j]);
            }
        }
        for (int i = 0; i < n_neurons; i++) {
            biases[layer_index][i] = check_if_isnan(biases[layer_index][i]);
        }
    }

    // Clearing memory

    if (verbose) {
        logger_info("Clearing memory.\n");
    }

    free(backward_thread_data);

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

void linear_predict_one(
    float **sample,
    int sample_rows,
    int sample_cols,
    float *prediction,
    float ***weights,
    float **biases,
    float *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols,
    float *activations,
    int gpu,
    cl_context context,
    cl_command_queue queue,
    cl_program program,
    cl_mem weights_vec_buf
) {
    float ***__restrict Y = malloc(layer_sizes_rows * sizeof(float**));
    check_if_null((float *)Y, "Y");

    forward(sample, sample_rows, sample_cols, weights, biases, Y, layer_sizes, layer_sizes_rows, layer_sizes_cols, activations, gpu, context, queue, program, weights_vec_buf);

    const int n_inputs = (int)layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols];
    const int n_neurons = (int)layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols + 1];

    int matrix_rows = sample_rows;

    float **__restrict y = create_matrix(matrix_rows, n_neurons);
    for (int i = 0; i < matrix_rows; i++) {
       #pragma omp simd
        for (int j = 0; j < n_neurons; j++) {
            y[i][j] = Y[layer_sizes_rows - 1][i][j];
        }
    }

    // Return predict
    #pragma omp simd
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
}

void linear_predict(
    float ***weights,
    float **biases,
    float ***samples,
    int dataset_samples_rows,
    int dataset_samples_cols,
    int dataset_samples_depth,
    float *predictions,
    float *layer_sizes,
    int layer_sizes_rows,
    int layer_sizes_cols,
    float *activations,
    int num_cpu,
    int gpu,
    cl_context context,
    cl_command_queue queue,
    cl_program program,
    cl_mem weights_vec_buf
) {
    pthread_t *threads = malloc(num_cpu * sizeof(pthread_t));
    PredictTask *tasks = malloc(dataset_samples_rows * sizeof(PredictTask));

    // Get neurons count in last layer
    int n_neurons_last_layer = (int)layer_sizes[(layer_sizes_rows - 1) * layer_sizes_cols + 1];

    // Process samples in parallel
    for (register int dataset_index = 0; dataset_index < dataset_samples_rows; dataset_index++) {
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
        tasks[dataset_index].gpu = gpu;
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
        range->context = context;
        range->queue = queue;
        range->program = program;
        range->weights_vec_buf = weights_vec_buf;

        pthread_create(&threads[i], NULL, predict_thread, range);
    }

    // Wait for all threads to complete
    for (int i = 0; i < num_cpu; i++) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    free(tasks);
}
