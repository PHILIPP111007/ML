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
    float *losses,
    int gpu) {
    
    if (random_state != -1) {
        srand(random_state); // set the initial state of the generator
    }

    // Loading a dataset
    float ***__restrict samples = malloc(dataset_samples_rows * sizeof(float**));
    float **__restrict targets = malloc(dataset_targets_rows * sizeof(float*)); 

    for (register int dataset_index = 0; dataset_index < dataset_samples_rows; ++dataset_index) {
        samples[dataset_index] = create_matrix(dataset_samples_cols, dataset_samples_depth);
        for (int i = 0; i < dataset_samples_cols; i++) {

            #pragma omp simd
            for (register int j = 0; j < dataset_samples_depth; j++) {
                int index = dataset_index * dataset_samples_cols * dataset_samples_depth + i * dataset_samples_depth + j;
                samples[dataset_index][i][j] = (float)dataset_samples[index];
            }
        }
    }

    for (register int i = 0; i < dataset_targets_rows; i++) {
        targets[i] = malloc(dataset_targets_cols * sizeof(float));

        #pragma omp simd
        for (register int j = 0; j < dataset_targets_cols; j++) {
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

    // Create Adam optimizer
    struct AdamOptimizer *opt = create_adam(learning_rate, 0.9, 0.999, 1e-8, layer_sizes, layer_sizes_rows, layer_sizes_cols);

    // Preparing the GPU Kernel

    // Step 1: Get a platform and device
    cl_uint numPlatforms;
    cl_platform_id platforms[10];
    clGetPlatformIDs(10, platforms, &numPlatforms);

    // We use the first suitable device of the GPU type
    cl_device_id devices[10];
    cl_uint numDevices;
    clGetDeviceIDs(*platforms, CL_DEVICE_TYPE_GPU, 10, devices, &numDevices);

    if (verbose && gpu) {
        printf("Number of platforms: %u\n", numPlatforms);
        printf("Number of GPUs: %u\n", numDevices);
    }

    // Step 2: Create context
    cl_int err_code;
    cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[0], 0 };
    cl_context context = clCreateContext(properties, 1, devices, NULL, NULL, &err_code);
    if (context == NULL || err_code != CL_SUCCESS) {
        fprintf(stderr, "Error creating OpenCL context: %d\n", err_code);
        exit(EXIT_FAILURE);
    }

    // Step 3: Create a command queue
    cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, NULL);

    // Step 3: Read and compile the OpenCL kernel
    FILE* fp_matmul_gpu = fopen("src/matmul_gpu.cl", "rb");
    fseek(fp_matmul_gpu, 0, SEEK_END);
    long file_size_matmul_gpu = ftell(fp_matmul_gpu);
    rewind(fp_matmul_gpu);
    char* source_matmul_gpu = (char*)malloc(file_size_matmul_gpu + 1);
    fread(source_matmul_gpu, 1, file_size_matmul_gpu, fp_matmul_gpu);
    fclose(fp_matmul_gpu);

    FILE* fp_adam_step_gpu = fopen("src/adam_step_gpu.cl", "rb");
    fseek(fp_adam_step_gpu, 0, SEEK_END);
    long file_size_adam_step_gpu = ftell(fp_adam_step_gpu);
    rewind(fp_adam_step_gpu);
    char* source_adam_step_gpu = (char*)malloc(file_size_adam_step_gpu + 1);
    fread(source_adam_step_gpu, 1, file_size_adam_step_gpu, fp_adam_step_gpu);
    fclose(fp_adam_step_gpu);

    cl_program program_matmul_gpu = clCreateProgramWithSource(context, 1, (const char**)&source_matmul_gpu, NULL, NULL);
    clBuildProgram(program_matmul_gpu, 1, devices, "-cl-fast-relaxed-math", NULL, NULL);

    cl_program program_adam_step_gpu = clCreateProgramWithSource(context, 1, (const char**)&source_adam_step_gpu, NULL, NULL);
    clBuildProgram(program_adam_step_gpu, 1, devices, "-cl-fast-relaxed-math", NULL, NULL);

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

        // Create weights_vec_buffer and weights_transposed_vec_buffer

        float ***weights_transposed = malloc(layer_sizes_rows * sizeof(float**));
        
        for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
            const int n_inputs = (int)layer_sizes[layer_index * layer_sizes_cols];
            const int n_neurons = (int)layer_sizes[layer_index * layer_sizes_cols + 1];

            weights_transposed[layer_index] = transpose(weights[layer_index], n_inputs, n_neurons);
        }

        float *weights_vec = get_weights_vec(weights, layer_sizes_rows, layer_sizes_cols, layer_sizes);
        float *weights_transposed_vec = get_weights_transposed_vec(weights_transposed, layer_sizes_rows, layer_sizes_cols, layer_sizes);
        cl_mem weights_vec_buf = get_weights_vec_buf(weights_vec, layer_sizes_rows, layer_sizes_cols, layer_sizes, context);
        cl_mem weights_transposed_vec_buf = get_weights_vec_buf(weights_transposed_vec, layer_sizes_rows, layer_sizes_cols, layer_sizes, context);

        free(weights_vec);
        free(weights_transposed_vec);

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
            num_threads,
            gpu,
            context,
            queue,
            program_matmul_gpu,
            weights_transposed_vec_buf
        );

        clReleaseMemObject(weights_vec_buf);
        clReleaseMemObject(weights_transposed_vec_buf);

        free(backward_thread_data);

        // Update weights and biases

        if (verbose) {
            logger_info("Update weights and biases step\n");
        }

        if (gpu) {
            adam_step_gpu(opt, weights, grad_w_list, dataset_samples_rows, layer_sizes, layer_sizes_rows, layer_sizes_cols, max_change, context, queue, program_adam_step_gpu);
        } else {
            for (int dataset_index = 0; dataset_index < dataset_samples_rows; dataset_index++) {
                float ***__restrict grad_w = grad_w_list[dataset_index];
                adam_step(opt, weights, grad_w, layer_sizes, layer_sizes_rows, layer_sizes_cols, max_change);
            }
        }

        for (int dataset_index = 0; dataset_index < dataset_samples_rows; dataset_index++) {
            float **__restrict grad_b = grad_b_list[dataset_index];

            for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
                const int n_neurons = (int)layer_sizes[layer_index * layer_sizes_cols + 1];
                float *__restrict bias_layer = biases[layer_index];
                float *__restrict grad_b_layer = grad_b[layer_index];

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
                    weights[layer_index][i][j] = isnan(weights[layer_index][i][j]) ? 0.0f : weights[layer_index][i][j];
                }
            }
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
    // Save weights and biases in files
    char *file_weights = "weights.json";
    save_weights_as_json(file_weights, weights, layer_sizes, layer_sizes_rows, layer_sizes_cols);
    char *file_biases = "biases.json";
    save_biases_as_json(file_biases, biases, layer_sizes, layer_sizes_rows, layer_sizes_cols);

    // Clearing memory

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseProgram(program_matmul_gpu);

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
    float *prediction,
    int gpu) {

    float **__restrict sample = create_matrix(sample_rows, sample_cols);
    for (register int i = 0; i < sample_rows; i++) {
        #pragma omp simd
        for (register int j = 0; j < sample_cols; j++) {
            int index = i * sample_cols + j;
            sample[i][j] = sample_input[index];
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
        for (int i = 0; i < n_inputs; i++) {
            #pragma omp simd
            for (int j = 0; j < n_neurons; j++) {
                int index = current_weight_offset + i * n_neurons + j;
                weights[layer_index][i][j] = weights_input[index];
            }
        }

        biases[layer_index] = malloc(n_neurons * sizeof(float));

        #pragma omp simd
        for (int i = 0; i < n_neurons; i++) {
            int index = total_bias_count + i;
            biases[layer_index][i] = biases_input[index];
        }

        current_weight_offset += n_inputs * n_neurons;
        total_bias_count += n_neurons;
    }

    float ***__restrict Y = malloc(layer_sizes_rows * sizeof(float**));

    // Preparing the GPU Kernel

    // Step 1: Get a platform and device
    cl_uint numPlatforms;
    cl_platform_id platforms[10];
    clGetPlatformIDs(10, platforms, &numPlatforms);

    // We use the first suitable device of the GPU type
    cl_device_id devices[10];
    cl_uint numDevices;
    clGetDeviceIDs(*platforms, CL_DEVICE_TYPE_GPU, 10, devices, &numDevices);

    // Step 2: Create context
    cl_int err_code;
    cl_context context = clCreateContext(NULL, 1, devices, NULL, NULL, &err_code);
    if (context == NULL || err_code != CL_SUCCESS) {
        fprintf(stderr, "Error creating OpenCL context: %d\n", err_code);
        exit(EXIT_FAILURE);
    }

    // Step 3: Create a command queue
    cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, NULL);

    // Step 3: Read and compile the OpenCL kernel
    FILE* fp = fopen("src/matmul_gpu.cl", "rb");
    fseek(fp, 0, SEEK_END);
    long fileSize = ftell(fp);
    rewind(fp);
    char* sourceStr = (char*)malloc(fileSize + 1);
    fread(sourceStr, 1, fileSize, fp);
    fclose(fp);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&sourceStr, NULL, NULL);
    clBuildProgram(program, 1, devices, "-cl-fast-relaxed-math", NULL, NULL);

    // Forward pass

    float *weights_vec = get_weights_vec(weights, layer_sizes_rows, layer_sizes_cols, layer_sizes);
    cl_mem weights_vec_buf = get_weights_vec_buf(weights_vec, layer_sizes_rows, layer_sizes_cols, layer_sizes, context);

    forward(sample, sample_rows, sample_cols, weights, biases, Y, layer_sizes, layer_sizes_rows, layer_sizes_cols, activations, 1, gpu, context, queue, program, weights_vec_buf);

    // Free memory

    clReleaseMemObject(weights_vec_buf);
    free(weights_vec);

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseProgram(program);

    free_matrix(sample);

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
    for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
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
    int num_cpu,
    int gpu) {

    // Loading a dataset
    float ***__restrict samples = malloc(dataset_samples_rows * sizeof(float**));
    for (int dataset_index = 0; dataset_index < dataset_samples_rows; dataset_index++) {
        samples[dataset_index] = create_matrix(dataset_samples_cols, dataset_samples_depth);
        for (int i = 0; i < dataset_samples_cols; i++) {
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
            #pragma omp simd
            for (int j = 0; j < n_neurons; ++j) {
                int index = current_weight_offset + i * n_neurons + j;
                weights[layer_index][i][j] = weights_input[index];
            }
        }

        biases[layer_index] = malloc(n_neurons * sizeof(float));

        #pragma omp simd
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

    // Preparing the GPU Kernel

    // Step 1: Get a platform and device
    cl_uint numPlatforms;
    cl_platform_id platforms[10];
    clGetPlatformIDs(10, platforms, &numPlatforms);

    // We use the first suitable device of the GPU type
    cl_device_id devices[10];
    cl_uint numDevices;
    clGetDeviceIDs(*platforms, CL_DEVICE_TYPE_GPU, 10, devices, &numDevices);

    // Step 2: Create context
    cl_int err_code;
    cl_context context = clCreateContext(NULL, 1, devices, NULL, NULL, &err_code);
    if (context == NULL || err_code != CL_SUCCESS) {
        fprintf(stderr, "Error creating OpenCL context: %d\n", err_code);
        exit(EXIT_FAILURE);
    }

    // Step 3: Create a command queue
    cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, NULL);

    // Step 3: Read and compile the OpenCL kernel
    FILE* fp = fopen("src/matmul_gpu.cl", "rb");
    fseek(fp, 0, SEEK_END);
    long fileSize = ftell(fp);
    rewind(fp);
    char* sourceStr = (char*)malloc(fileSize + 1);
    fread(sourceStr, 1, fileSize, fp);
    fclose(fp);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&sourceStr, NULL, NULL);
    clBuildProgram(program, 1, devices, "-cl-fast-relaxed-math", NULL, NULL);

    float *weights_vec = get_weights_vec(weights, layer_sizes_rows, layer_sizes_cols, layer_sizes);
    cl_mem weights_vec_buf = get_weights_vec_buf(weights_vec, layer_sizes_rows, layer_sizes_cols, layer_sizes, context);

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

    // Free memory

    clReleaseMemObject(weights_vec_buf);
    free(weights_vec);

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseProgram(program);

    free(threads);
    free(tasks);

    for (int layer_index = 0; layer_index < layer_sizes_rows; layer_index++) {
        const int n_inputs = (int)layer_sizes[layer_index * layer_sizes_cols];

        free(weights[layer_index]);
        free(biases[layer_index]);
    }
    free(weights);
    free(biases);

    for (register int dataset_index = 0; dataset_index < dataset_samples_rows; dataset_index++) {
        for (register int i = 0; i < dataset_samples_cols; i++) {
            free(samples[dataset_index][i]);
        }
        free(samples[dataset_index]);
    }
    free(samples);
}
