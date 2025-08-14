// clang -shared -o main.so -fPIC -O3 -fopenmp -ffast-math -march=native main.c src/functions.c src/activations.c src/loss.c src/init.c src/json.c src/adam.c src/forward.c src/backward.c src/logger.c src/predict.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include "src/linear.h"


void fit(
    float *dataset_samples,
    float *dataset_targets,
    int dataset_samples_rows,
    int dataset_samples_cols,
    int dataset_samples_depth,
    int dataset_targets_rows,
    int dataset_targets_cols,
    float *linear_layer_sizes,
    int linear_layer_sizes_rows,
    int linear_layer_sizes_cols,
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
    int gpu,
    int large_matrices,
    const char *layers[],
    int number_of_layers
) {
    if (random_state != -1) {
        srand(random_state); // set the initial state of the generator
    }

    // Loading a dataset
    float ***__restrict samples = malloc(dataset_samples_rows * sizeof(float**));
    check_if_null((float *)samples, "samples");
    float **__restrict targets = malloc(dataset_targets_rows * sizeof(float*)); 
    check_if_null((float *)targets, "targets");

    for (register int dataset_index = 0; dataset_index < dataset_samples_rows; ++dataset_index) {
        samples[dataset_index] = create_matrix(dataset_samples_cols, dataset_samples_depth);
        for (int i = 0; i < dataset_samples_cols; i++) {
            #pragma omp simd
            for (register int j = 0; j < dataset_samples_depth; j++) {
                int index = dataset_index * dataset_samples_cols * dataset_samples_depth + i * dataset_samples_depth + j;
                check_if_index_out_of_bounds(index, get_array_size(dataset_samples), "dataset_samples");
                samples[dataset_index][i][j] = (float)dataset_samples[index];
            }
        }
    }

    for (register int i = 0; i < dataset_targets_rows; i++) {
        targets[i] = malloc(dataset_targets_cols * sizeof(float));
        check_if_null((float *)targets[i], "targets[i]");
        #pragma omp simd
        for (register int j = 0; j < dataset_targets_cols; j++) {
            int index = i * dataset_targets_cols + j;

            check_if_index_out_of_bounds(index, get_array_size(dataset_targets), "dataset_targets");

            targets[i][j] = (float)dataset_targets[index];
        }
    }

    // Initialization of biases and weights
    float **__restrict biases = malloc(linear_layer_sizes_rows * sizeof(float*));
    check_if_null((float *)biases, "biases");
    float ***__restrict weights = malloc(linear_layer_sizes_rows * sizeof(float**));
    check_if_null((float *)weights, "weights");

    for (int layer_index = 0; layer_index < linear_layer_sizes_rows; layer_index++) {
        const int n_inputs = (int)linear_layer_sizes[layer_index * linear_layer_sizes_cols];
        const int n_neurons = (int)linear_layer_sizes[layer_index * linear_layer_sizes_cols + 1];

        // Initialize biases
        biases[layer_index] = init_bias(n_neurons, n_inputs);

        // Initialize weights
        weights[layer_index] = init_weights(n_neurons, n_inputs);
    }

    // Training

    // Create Adam optimizer
    struct AdamOptimizer *opt = create_adam(learning_rate, 0.9, 0.999, 1e-8, linear_layer_sizes, linear_layer_sizes_rows, linear_layer_sizes_cols);

    const int num_threads = (dataset_samples_rows < num_cpu) ? dataset_samples_rows : num_cpu;

    // Preparing the GPU Kernel

    // Step 1: Get a platform and device
    cl_uint num_platforms;
    cl_platform_id platforms[10];
    clGetPlatformIDs(10, platforms, &num_platforms);
    if (verbose && num_platforms == 0 && gpu) {
        logger_error("Number of platforms is 0. Returning to CPU.");
        gpu = 0;
    }

    cl_device_id devices[10];
    cl_uint num_devices;
    if (gpu) {
        // We use the first suitable device of the GPU type
        clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 10, devices, &num_devices);
        if (verbose) {
            printf("Number of platforms: %u\n", num_platforms);
            printf("Number of GPUs: %u\n", num_devices);
        }
    }

    // Step 2: Create context
    cl_int err_code;
    cl_context context;
    if (gpu) {
        context = clCreateContext(NULL, 1, devices, NULL, NULL, &err_code);
        if (context == NULL || err_code != CL_SUCCESS) {
            fprintf(stderr, "Error creating OpenCL context: %d\n", err_code);
            exit(EXIT_FAILURE);
        }
    }

    // Step 3: Create a command queue
    cl_command_queue queue;
    if (gpu) {
        #ifdef __APPLE__
            queue = clCreateCommandQueue(context, devices[0], 0, NULL);
        #else
            queue = clCreateCommandQueueWithProperties(context, devices[0], 0, NULL);
        #endif
    }

    char *source_matmul_gpu;
    char *source_adam_step_gpu;
    cl_program program_matmul_gpu;
    cl_program program_adam_step_gpu;
    if (gpu) {
        source_matmul_gpu = get_file_content("src/src/matmul_gpu.cl");
        source_adam_step_gpu = get_file_content("src/src/adam_step_gpu.cl");

        program_matmul_gpu = clCreateProgramWithSource(context, 1, (const char**)&source_matmul_gpu, NULL, NULL);
        clBuildProgram(program_matmul_gpu, 1, devices, "-cl-fast-relaxed-math", NULL, NULL);
    
        program_adam_step_gpu = clCreateProgramWithSource(context, 1, (const char**)&source_adam_step_gpu, NULL, NULL);
        clBuildProgram(program_adam_step_gpu, 2, devices, "-cl-fast-relaxed-math", NULL, NULL);
    }

    // Iteration over epochs
    for (int epoch = 0; epoch < n_epoch; epoch++) {

        // Iterating through neural network layers
        for (int layer_index = 0; layer_index < number_of_layers; layer_index++) {

            // If the layer is Linear, then we call linear_fit
            if (strcmp(layers[layer_index], "Linear") == 0) {
                linear_fit(
                    samples,
                    dataset_samples_rows,
                    dataset_samples_cols,
                    dataset_samples_depth,
                    targets,
                    dataset_targets_cols,
                    weights,
                    biases,
                    losses,
                    loss,
                    linear_layer_sizes,
                    linear_layer_sizes_rows,
                    linear_layer_sizes_cols,
                    activations,
                    dropouts,
                    epoch,
                    n_epoch,
                    verbose,
                    regression,
                    max_change,
                    learning_rate,
                    num_threads,
                    gpu,
                    large_matrices,
                    opt,
                    context,
                    queue,
                    program_matmul_gpu,
                    program_adam_step_gpu
                );
            }

            // If the layer is Conv2d, then we call conv2d_fit
            if (strcmp(layers[layer_index], "Conv2d") == 0) {}
        }
    }

    // Save weights and biases in files
    char *file_weights = "weights.json";
    save_weights_as_json(file_weights, weights, linear_layer_sizes, linear_layer_sizes_rows, linear_layer_sizes_cols);
    char *file_biases = "biases.json";
    save_biases_as_json(file_biases, biases, linear_layer_sizes, linear_layer_sizes_rows, linear_layer_sizes_cols);

    // Clearing memory

    if (gpu) {
        clReleaseProgram(program_matmul_gpu);
        clReleaseProgram(program_adam_step_gpu);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    }

    destroy_adam(opt, linear_layer_sizes, linear_layer_sizes_rows, linear_layer_sizes_cols);

    for (int dataset_index = 0; dataset_index < dataset_samples_rows; dataset_index++) {
        for (int i = 0; i < dataset_samples_cols; ++i) {
            free(samples[dataset_index][i]);
        }
        free(samples[dataset_index]);
    }
    free(samples);

    for (int layer_index = 0; layer_index < linear_layer_sizes_rows; layer_index++) {
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
    float *linear_layer_sizes,
    int linear_layer_sizes_rows,
    int linear_layer_sizes_cols,
    float *activations,
    int activations_len,
    float *prediction,
    int gpu,
    const char *layers[],
    int number_of_layers
) {

    float **__restrict sample = create_matrix(sample_rows, sample_cols);
    for (register int i = 0; i < sample_rows; i++) {
        #pragma omp simd
        for (register int j = 0; j < sample_cols; j++) {
            int index = i * sample_cols + j;
            check_if_index_out_of_bounds(index, get_array_size(sample_input), "sample_input");
            sample[i][j] = sample_input[index];
        }
    }

    float ***__restrict weights = malloc(linear_layer_sizes_rows * sizeof(float**));
    check_if_null((float *)weights, "weights");
    float **__restrict biases = malloc(linear_layer_sizes_rows * sizeof(float*));
    check_if_null((float *)biases, "biases");

    int current_weight_offset = 0;
    int total_bias_count = 0;
    for (int layer_index = 0; layer_index < linear_layer_sizes_rows; layer_index++) {
        const int n_inputs = (int)linear_layer_sizes[layer_index * linear_layer_sizes_cols];
        const int n_neurons = (int)linear_layer_sizes[layer_index * linear_layer_sizes_cols + 1];

        weights[layer_index] = create_matrix(n_inputs, n_neurons);
        for (int i = 0; i < n_inputs; i++) {
            #pragma omp simd
            for (int j = 0; j < n_neurons; j++) {
                int index = current_weight_offset + i * n_neurons + j;
                check_if_index_out_of_bounds(index, get_array_size(weights_input), "weights_input");
                weights[layer_index][i][j] = weights_input[index];
            }
        }

        biases[layer_index] = malloc(n_neurons * sizeof(float));
        check_if_null((float *)biases[layer_index], "biases[layer_index]");

        #pragma omp simd
        for (int i = 0; i < n_neurons; i++) {
            int index = total_bias_count + i;
            check_if_index_out_of_bounds(index, get_array_size(biases_input), "biases_input");
            biases[layer_index][i] = biases_input[index];
        }

        current_weight_offset += n_inputs * n_neurons;
        total_bias_count += n_neurons;
    }

    // Preparing the GPU Kernel

    // Step 1: Get a platform and device
    cl_uint num_platforms;
    cl_platform_id platforms[10];
    clGetPlatformIDs(10, platforms, &num_platforms);
    if (num_platforms == 0 && gpu) {
        gpu = 0;
    }

    cl_device_id devices[10];
    cl_uint num_devices;
    if (gpu) {
        // We use the first suitable device of the GPU type
        clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 10, devices, &num_devices);
    }

    // Step 2: Create context
    cl_int err_code;
    cl_context context;
    if (gpu) {
        context = clCreateContext(NULL, 1, devices, NULL, NULL, &err_code);
        if (context == NULL || err_code != CL_SUCCESS) {
            fprintf(stderr, "Error creating OpenCL context: %d\n", err_code);
            exit(EXIT_FAILURE);
        }
    }

    // Step 3: Create a command queue
    cl_command_queue queue;
    if (gpu) {
        #ifdef __APPLE__
            queue = clCreateCommandQueue(context, devices[0], 0, NULL);
        #else
            queue = clCreateCommandQueueWithProperties(context, devices[0], 0, NULL);
        #endif
    }

    char *source_matmul_gpu;
    cl_program program_matmul_gpu;
    if (gpu) {
        source_matmul_gpu = get_file_content("src/src/matmul_gpu.cl");

        program_matmul_gpu = clCreateProgramWithSource(context, 1, (const char**)&source_matmul_gpu, NULL, NULL);
        clBuildProgram(program_matmul_gpu, 1, devices, "-cl-fast-relaxed-math", NULL, NULL);
    }

    // Forward pass

    float *weights_vec;
    cl_mem weights_vec_buf;
    if (gpu) {
        weights_vec = get_weights_vec(weights, linear_layer_sizes_rows, linear_layer_sizes_cols, linear_layer_sizes);
        weights_vec_buf = get_weights_vec_buf(weights_vec, linear_layer_sizes_rows, linear_layer_sizes_cols, linear_layer_sizes, context);
    }

    // Iterating through neural network layers
    for (int layer_index = 0; layer_index < number_of_layers; layer_index++) {

        // If the layer is Linear, then we call linear_predict_one
        if (strcmp(layers[layer_index], "Linear") == 0) {
            linear_predict_one(
                sample,
                sample_rows,
                sample_cols,
                prediction,
                weights,
                biases,
                linear_layer_sizes,
                linear_layer_sizes_rows,
                linear_layer_sizes_cols,
                activations,
                gpu,
                context,
                queue,
                program_matmul_gpu,
                weights_vec_buf
            );
        }

        // If the layer is Conv2d, then we call conv2d_predict_one
        if (strcmp(layers[layer_index], "Conv2d") == 0) {}
    }

    // Free memory

    if (gpu) {
        clReleaseProgram(program_matmul_gpu);
        clReleaseMemObject(weights_vec_buf);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        free(weights_vec);
    }

    free_matrix(sample);

    for (int layer_index = 0; layer_index < linear_layer_sizes_rows; layer_index++) {
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
    float *linear_layer_sizes,
    int linear_layer_sizes_rows,
    int linear_layer_sizes_cols,
    float *activations,
    int activations_len,
    float *predictions,
    int num_cpu,
    int gpu,
    const char *layers[],
    int number_of_layers
) {

    // Loading a dataset
    float ***__restrict samples = malloc(dataset_samples_rows * sizeof(float**));
    check_if_null((float *)samples, "samples");
    for (int dataset_index = 0; dataset_index < dataset_samples_rows; dataset_index++) {
        samples[dataset_index] = create_matrix(dataset_samples_cols, dataset_samples_depth);
        for (int i = 0; i < dataset_samples_cols; i++) {
            for (int j = 0; j < dataset_samples_depth; j++) {
                int index = dataset_index * dataset_samples_cols * dataset_samples_depth + i * dataset_samples_depth + j;
                check_if_index_out_of_bounds(index, get_array_size(dataset_samples), "dataset_samples");
                samples[dataset_index][i][j] = (float)dataset_samples[index];
            }
        }
    }

    float ***__restrict weights = malloc(linear_layer_sizes_rows * sizeof(float**));
    check_if_null((float *)weights, "weights");
    float **__restrict biases = malloc(linear_layer_sizes_rows * sizeof(float*));
    check_if_null((float *)biases, "biases");

    int current_weight_offset = 0;
    int total_bias_count = 0;
    for (int layer_index = 0; layer_index < linear_layer_sizes_rows; layer_index++) {
        const int n_inputs = (int)linear_layer_sizes[layer_index * linear_layer_sizes_cols];
        const int n_neurons = (int)linear_layer_sizes[layer_index * linear_layer_sizes_cols + 1];

        weights[layer_index] = create_matrix(n_inputs, n_neurons);
        for (int i = 0; i < n_inputs; ++i) {
            #pragma omp simd
            for (int j = 0; j < n_neurons; ++j) {
                int index = current_weight_offset + i * n_neurons + j;
                check_if_index_out_of_bounds(index, get_array_size(weights_input), "weights_input");
                weights[layer_index][i][j] = weights_input[index];
            }
        }

        biases[layer_index] = malloc(n_neurons * sizeof(float));
        check_if_null((float *)biases[layer_index], "biases[layer_index]");

        #pragma omp simd
        for (int i = 0; i < n_neurons; ++i) {
            int index = total_bias_count + i;
            check_if_index_out_of_bounds(index, get_array_size(biases_input), "biases_input");
            biases[layer_index][i] = biases_input[index];
        }

        current_weight_offset += n_inputs * n_neurons;
        total_bias_count += n_neurons;
    }

    if (num_cpu > dataset_samples_rows) {
        num_cpu = dataset_samples_rows;
    }
    
    // Preparing the GPU Kernel

    // Step 1: Get a platform and device
    cl_uint num_platforms;
    cl_platform_id platforms[10];
    clGetPlatformIDs(10, platforms, &num_platforms);
    if (num_platforms == 0 && gpu) {
        gpu = 0;
    }

    cl_device_id devices[10];
    cl_uint num_devices;
    if (gpu) {
        // We use the first suitable device of the GPU type
        clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 10, devices, &num_devices);
    }

    // Step 2: Create context
    cl_int err_code;
    cl_context context;
    if (gpu) {
        context = clCreateContext(NULL, 1, devices, NULL, NULL, &err_code);
        if (context == NULL || err_code != CL_SUCCESS) {
            fprintf(stderr, "Error creating OpenCL context: %d\n", err_code);
            exit(EXIT_FAILURE);
        }
    }

    // Step 3: Create a command queue
    cl_command_queue queue;
    if (gpu) {
        #ifdef __APPLE__
            queue = clCreateCommandQueue(context, devices[0], 0, NULL);
        #else
            queue = clCreateCommandQueueWithProperties(context, devices[0], 0, NULL);
        #endif
    }

    char *source_matmul_gpu;
    cl_program program_matmul_gpu;
    if (gpu) {
        source_matmul_gpu = get_file_content("src/src/matmul_gpu.cl");

        program_matmul_gpu = clCreateProgramWithSource(context, 1, (const char**)&source_matmul_gpu, NULL, NULL);
        clBuildProgram(program_matmul_gpu, 1, devices, "-cl-fast-relaxed-math", NULL, NULL);
    }

    // Forward pass

    float *weights_vec;
    cl_mem weights_vec_buf;
    if (gpu) {
        weights_vec = get_weights_vec(weights, linear_layer_sizes_rows, linear_layer_sizes_cols, linear_layer_sizes);
        weights_vec_buf = get_weights_vec_buf(weights_vec, linear_layer_sizes_rows, linear_layer_sizes_cols, linear_layer_sizes, context);
    }

    // Iterating through neural network layers
    for (int layer_index = 0; layer_index < number_of_layers; layer_index++) {

        // If the layer is Linear, then we call linear_predict
        if (strcmp(layers[layer_index], "Linear") == 0) {
             linear_predict(
                weights,
                biases,
                samples,
                dataset_samples_rows,
                dataset_samples_cols,
                dataset_samples_depth,
                predictions,
                linear_layer_sizes,
                linear_layer_sizes_rows,
                linear_layer_sizes_cols,
                activations,
                num_cpu,
                gpu,
                context,
                queue,
                program_matmul_gpu,
                weights_vec_buf
            );
        }

        // If the layer is Conv2d, then we call conv2d_predict
        if (strcmp(layers[layer_index], "Conv2d") == 0) {}
    }

    // Free memory

    if (gpu) {
        clReleaseProgram(program_matmul_gpu);
        clReleaseMemObject(weights_vec_buf);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        free(weights_vec);
    }

    for (int layer_index = 0; layer_index < linear_layer_sizes_rows; layer_index++) {
        const int n_inputs = (int)linear_layer_sizes[layer_index * linear_layer_sizes_cols];

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
