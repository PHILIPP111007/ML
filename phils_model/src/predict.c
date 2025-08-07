#include <stdlib.h>
#include <stdio.h>
#include "functions.h"
#include "forward.h"
#include "predict.h"
#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif


void *predict_thread(void *arg) {
    ThreadRange *range = (ThreadRange*)arg;



    // Шаг 1: получение платформы и устройства
    cl_uint numPlatforms;
    cl_platform_id platforms[10];
    clGetPlatformIDs(10, platforms, &numPlatforms);

    // Используем первое подходящее устройство типа GPU
    cl_device_id devices[10];
    cl_uint numDevices;
    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 10, devices, &numDevices);

    // Шаг 2: создание контекста
    cl_context context = clCreateContext(NULL, 1, devices, NULL, NULL, NULL);

    // Шаг 3: создание очереди команд
    cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, NULL);

    // Шаг 3: чтение и компиляция ядра OpenCL
    FILE* fp = fopen("src/kernel.cl", "rb");
    fseek(fp, 0, SEEK_END);
    long fileSize = ftell(fp);
    rewind(fp);
    char* sourceStr = (char*)malloc(fileSize + 1);
    fread(sourceStr, 1, fileSize, fp);
    fclose(fp);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&sourceStr, NULL, NULL);
    clBuildProgram(program, 1, devices, "-cl-fast-relaxed-math", NULL, NULL);






    for (int i = range->start; i < range->end; i++) {
        PredictTask *task = &range->tasks[i];

        float ***__restrict Y = malloc(task->layer_sizes_rows * sizeof(float**));

        // Forward pass
        forward(
            task->sample,
            task->dataset_samples_cols,
            task->dataset_samples_depth,
            task->weights,
            task->biases,
            Y,
            task->layer_sizes,
            task->layer_sizes_rows,
            task->layer_sizes_cols,
            task->activations,
            task->gpu,
            context,
            queue,
            program
        );

        // Get predictions from last layer
        for (int j = 0; j < task->n_neurons_last_layer; j++) {
            int index = task->dataset_index * task->n_neurons_last_layer + j;
            task->predictions[index] = Y[task->layer_sizes_rows - 1][0][j];
        }

        // Free memory

        for (register int layer_index = 0; layer_index < task->layer_sizes_rows; layer_index++) {
            for (register int i = 0; i < task->dataset_samples_cols; i++) {
                free(Y[layer_index][i]);
            }
            free(Y[layer_index]);
        }
        free(Y);
    }

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseProgram(program);

    free(range);

    return NULL;
}
