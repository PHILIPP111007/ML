// clang -shared -o functions.so -fPIC -O3 functions.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>


typedef struct {
    double (**A);
    double (**B);
    double (**C);
    int row_A_index;
    int cols_B_index;
    int rows_A;
    int cols_A;
    int rows_B;
    int cols_B;
    int startRow;
    int endRow;
} ThreadData;

void *matmul_thread(void *arg) {
    ThreadData *td = (ThreadData *)arg;

    for (int i = td->startRow; i < td->endRow; i++) {
        for (int j = 0; j < td->cols_B; j++) {
            double sum = 0.0;
            for (int k = 0; k < td->cols_A; k++) {
                sum += td->A[i][k] * td->B[k][j];
            }
            td->C[i][j] = sum;
        }
    }
    return NULL;
}

void matmul(double **A, double **B, double **C, int rows_A, int cols_A, int rows_B, int cols_B) {
    if (cols_A != rows_B) {
        fprintf(stderr, "Cols first != rows second!\n");
        return;
    }

    const int num_threads = 7;
    pthread_t threads[num_threads];
    ThreadData thread_data;

    for (int i = 0; i < rows_A; i++) {
        C[i] = malloc(cols_B * sizeof(double));
    }

    // Разделяем строки между потоками
    int rowsPerThread = rows_A / num_threads;
    int remainder = rows_A % num_threads;

    thread_data.A = A;
    thread_data.B = B;
    thread_data.C = C;
    thread_data.rows_A = rows_A;
    thread_data.cols_A = cols_A;
    thread_data.cols_B = cols_B;

    for (int t = 0; t < num_threads; t++) {
        thread_data.startRow = t * rowsPerThread + ((t < remainder) ? t : remainder); 
        thread_data.endRow = (t + 1) * rowsPerThread + ((t + 1 <= remainder) ? (t + 1) : remainder);
        
        pthread_create(&threads[t], NULL, matmul_thread, &thread_data);
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
}
