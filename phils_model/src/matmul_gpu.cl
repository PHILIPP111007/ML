__kernel void matmul_gpu(__global const float *A, __global const float *B,
                         __global float *C, const int rows_A, const int cols_A,
                         const int cols_B) {

  int row = get_global_id(0);
  int col = get_global_id(1);

  if (row < rows_A && col < cols_B) {
    float sum = 0.0f;

    for (int k = 0; k < cols_A; k++) {
      sum += A[row * cols_A + k] * B[k * cols_B + col];
    }
    C[row * cols_B + col] = sum;
  }
}