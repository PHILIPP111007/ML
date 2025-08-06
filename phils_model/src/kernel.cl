__kernel void matmul(
    __global const float *A,
    __global const float *B,
    __global float *C,
    const int rows_A,
    const int cols_A,
    const int cols_B
) {
    // Global thread index by rows and columns
    int row = get_global_id(0); // Row number of matrix C
    int col = get_global_id(1); // Column number of matrix C

    // Cumulative amount
    float sum = 0.0f;

    // Multiply each element of row A by each element of column B
    for (int k = 0; k < cols_A; k++) {
        sum += A[row * cols_A + k] * B[k * cols_B + col];
    }

    // Save the result in the output matrix
    if ((row < rows_A) && (col < cols_B)) {
        C[row * cols_B + col] = sum;
    }
}
