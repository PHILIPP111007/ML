// gcc -shared -o functions.so -fPIC -O3 functions.c

void forward(double *output_list, double *weights, int rows, int cols, double *inputs) {
    for (int i = 0; i < rows; i++) {
        double output = 0;
        for (int j = 0; j < cols; j++) {
            output += weights[i * cols + j] * inputs[j];
        }
        output_list[i] = output;
    }
}


double clip(double value, double min, double max) {
    if (value > max) {
        return max;
    } else if (value < min) {
        return min;
    }
    return value;
}


void backward(double *weights, int weights_rows, int weights_cols, double *delta, double *new_deltas) {
    for (int i = 0; i < weights_rows; i++) {
        for (int j = 0; j < weights_cols; j++) {
            double new_delta = clip(weights[i * weights_cols + j] * delta[i], -0.1, 0.1);
            new_deltas[i + j] = new_delta;
        }
    }
}

