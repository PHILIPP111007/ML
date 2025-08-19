__kernel void adam_step_gpu(__global float *weights, __global float *grads,
                            __global float *m, __global float *v,
                            const float b1, const float b2, const float lr,
                            const float eps, const float max_change,
                            const int total_elements_per_sample,
                            const int dataset_samples_rows,
                            const int epoch) {

  int tid = get_global_id(0);

  int total_elements = total_elements_per_sample * dataset_samples_rows;

  if (tid >= total_elements) {
    return;
  }

  int new_epoch = tid + epoch;

  double b1_pow = pow(b1, new_epoch);
  double b2_pow = pow(b2, new_epoch);
  double inv_1mb1 = 1.0f / (1.0f - b1_pow + 1e-6f);
  double inv_1mb2 = 1.0f / (1.0f - b2_pow + 1e-6f);
  double b1_minus_1 = 1.0f - b1;
  double b2_minus_1 = 1.0f - b2;

  int idx = tid % total_elements_per_sample;
  int sample_idx = tid / total_elements_per_sample;

  double grad = grads[idx + sample_idx * total_elements_per_sample];
  double grad_sq = grad * grad;

  m[idx] = b1 * m[idx] + b1_minus_1 * grad;
  v[idx] = b2 * v[idx] + b2_minus_1 * grad_sq;

  double m_hat = m[idx] * inv_1mb1;
  double v_hat = v[idx] * inv_1mb2;

  float delta = lr * m_hat / (sqrt(v_hat) + eps);

  delta = fmin(fmax(delta, -max_change), max_change);

  weights[idx] -= delta;
}
