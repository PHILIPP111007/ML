__kernel void adam_step_gpu(__global float *weights, __global float *grads,
                            __global float *m, __global float *v,
                            const float b1, const float b2, const float lr,
                            const float eps, const float max_change,
                            const int total_elements_per_sample,
                            const int dataset_samples_rows,
                            __global int *epoch) {

  int tid = get_global_id(0);

  int total_elements = total_elements_per_sample * dataset_samples_rows;

  if (tid >= total_elements) {
    return;
  }

  atomic_inc(epoch);

  barrier(CLK_GLOBAL_MEM_FENCE);

  float b1_pow = powf(b1, *epoch);
  float b2_pow = powf(b2, *epoch);
  float inv_1mb1 = 1.0f / (1.0f - b1_pow + 1e-10f);
  float inv_1mb2 = 1.0f / (1.0f - b2_pow + 1e-10f);
  float b1_minus_1 = 1.0f - b1;
  float b2_minus_1 = 1.0f - b2;

  int idx = tid % total_elements_per_sample;
  int sample_idx = tid / total_elements_per_sample;

  float grad = grads[idx + sample_idx * total_elements_per_sample];
  float grad_sq = grad * grad;

  m[idx] = b1 * m[idx] + b1_minus_1 * grad;
  v[idx] = b2 * v[idx] + b2_minus_1 * grad_sq;

  float m_hat = m[idx] * inv_1mb1;
  float v_hat = v[idx] * inv_1mb2;

  float delta = lr * m_hat / (sqrt(v_hat) + eps);

  delta = fmin(fmax(delta, -max_change), max_change);

  weights[idx] -= delta;
}
