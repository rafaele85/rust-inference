extern "C" __global__ void rms(
    float* out,
    const float* x,
    const float* gamma,
    const int rows,
    const int cols,
    const float epsilon
) {
      int row = blockIdx.x * blockDim.x + threadIdx.x;

      if (row < rows) {
          float sum_sq = 0.0f;

          for (int i=0; i<cols; i++) {
             float xi = x[row * cols + i];
             sum_sq += xi * xi;
          }

          float rms = sqrtf(sum_sq / cols + epsilon);
          float inv_rms = 1.0f / rms;


          for (int i=0; i<cols; i++) {
             out[row * cols + i] = x[row * cols + i] * inv_rms * gamma[i];
          }
      }
}