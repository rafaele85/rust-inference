extern "C" __global__ void scale(float* x, float factor, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    x[i] *= factor;
  }
}
