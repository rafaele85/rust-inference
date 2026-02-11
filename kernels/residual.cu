extern "C" __global__ void residual_add(
    float* a,
    const float* b,
    const int n
) {
    int i = blockIdx    .x * blockDim.x + threadIdx.x;

    if (i < n) {
        a[i] = a[i] + b[i];
    }
}
