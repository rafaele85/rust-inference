extern "C" __global__ void mul(
    float* a,
    const float* b,
    const int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        a[i] = a[i] * b[i];
    }
}