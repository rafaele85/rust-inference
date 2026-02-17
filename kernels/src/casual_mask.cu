extern "C" __global__ void casual_mask(
    float* scores,
    const int seq_len
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int total = seq_len * seq_len;

    if (i < total) {
        int row = i / seq_len;
        int col = i % seq_len;

        if (col > row) {
            scores[i] = -INFINITY;
        }
    }
}