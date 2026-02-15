use std::error::Error;
use crate::gpu_tensor::GpuTensor;
use crate::layer_norm::layer_norm;
use crate::matmul::matmul_gpu;

pub fn output_projections(
    x: &GpuTensor,
    gamma: &GpuTensor,
    beta: &GpuTensor,
    w_vocab: &GpuTensor,
    seq_len: usize,
    dim: usize,
    vocab_size: usize
) -> Result<GpuTensor, Box<dyn Error>> {
    let x_norm = layer_norm(&x, gamma, beta, 1e-5)?;

    let logits = matmul_gpu(&x_norm, &w_vocab, seq_len as i32, vocab_size as i32, dim as i32)?;

    Ok(logits)
}


pub fn get_last_logits(logits: &GpuTensor) -> Result<Vec<f32>, Box<dyn Error>> {
    let seq_len = logits.shape[0];
    let vocab_size = logits.shape[1];

    let all_logits = logits.to_cpu()?;
    let start = (seq_len - 1) * vocab_size;
    let last_logits = all_logits[start..start + vocab_size].to_vec();

    Ok(last_logits)
}

pub fn arg_max(arr: &[f32]) -> usize {
    let mut index: usize = 0;
    let mut max_value = arr[index];

    for (i, &val) in arr.iter().enumerate() {
        if val > max_value {
            max_value = val;
            index = i;
        }
    }

    index
}