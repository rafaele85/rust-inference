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