use std::error::Error;
use crate::gelu::gelu;
use crate::gpu_tensor::GpuTensor;
use crate::matmul::matmul_gpu;

pub fn ffn(x: &GpuTensor, w1: &GpuTensor, w2: &GpuTensor, seq_len: usize, dim: usize, hidden_dim: usize) -> Result<GpuTensor, Box<dyn Error>> {
    let mut hidden = matmul_gpu(x, w1, seq_len as i32, hidden_dim as i32, dim as i32)?;
    gelu(&mut hidden)?;
    let output = matmul_gpu(&hidden, w2, seq_len as i32, dim as i32, hidden_dim as i32)?;

    Ok((output))
}