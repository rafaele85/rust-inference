use std::error::Error;
use crate::attention::attention_multihead_gpu;
use crate::ffn::ffn;
use crate::gpu_tensor::GpuTensor;
use crate::layer_norm::layer_norm;
use crate::residual::residual_add;

pub fn transformer_block(
    x: &GpuTensor,
    gamma1: &GpuTensor,
    beta1: &GpuTensor,
    w_q: &GpuTensor,
    w_k: &GpuTensor,
    w_v: &GpuTensor,
    w_o: &GpuTensor,
    gamma2: &GpuTensor,
    beta2: &GpuTensor,
    w1: &GpuTensor,
    w2: &GpuTensor,
    seq_len: usize,
    dim: usize,
    num_heads: usize,
    hidden_dim: usize
) -> Result<GpuTensor, Box<dyn Error>> {

    let x_norm = layer_norm(x, gamma1, beta1, 1e-5)?;
    println!("after layernorm1: {:?}", x_norm.to_cpu()?);
    let attn_out = attention_multihead_gpu(&x_norm, w_q, w_k, w_v, w_o, seq_len, dim, num_heads)?;
    let mut x = x.copy()?;
    residual_add(&mut x, &attn_out)?;

    let x_norm = layer_norm(&x, gamma2, beta2, 1e-5)?;
    let ffn_out = ffn(&x_norm, w1, w2, seq_len, dim, hidden_dim)?;
    residual_add(&mut x, &ffn_out)?;

    Ok(x)
}