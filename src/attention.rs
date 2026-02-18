use std::error::Error;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::cuda_kernels::{get_kernels};
use crate::gpu_tensor::GpuTensor;
use crate::head_ops::{combine_heads, extract_head};
use crate::matmul::{matmul_gpu, matmul_gpu_bt};
use crate::rope::rope;
use crate::scale::{scale_gpu};
use crate::softmax::{softmax_cudnn_gpu_2d};


pub fn attention_gpu(
    x: &GpuTensor,
    w_q: &GpuTensor,
    w_k: &GpuTensor,
    w_v: &GpuTensor,
    seq_len: usize,
    dim: usize
) -> Result<GpuTensor, Box<dyn Error>> {
    let cuda_kernels = get_kernels();
    let q = matmul_gpu(
        x,
        w_q,
        seq_len as i32,
        dim as i32,
        dim as i32
    )?;

    let k = matmul_gpu(
        x,
        w_k,
        seq_len as i32,
        dim as i32,
        dim as i32
    )?;

    let v = matmul_gpu(
        x,
        w_v,
        seq_len as i32,
        dim as i32,
        dim as i32
    )?;


    let scores = matmul_gpu_bt(
        &q,
        &k,
        seq_len as i32,
        seq_len as i32,
        dim as i32
    )?;

    let scale_factor = 1.0 / (dim as f32).sqrt();
    let mut scores = scores;
    scale_gpu(&mut scores, scale_factor)?;

    softmax_cudnn_gpu_2d(&mut scores)?;

    let output = matmul_gpu(
        &scores,
        &v,
        seq_len as i32,
        dim as i32,
        seq_len as i32
    )?;

    Ok(output)
}



pub fn attention_multihead_gpu (
    x: &GpuTensor,
    w_q: &GpuTensor,
    w_k: &GpuTensor,
    w_v: &GpuTensor,
    w_o: &GpuTensor,
    seq_len: usize,
    dim: usize,
    num_heads: usize
) -> Result<GpuTensor, Box<dyn Error>> {

    let cuda_kernels = get_kernels();
    let head_dim = dim / num_heads;

    let q = matmul_gpu(
        x,
        w_q,
        seq_len as i32,
        dim as i32,
        dim as i32
    )?;

    let mut k = matmul_gpu(
        x,
        w_k,
        seq_len as i32,
        dim as i32,
        dim as i32
    )?;

    let mut v = matmul_gpu(
        x,
        w_v,
        seq_len as i32,
        dim as i32,
        dim as i32
    )?;

    let mut head_outputs = GpuTensor::alloc_zeros(x.stream.clone(), seq_len * dim, vec![seq_len, dim])?;

    for h in 0..num_heads {
        let mut q_head = extract_head(&q, seq_len, dim, head_dim, h)?;
        let mut k_head = extract_head(&k, seq_len, dim, head_dim, h)?;
        let v_head = extract_head(&v, seq_len, dim, head_dim, h)?;

        rope(&mut q_head);
        rope(&mut k_head);


        let mut scores = matmul_gpu_bt(
            &q_head,
            &k_head,
            seq_len as i32,
            seq_len as i32,
            head_dim as i32
        )?;

        let scale_factor = 1.0 / (head_dim as f32).sqrt();
        scale_gpu(&mut scores, scale_factor)?;
        softmax_cudnn_gpu_2d(&mut scores)?;


        let head_out = matmul_gpu(
            &scores,
            &v_head,
            seq_len as i32,
            head_dim as i32,
            seq_len as i32
        )?;

        combine_heads(&mut head_outputs, &head_out, seq_len, dim, head_dim, h)?;
    }

    let output = matmul_gpu(
        &head_outputs,
        &w_o,
        seq_len as i32,
        dim as i32,
        dim as i32
    )?;

    Ok(output)
}




pub fn attention_gqa (
    x: &GpuTensor,
    w_q: &GpuTensor,
    w_k: &GpuTensor,
    w_v: &GpuTensor,
    w_o: &GpuTensor,
    seq_len: usize,
    dim: usize,
    num_heads: usize,
    num_kv_heads: usize
) -> Result<GpuTensor, Box<dyn Error>> {

    let cuda_kernels = get_kernels();

    let head_dim = dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let num_kv_groups = num_heads / num_kv_heads;

    let q = matmul_gpu(
        x,
        w_q,
        seq_len as i32,
        dim as i32,
        dim as i32
    )?;

    let k = matmul_gpu(
        x,
        w_k,
        seq_len as i32,
        kv_dim as i32,
        dim as i32
    )?;

    let v = matmul_gpu(
        x,
        w_v,
        seq_len as i32,
        kv_dim as i32,
        dim as i32
    )?;

    let mut head_outputs = GpuTensor::alloc_zeros(x.stream.clone(), seq_len * dim, vec![seq_len, dim])?;

    for h in 0..num_heads {
        let mut q_head = extract_head(&q, seq_len, dim, head_dim, h)?;
        let kv_head_idx = h / num_kv_groups;
        let mut k_head = extract_head(&k, seq_len, kv_dim, head_dim, kv_head_idx)?;
        let v_head = extract_head(&v, seq_len, kv_dim, head_dim, kv_head_idx)?;

        rope(&mut q_head)?;
        rope(&mut k_head)?;


        let mut scores = matmul_gpu_bt(
            &q_head,
            &k_head,
            seq_len as i32,
            seq_len as i32,
            head_dim as i32
        )?;

        let scale_factor = 1.0 / (head_dim as f32).sqrt();
        scale_gpu(&mut scores, scale_factor)?;

        const BLOCK_SIZE: usize = 256;
        let grid_size = (seq_len * seq_len + BLOCK_SIZE -1 ) / BLOCK_SIZE;

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0
        };

        let mut builder = x.stream.launch_builder(&cuda_kernels.casual_mask);
        builder.arg(&mut scores.data);
        let i32_seq_len = seq_len as i32;
        builder.arg(&i32_seq_len);

        unsafe { builder.launch(config) }?;


        softmax_cudnn_gpu_2d(&mut scores)?;


        let head_out = matmul_gpu(
            &scores,
            &v_head,
            seq_len as i32,
            head_dim as i32,
            seq_len as i32
        )?;

        combine_heads(&mut head_outputs, &head_out, seq_len, dim, head_dim, h)?;
    }

    let output = matmul_gpu(
        &head_outputs,
        &w_o,
        seq_len as i32,
        dim as i32,
        dim as i32
    )?;

    Ok(output)
}
