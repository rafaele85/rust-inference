use std::error::Error;
use crate::cuda_kernels::CudaKernels;
use crate::gpu_tensor::GpuTensor;
use crate::head_ops::{combine_heads, extract_head};
use crate::matmul::{matmul, matmul_gpu, matmul_gpu_bt};
use crate::scale::{scale_gpu};
use crate::softmax::{softmax_cpu, softmax_cudnn_gpu_2d};


pub fn attention_multihead (
    x: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    w_o: &[f32],
    seq_len: usize,
    dim: usize,
    num_heads: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {

    let head_dim = dim / num_heads;

    let q = matmul(
        x.to_vec(),
        w_q.to_vec(),
        seq_len as i32,
        dim as i32,
        dim as i32,
    )?;

    println!("q computed");

    let k = matmul(
        x.to_vec(),
        w_k.to_vec(),
        seq_len as i32,
        dim as i32,
        dim as i32,
    )?;

    println!("k computed");

    let v = matmul(
        x.to_vec(),
        w_v.to_vec(),
        seq_len as i32,
        dim as i32,
        dim as i32,
    )?;

    println!("v computed");
    println!("Q: {:?}", &q);
    println!("K: {:?}", &k);
    println!("V: {:?}", &v);

    let mut head_outputs = vec![0.0f32; seq_len * dim];

    for h in 0..num_heads {
        let mut q_head = vec![0.0f32; seq_len * head_dim];
        let mut k_head = vec![0.0f32; seq_len * head_dim];
        let mut v_head = vec![0.0f32; seq_len * head_dim];

        for s in 0..seq_len {
            for d in 0..head_dim {
                println!("s={}, d={}, h={}, head_dim={}, dim={}, q.len()={}", s, d, h, head_dim, dim, q.len());

                q_head[s * head_dim + d] = q[s * dim + h * head_dim + d];
                k_head[s * head_dim + d] = k[s * dim + h * head_dim + d];
                v_head[s * head_dim + d] = v[s * dim + h * head_dim + d];
            }
        }

        let mut k_head_t = vec![0.0f32; head_dim * seq_len];
        for i in 0..seq_len {
            for j in 0..head_dim {
                k_head_t[j * seq_len + i] = k_head[i * head_dim + j];
            }
        }

        let mut scores = matmul(
            q_head,
            k_head_t,
            seq_len as i32,
            seq_len as i32,
            head_dim as i32
        )?;

        for i in 0..seq_len {
            let start = i * seq_len;
            let end = start + seq_len;
            let mut row = scores[start..end].to_vec();
            softmax_cpu(&mut row);
            scores[start..end].copy_from_slice(&row);
        }

        let head_out = matmul(
            scores,
            v_head,
            seq_len as i32,
            head_dim as i32,
            seq_len as i32,
        )?;

        for s in 0..seq_len {
            for d in 0..head_dim {
                head_outputs[s * dim + h * head_dim + d] = head_out[s * head_dim + d];
            }
        }
    }

    let output = matmul(
        head_outputs,
        w_o.to_vec(),
        seq_len as i32,
        dim as i32,
        dim as i32
    )?;

    Ok(output.to_vec())
}

pub fn attention_gpu(
    x: &GpuTensor,
    w_q: &GpuTensor,
    w_k: &GpuTensor,
    w_v: &GpuTensor,
    seq_len: usize,
    dim: usize,
    cuda_kernels: &CudaKernels
) -> Result<GpuTensor, Box<dyn Error>> {
    let q = matmul_gpu(
        x,
        w_q,
        seq_len as i32,
        dim as i32,
        dim as i32,
        &cuda_kernels
    )?;

    let k = matmul_gpu(
        x,
        w_k,
        seq_len as i32,
        dim as i32,
        dim as i32,
        &cuda_kernels
    )?;

    let v = matmul_gpu(
        x,
        w_v,
        seq_len as i32,
        dim as i32,
        dim as i32,
        &cuda_kernels
    )?;


    let scores = matmul_gpu_bt(
        &q,
        &k,
        seq_len as i32,
        seq_len as i32,
        dim as i32,
        &cuda_kernels
    )?;

    let scale_factor = 1.0 / (dim as f32).sqrt();
    let mut scores = scores;
    scale_gpu(&mut scores, scale_factor, &cuda_kernels)?;

    softmax_cudnn_gpu_2d(&mut scores, &cuda_kernels)?;

    let output = matmul_gpu(
        &scores,
        &v,
        seq_len as i32,
        dim as i32,
        seq_len as i32,
        &cuda_kernels
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
    num_heads: usize,
    cuda_kernels: &CudaKernels
) -> Result<GpuTensor, Box<dyn Error>> {

    let head_dim = dim / num_heads;

    let q = matmul_gpu(
        x,
        w_q,
        seq_len as i32,
        dim as i32,
        dim as i32,
        &cuda_kernels
    )?;

    let k = matmul_gpu(
        x,
        w_k,
        seq_len as i32,
        dim as i32,
        dim as i32,
        &cuda_kernels
    )?;

    let v = matmul_gpu(
        x,
        w_v,
        seq_len as i32,
        dim as i32,
        dim as i32,
        &cuda_kernels
    )?;

    let mut head_outputs = GpuTensor::alloc_zeros(x.stream.clone(), seq_len * dim, vec![seq_len, dim])?;

    for h in 0..num_heads {
        let q_head = extract_head(&q, seq_len, dim, head_dim, h, &cuda_kernels)?;
        let k_head = extract_head(&k, seq_len, dim, head_dim, h, &cuda_kernels)?;
        let v_head = extract_head(&v, seq_len, dim, head_dim, h, &cuda_kernels)?;

        let mut scores = matmul_gpu_bt(
            &q_head,
            &k_head,
            seq_len as i32,
            seq_len as i32,
            head_dim as i32,
            &cuda_kernels
        )?;

        let scale_factor = 1.0 / (head_dim as f32).sqrt();
        scale_gpu(&mut scores, scale_factor, &cuda_kernels)?;
        softmax_cudnn_gpu_2d(&mut scores, &cuda_kernels)?;


        let head_out = matmul_gpu(
            &scores,
            &v_head,
            seq_len as i32,
            head_dim as i32,
            seq_len as i32,
            &cuda_kernels
        )?;

        combine_heads(&mut head_outputs, &head_out, seq_len, dim, head_dim, h, &cuda_kernels)?;
    }

    let output = matmul_gpu(
        &head_outputs,
        &w_o,
        seq_len as i32,
        dim as i32,
        dim as i32,
        &cuda_kernels
    )?;

    Ok(output)
}
