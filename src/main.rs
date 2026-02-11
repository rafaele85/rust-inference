mod matmul;
mod softmax;
mod scale;
mod attention;
mod gpu_tensor;
mod head_ops;
mod cuda_kernels;

use std::error::Error;
use cudarc::driver::CudaContext;
use crate::attention::{attention_gpu, attention_multihead, attention_multihead_gpu};
use crate::cuda_kernels::CudaKernels;
use crate::gpu_tensor::GpuTensor;
use crate::head_ops::{combine_heads, extract_head};
use crate::matmul::{matmul, matmul_gpu, matmul_gpu_bt};
use crate::scale::scale_gpu;
use crate::softmax::{softmax_cudnn_gpu, softmax_cudnn_gpu_2d};

fn main() -> Result<(), Box<dyn Error>> {
    const N: usize = 4096;
    let a_host: Vec<f32> = vec![1.0; N * N];
    let b_host: Vec<f32> = vec![1.0; N * N];

    let expected = N as f32;
    let m = N as i32;
    let n = N as i32;
    let k = N as i32;

    let c_host = matmul(a_host, b_host, m, n, k)?;

    for (i, val) in c_host.iter().enumerate() {
        if (val - expected).abs() > 0.01 {
            println!("Mismatch at {}: expected {} received {}", i, expected, val);
        }
    }

    println!("Result verified");

    let seq_len = 2;
    const dim: usize = 4;
    const num_heads: usize = 2;

    let x = vec![1.0f32; seq_len * dim];
    let w_q = [1.0; dim * dim];
    let w_k = [1.0; dim * dim];
    let w_v = [1.0; dim * dim];
    let w_o = [1.0; dim * dim];
    let output = attention_multihead(&x, &w_q, &w_k, &w_v, &w_o, seq_len, dim, num_heads);

    println!("Result of multi-head attention {:?}", output);

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let cuda_kernels = CudaKernels::new(&ctx)?;


    let a = GpuTensor::from_cpu(stream.clone(), &vec![1.0f32; 6], vec![2, 3])?;
    let b = GpuTensor::from_cpu(stream.clone(), &vec![1.0f32; 6], vec![3, 2])?;
    let c = matmul_gpu(&a, &b, 2, 2, 3, &cuda_kernels)?;
    let result = c.to_cpu()?;
    println!("Result of matmul_gpu {:?}", result);



    let mut x = GpuTensor::from_cpu(stream.clone(), &vec![2.0f32, 4.0, 6.0, 8.0], vec![4])?;
    scale_gpu(&mut x, 0.5, &cuda_kernels)?;
    let result = x.to_cpu()?;
    println!("Result of scale_gpu {:?}", result);

    let mut t = GpuTensor::from_cpu(stream.clone(), &vec![1.0, 2.0, 3.0, 4.0], vec![4])?;
    softmax_cudnn_gpu(&mut t, &cuda_kernels)?;
    let result = t.to_cpu()?;
    println!("Result of softmax_cudnn_gpu {:?}", result);

    let mut scores = GpuTensor::from_cpu(stream.clone(), &vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    softmax_cudnn_gpu_2d(&mut scores, &cuda_kernels)?;
    let result = scores.to_cpu()?;
    println!("Result of softmax_cudnn_gpu_2d {:?}", result);


    let stream3 = ctx.default_stream();
    let q = GpuTensor::from_cpu(stream3.clone(), &vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
    // K: 2x3 matrix (will be transposed to 3x2)
    let k = GpuTensor::from_cpu(stream3.clone(), &vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;

    let scores = matmul_gpu_bt(&q, &k, 2, 2, 3, &cuda_kernels)?;
    println!("matmul_gpu_bt: {:?}", scores.to_cpu()?);

    let seq_len = 2;
    let x = GpuTensor::from_cpu(stream3.clone(), &vec![1.0f32; seq_len * dim], vec![seq_len, dim])?;
    let w_q = GpuTensor::from_cpu(stream3.clone(), &vec![1.0f32; dim * dim], vec![dim, dim])?;
    let w_k = GpuTensor::from_cpu(stream3.clone(), &vec![1.0f32; dim * dim], vec![dim, dim])?;
    let w_v = GpuTensor::from_cpu(stream3.clone(), &vec![1.0f32; dim * dim], vec![dim, dim])?;

    let output = attention_gpu(&x, &w_q, &w_k, &w_v, seq_len, dim, &cuda_kernels)?;
    println!("attention_gpu: {:?}", output.to_cpu()?);

    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let t = GpuTensor::from_cpu(stream3.clone(), &data, vec![2, 4])?;
    let head0 = extract_head(&t, 2, 4, 2, 0, &cuda_kernels)?;
    println!("head0: {:?}", head0.to_cpu());
    let head1 = extract_head(&t, 2, 4, 2, 1, &cuda_kernels)?;
    println!("head1: {:?}", head1.to_cpu());

    let mut combined = GpuTensor::from_cpu(stream3.clone(), &vec![0.0f32; 8], vec![2, 4])?;
    combine_heads(&mut combined, &head0, 2, 4, 2, 0, &cuda_kernels)?;
    combine_heads(&mut combined, &head1, 2, 4, 2, 1, &cuda_kernels)?;

    println!("combined: {:?}", combined.to_cpu());

    let seq_len = 2;

    let x = GpuTensor::from_cpu(stream3.clone(), &vec![1.0f32; seq_len * dim], vec![seq_len, dim])?;
    let w_q = GpuTensor::from_cpu(stream3.clone(), &vec![1.0f32; dim * dim], vec![dim, dim])?;
    let w_k = GpuTensor::from_cpu(stream3.clone(), &vec![1.0f32; dim * dim], vec![dim, dim])?;
    let w_v = GpuTensor::from_cpu(stream3.clone(), &vec![1.0f32; dim * dim], vec![dim, dim])?;
    let w_o = GpuTensor::from_cpu(stream3.clone(), &vec![1.0f32; dim * dim], vec![dim, dim])?;

    let output = attention_multihead_gpu(&x, &w_q, &w_k, &w_v, &w_o, seq_len, dim, num_heads, &cuda_kernels)?;
    println!("Result of multi-head attention {:?}", output.to_cpu()?);

    Ok(())
}