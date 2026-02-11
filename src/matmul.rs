use std::error::Error;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::driver::CudaContext;
use crate::gpu_tensor::GpuTensor;

pub fn matmul(a_host: Vec<f32>, b_host: Vec<f32>, m: i32, n: i32, k: i32) -> Result<Vec<f32>, Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone())?;
    println!("Cuda initialized on device 0");

    let a_dev = stream.memcpy_stod(&a_host)?;
    let b_dev = stream.memcpy_stod(&b_host)?;
    let mut c_dev = stream.alloc_zeros::<f32>(m as usize * n as usize)?;

    println!("Memory allocated");

    let config = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: n, // row_major -> column_major
        n: m, // row_major -> column_major
        k,
        alpha: 1.0f32,
        lda: n,
        ldb: k,
        ldc: n,
        beta: 0.0f32,
    };

    unsafe {
        blas.gemm(
            config,
            &b_dev, // row_major -> col_major
            &a_dev, // row_major -> col_major
            &mut c_dev
        )?;
    }

    stream.synchronize()?;
    println!("Cublas gemm done");

    let c_host: Vec<f32> = stream.memcpy_dtov(&c_dev)?;

    Ok(c_host)
}



pub fn matmul_gpu(a: &GpuTensor, b: &GpuTensor, m: i32, n: i32, k: i32) -> Result<GpuTensor, Box<dyn Error>> {
    let blas = CudaBlas::new(a.stream.clone())?;

    let mut c_data = a.stream.alloc_zeros::<f32>(m as usize * n as usize)?;

    let config = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: n, // row_major -> column_major
        n: m, // row_major -> column_major
        k,
        alpha: 1.0f32,
        lda: n,
        ldb: k,
        ldc: n,
        beta: 0.0f32,
    };

    unsafe {
        blas.gemm(
            config,
            &b.data, // row_major -> col_major
            &a.data, // row_major -> col_major
            &mut c_data
        )?;
    }

    Ok(GpuTensor {
        data: c_data,
        stream: a.stream.clone(),
        shape: vec![m as usize, n as usize],
    })
}