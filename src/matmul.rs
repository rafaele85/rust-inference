use std::error::Error;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::driver::CudaContext;
use crate::cuda_kernels::{get_kernels, CudaKernels};
use crate::gpu_tensor::GpuTensor;

pub fn matmul_gpu(a: &GpuTensor, b: &GpuTensor, m: i32, n: i32, k: i32) -> Result<GpuTensor, Box<dyn Error>> {
    let cuda_kernels = get_kernels();
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
        cuda_kernels.blas.gemm(
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





pub fn matmul_gpu_bt(a: &GpuTensor, b: &GpuTensor, m: i32, n: i32, k: i32) -> Result<GpuTensor, Box<dyn Error>> {
    let cuda_kernels = get_kernels();
    let mut c_data = a.stream.alloc_zeros::<f32>(m as usize * n as usize)?;

    let config = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_T,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: n, // row_major -> column_major
        n: m, // row_major -> column_major
        k,
        alpha: 1.0f32,
        lda: k,
        ldb: k,
        ldc: n,
        beta: 0.0f32,
    };

    unsafe {
        cuda_kernels.blas.gemm(
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