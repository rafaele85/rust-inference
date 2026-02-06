use std::error::Error;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::driver::CudaContext;

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
        m,
        n,
        k,
        alpha: 1.0f32,
        lda: k,
        ldb: n,
        ldc: n,
        beta: 0.0f32,
    };

    unsafe {
        blas.gemm(
            config,
            &a_dev,
            &b_dev,
            &mut c_dev
        )?;
    }

    stream.synchronize()?;
    println!("Cublas gemm done");

    let c_host: Vec<f32> = stream.memcpy_dtov(&c_dev)?;

    println!("Result {:?}", &c_host[..10]);

    Ok(c_host)
}