use std::error::Error;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::driver::{CudaContext};
use cudarc::nvrtc::Ptx;

fn main() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone())?;
    println!("Cuda initialized on device 0");

    let ptx = Ptx::from_file("kernel.ptx");
    let module = ctx.load_module(ptx)?;

    println!("Kernel loaded");

    const N: usize = 4096;
    let a_host: Vec<f32> = vec![1.0; N * N];
    let b_host: Vec<f32> = vec![1.0; N * N];

    let a_dev = stream.memcpy_stod(&a_host)?;
    let b_dev = stream.memcpy_stod(&b_host)?;
    let mut c_dev = stream.alloc_zeros::<f32>(N * N)?;

    println!("Memory allocated");

    let m = N as i32;
    let n = N as i32;
    let k = N as i32;

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

    let expected = N as f32;

    for (i, val) in c_host.iter().enumerate() {
        if (val - expected).abs() > 0.01 {
            println!("Mismatch at {}: expected {} received {}", i, expected, val);
        }
    }

    println!("Result verified");

    Ok(())
}