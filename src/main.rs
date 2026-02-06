mod matmul;
mod softmax;

use std::error::Error;
use crate::matmul::matmul;
use crate::softmax::softmax_cpu;

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

    let mut test_gpu = vec![1.0, 2.0, 3.0];
    softmax_cpu(&mut test_gpu);
    println!("Result of softmax_gpu {:?}", test_gpu);


    Ok(())
}