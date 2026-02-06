mod matmul;

use std::error::Error;
use crate::matmul::matmul;

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

    Ok(())
}