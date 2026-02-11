mod matmul;
mod softmax;
mod scale;
mod attention;

use std::error::Error;
use crate::attention::{attention, attention_multihead};
use crate::matmul::matmul;
use crate::scale::scale_gpu;
use crate::softmax::{softmax_cudnn};

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
    softmax_cudnn(&mut test_gpu);
    println!("Result of softmax_cudnn {:?}", test_gpu);

    let mut test_scale = vec![1.0, 2.0, 3.0, 4.0];
    scale_gpu(&mut test_scale, 0.5);
    println!("Result of scale_gpu {:?}", test_scale);

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

    Ok(())
}