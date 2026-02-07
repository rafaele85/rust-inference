use std::error::Error;
use crate::matmul::matmul;
use crate::scale::scale_gpu;
use crate::softmax::softmax_cpu;

pub fn attention(
    x: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    seq_len: usize,
    dim: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
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

    let mut k_t = vec![0.0; dim * seq_len];
    for i in 0..seq_len {
        for j in 0..dim {
            k_t[j * seq_len + i] = k[i * dim + j];
        }
    }

    println!("transpose done. q len: {}, k_t len: {}", q.len(), k_t.len());

    let scores = matmul(
        q,
        k_t,
        seq_len as i32,
        seq_len as i32,
        dim as i32
    )?;

    let scale_factor = 1.0 / (dim as f32).sqrt();
    let mut scores = scores;
    scale_gpu(&mut scores, scale_factor);

    for i in 0..seq_len {
        let start = i * seq_len;
        let end = start + seq_len;
        let mut row = scores[start..end].to_vec();
        softmax_cpu(&mut row);
        scores[start..end].copy_from_slice(&row);
    }

    let output = matmul(
        scores,
        v,
        seq_len as i32,
        dim as i32,
        seq_len as i32,
    )?;

    Ok(output)
}