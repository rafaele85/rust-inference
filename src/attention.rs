use std::error::Error;
use crate::matmul::matmul;
use crate::scale::scale;
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
    scale(&mut scores, scale_factor);

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
