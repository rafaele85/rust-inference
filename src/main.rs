mod matmul;
mod softmax;
mod scale;
mod attention;
mod gpu_tensor;
mod head_ops;
mod cuda_kernels;
mod residual;
mod gelu;
mod ffn;
mod layer_norm;
mod transformer_block;
mod output_projections;
mod embeddings;
mod rope;
mod rms_norm;
mod silu;
mod mul;
mod gguf;

mod transpose;

use std::error::Error;
use std::fs::File;
use cudarc::driver::CudaContext;
use crate::attention::{attention_gpu, attention_gqa};
use crate::cuda_kernels::CudaKernels;
use crate::embeddings::embeddings;
use crate::ffn::{ffn, ffn_swiglu};
use crate::gelu::gelu;
use crate::gguf::gguf::read_gguf_header;
use crate::gguf::tensor_data::{read_tensor_f32, read_tensor_q4_0};
use crate::gpu_tensor::GpuTensor;
use crate::head_ops::{combine_heads, extract_head};
use crate::layer_norm::layer_norm;
use crate::matmul::{matmul_gpu, matmul_gpu_bt};
use crate::output_projections::{arg_max, get_last_logits, output_projections};
use crate::residual::residual_add;
use crate::rms_norm::rms_norm;
use crate::rope::rope;
use crate::scale::scale_gpu;
use crate::silu::silu;
use crate::softmax::{softmax_cudnn_gpu, softmax_cudnn_gpu_2d};
use crate::transformer_block::{transformer_block_llama};
use crate::transpose::{transpose_2d};

fn load_f32_bin(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    let raw: Vec<f32> = bytes.chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    println!("path {} wq len: {}", path, raw.len());
    raw
}

// Add to main.rs or a test module
fn test_rope_kernel() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // Test 1: small
    let input = load_f32_bin("python/test_data/rope_input_small.bin");
    let expected = load_f32_bin("python/test_data/rope_expected_small.bin");
    let seq_len = 2;
    let dim = 8;

    let mut x = GpuTensor::from_cpu(stream.clone(), &input, vec![seq_len, dim])?;
    rope(&mut x)?;
    let result = x.to_cpu()?;

    println!("RoPE small test:");
    println!("  Expected: {:?}", &expected[..8]);
    println!("  Got:      {:?}", &result[..8]);
    let max_diff: f32 = result.iter().zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("  Max diff: {}", max_diff);
    assert!(max_diff < 1e-5, "Small RoPE test FAILED: max_diff={}", max_diff);
    println!("  PASSED");

    // Test 2: realistic
    let input = load_f32_bin("python/test_data/rope_input_real.bin");
    let expected = load_f32_bin("python/test_data/rope_expected_real.bin");
    let seq_len = 4;
    let dim = 64;

    let mut x = GpuTensor::from_cpu(stream.clone(), &input, vec![seq_len, dim])?;
    rope(&mut x)?;
    let result = x.to_cpu()?;

    println!("\nRoPE realistic test:");
    println!("  Expected pos1 first 8: {:?}", &expected[dim..dim+8]);
    println!("  Got      pos1 first 8: {:?}", &result[dim..dim+8]);
    let max_diff: f32 = result.iter().zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("  Max diff: {}", max_diff);
    assert!(max_diff < 1e-5, "Realistic RoPE test FAILED: max_diff={}", max_diff);
    println!("  PASSED");

    Ok(())
}

fn test_layer0() -> Result<(), Box<dyn Error>> {

    println!("===========Load blk0=================");
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let seq_len = 4;
    let dim = 2048;
    let num_heads = 32;
    let num_kv_heads = 4;

    let head_dim = dim / num_heads;  // 64
    let kv_dim = num_kv_heads * head_dim;  // 4 * 64 = 256

    let hidden_dim = 5632;




    let (tensor_metadata_map, data_start_offset) = read_gguf_header("tinyllama-1.1b-chat-v1.0.Q4_0.gguf")?;

    let tensor_q = tensor_metadata_map.get("blk.0.attn_q.weight").unwrap();
    let tensor_k = tensor_metadata_map.get("blk.0.attn_k.weight").unwrap();
    let tensor_v = tensor_metadata_map.get("blk.0.attn_v.weight").unwrap();
    let tensor_o = tensor_metadata_map.get("blk.0.attn_output.weight").unwrap();


    let mut file = File::open("tinyllama-1.1b-chat-v1.0.Q4_0.gguf")?;

//    let w_q_data = &read_tensor_q4_0(&mut file, tensor_q, data_start_offset)?;        // [2048, 2048]

    let w_q_raw = read_tensor_q4_0(&mut file, tensor_q, data_start_offset)?;
    let w_q_data = transpose_2d(&w_q_raw, 2048, 2048);
    println!("w_q[0]: {}", w_q_raw[0]);      // row 0, block 0, elem 0
    println!("w_q[32]: {}", w_q_raw[32]);    // row 0, block 1, elem 0
    println!("w_q[64]: {}", w_q_raw[64]);    // row 0, block 2, elem 0
    println!("w_q[2048]: {}", w_q_raw[2048]); // row 1, block 0, elem 0


    let w_k_data = transpose_2d(&read_tensor_q4_0(&mut file, tensor_k, data_start_offset)?, dim, kv_dim);     // [2048, 256]
    let w_v_data = transpose_2d(&read_tensor_q4_0(&mut file, tensor_v, data_start_offset)?, dim, kv_dim);     // [2048, 256]
    let w_o_data = transpose_2d(&read_tensor_q4_0(&mut file, tensor_o, data_start_offset)?, dim, dim);        // [2048, 2048]

    println!("w_q: {}", w_q_data.len());  // Should be 2048*2048 = 4194304
    println!("w_k: {}", w_k_data.len());  // Should be 2048*256 = 524288
    println!("w_v: {}", w_v_data.len());  // Should be 2048*256 = 524288
    println!("w_o: {}", w_o_data.len());  // Should be 2048*2048 = 4194304

    let w_q = GpuTensor::from_cpu(stream.clone(), &w_q_data, vec![dim, dim])?;
    let w_k = GpuTensor::from_cpu(stream.clone(), &w_k_data, vec![dim, kv_dim])?;
    let w_v = GpuTensor::from_cpu(stream.clone(), &w_v_data, vec![dim, kv_dim])?;
    let w_o = GpuTensor::from_cpu(stream.clone(), &w_o_data, vec![dim, dim])?;


    let tensor_attn_norm = tensor_metadata_map.get("blk.0.attn_norm.weight").unwrap();
    let tensor_ffn_norm = tensor_metadata_map.get("blk.0.ffn_norm.weight").unwrap();
    let tensor_ffn_gate = tensor_metadata_map.get("blk.0.ffn_gate.weight").unwrap();
    let tensor_ffn_up = tensor_metadata_map.get("blk.0.ffn_up.weight").unwrap();
    let tensor_ffn_down = tensor_metadata_map.get("blk.0.ffn_down.weight").unwrap();

    let attn_norm_data = read_tensor_f32(&mut file, tensor_attn_norm, data_start_offset)?;
    let ffn_norm_data = read_tensor_f32(&mut file, tensor_ffn_norm, data_start_offset)?;

    let ffn_gate_data = transpose_2d(&read_tensor_q4_0(&mut file, tensor_ffn_gate, data_start_offset)?, dim, hidden_dim);
    let ffn_up_data = transpose_2d(&read_tensor_q4_0(&mut file, tensor_ffn_up, data_start_offset)?, dim, hidden_dim);
    let ffn_down_data = transpose_2d(&read_tensor_q4_0(&mut file, tensor_ffn_down, data_start_offset)?, hidden_dim, dim);

    println!("attn_norm: {}", attn_norm_data.len());  // 2048
    println!("ffn_norm: {}", ffn_norm_data.len());    // 2048
    println!("ffn_gate: {}", ffn_gate_data.len());    // 2048*5632
    println!("ffn_up: {}", ffn_up_data.len());        // 2048*5632
    println!("ffn_down: {}", ffn_down_data.len());    // 5632*2048

    let x_data: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32 * 0.001).sin()).collect();
    let x = GpuTensor::from_cpu(stream.clone(), &x_data, vec![seq_len, dim])?;
    let output = attention_gqa(&x, &w_q, &w_k, &w_v, &w_o, seq_len, dim, num_heads, num_kv_heads)?;
    let result = output.to_cpu()?;

    println!("layer0 output shape {:?}", &output.shape);
    println!("layer0 first 8 {:?}", &result[0..8]);
    println!("layer0 attn_norm: {}", attn_norm_data.len());  // 2048
    println!("layer0 ffn_norm: {}", ffn_norm_data.len());    // 2048
    println!("layer0 ffn_gate: {}", ffn_gate_data.len());    // 2048*5632
    println!("layer0 ffn_up: {}", ffn_up_data.len());        // 2048*5632
    println!("layer0 ffn_down: {}", ffn_down_data.len());    // 5632*2048

    let attn_norm = GpuTensor::from_cpu(stream.clone(), &attn_norm_data, vec![dim])?;
    let ffn_norm = GpuTensor::from_cpu(stream.clone(), &ffn_norm_data, vec![dim])?;
    let ffn_gate = GpuTensor::from_cpu(stream.clone(), &ffn_gate_data, vec![dim, hidden_dim])?;
    let ffn_up = GpuTensor::from_cpu(stream.clone(), &ffn_up_data, vec![dim, hidden_dim])?;
    let ffn_down = GpuTensor::from_cpu(stream.clone(), &ffn_down_data, vec![hidden_dim, dim])?;

    let x_norm = rms_norm(&x, &attn_norm)?;
    println!("After RMSNorm, first 8: {:?}", &x_norm.to_cpu()?[..8]);
    let q = matmul_gpu(&x_norm, &w_q, seq_len as i32, dim as i32, dim as i32)?;
    println!("Q proj output (no transpose), first 8: {:?}", &q.to_cpu()?[..8]);
    let attn_out = attention_gqa(&x_norm, &w_q, &w_k, &w_v, &w_o, seq_len, dim, num_heads, num_kv_heads)?;
    println!("Attention output, first 8: {:?}", &attn_out.to_cpu()?[..8]);
    let mut output = x.copy()?;
    residual_add(&mut output, &attn_out)?;

    println!("After attention + residual, first 8: {:?}", &output.to_cpu()?[..8]);
    let output = transformer_block_llama(
        &x,
        &attn_norm, &ffn_norm,
        &w_q, &w_k, &w_v, &w_o,
        &ffn_gate, &ffn_up, &ffn_down,
        seq_len, dim, num_heads, num_kv_heads, hidden_dim
    )?;

    println!("Block 0 output shape: {:?}", output.shape);
    println!("Block 0 first 8: {:?}", &output.to_cpu()?[..8]);

    println!("GGUF first 8: {:?}", &w_q_data[..8]);
    println!("GGUF stride 2048: {:?}", (0..8).map(|i| w_q_data[i * 2048]).collect::<Vec<_>>());


    Ok(())
}

fn test_rope() -> Result<(), Box<dyn Error>> {
    let seq_len = 4;
    let dim = 2048;
    let num_heads = 32;
    let num_kv_heads = 4;
    let kv_dim = num_kv_heads * 64;

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();


    let wq_data = transpose_2d(&load_f32_bin("python/attention_wq.bin"), dim, dim);
    let wk_data = transpose_2d(&load_f32_bin("python/attention_wk.bin"), kv_dim, dim);
    let wv_data = transpose_2d(&load_f32_bin("python/attention_wv.bin"), kv_dim, dim);
    let wo_data = transpose_2d(&load_f32_bin("python/attention_wo.bin"), dim, dim);
    let expected = load_f32_bin("python/attention_output.bin");
    let x_data = load_f32_bin("python/attention_x.bin");

    let x = GpuTensor::from_cpu(stream.clone(), &x_data, vec![seq_len, dim])?;
    let w_q = GpuTensor::from_cpu(stream.clone(), &wq_data, vec![dim, dim])?;
    let w_k = GpuTensor::from_cpu(stream.clone(), &wk_data, vec![dim, kv_dim])?;
    let w_v = GpuTensor::from_cpu(stream.clone(), &wv_data, vec![dim, kv_dim])?;
    let w_o = GpuTensor::from_cpu(stream.clone(), &wo_data, vec![dim, dim])?;

    let output = attention_gqa(&x, &w_q, &w_k, &w_v, &w_o, seq_len, dim, num_heads, num_kv_heads)?;
    let result = output.to_cpu()?;

    println!("Expected first 8: {:?}", &expected[0..8]);
    println!("Got first 8:      {:?}", &result[0..8]);

    let mut max_diff = 0.0f32;
    for i in 0..result.len() {
        let diff = (result[i] - expected[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    println!("Max difference: {}", max_diff);

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    const N: usize = 4096;
    let a_host: Vec<f32> = vec![1.0; N * N];
    let b_host: Vec<f32> = vec![1.0; N * N];


    let seq_len = 2;
    const dim: usize = 4;
    const num_heads: usize = 2;


    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let cuda_kernels = CudaKernels::new(&ctx)?;


    let a = GpuTensor::from_cpu(stream.clone(), &vec![1.0f32; 6], vec![2, 3])?;
    let b = GpuTensor::from_cpu(stream.clone(), &vec![1.0f32; 6], vec![3, 2])?;
    let c = matmul_gpu(&a, &b, 2, 2, 3)?;
    let result = c.to_cpu()?;
    println!("Result of matmul_gpu {:?}", result);



    let mut x = GpuTensor::from_cpu(stream.clone(), &vec![2.0f32, 4.0, 6.0, 8.0], vec![4])?;
    scale_gpu(&mut x, 0.5)?;
    let result = x.to_cpu()?;
    println!("Result of scale_gpu {:?}", result);

    let mut t = GpuTensor::from_cpu(stream.clone(), &vec![1.0, 2.0, 3.0, 4.0], vec![4])?;
    softmax_cudnn_gpu(&mut t)?;
    let result = t.to_cpu()?;
    println!("Result of softmax_cudnn_gpu {:?}", result);

    let mut scores = GpuTensor::from_cpu(stream.clone(), &vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    softmax_cudnn_gpu_2d(&mut scores)?;


    let stream3 = ctx.default_stream();
    let q = GpuTensor::from_cpu(stream3.clone(), &vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
    // K: 2x3 matrix (will be transposed to 3x2)
    let k = GpuTensor::from_cpu(stream3.clone(), &vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;

    let scores = matmul_gpu_bt(&q, &k, 2, 2, 3)?;
    println!("matmul_gpu_bt: {:?}", scores.to_cpu()?);

    let seq_len = 2;
    let x = GpuTensor::from_cpu(stream3.clone(), &vec![1.0f32; seq_len * dim], vec![seq_len, dim])?;
    let w_q = GpuTensor::from_cpu(stream3.clone(), &vec![1.0f32; dim * dim], vec![dim, dim])?;
    let w_k = GpuTensor::from_cpu(stream3.clone(), &vec![1.0f32; dim * dim], vec![dim, dim])?;
    let w_v = GpuTensor::from_cpu(stream3.clone(), &vec![1.0f32; dim * dim], vec![dim, dim])?;

    let output = attention_gpu(&x, &w_q, &w_k, &w_v, seq_len, dim)?;
    println!("attention_gpu: {:?}", output.to_cpu()?);

    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let t = GpuTensor::from_cpu(stream3.clone(), &data, vec![2, 4])?;
    let head0 = extract_head(&t, 2, 4, 2, 0)?;
    println!("head0: {:?}", head0.to_cpu());
    let head1 = extract_head(&t, 2, 4, 2, 1)?;
    println!("head1: {:?}", head1.to_cpu());

    let mut combined = GpuTensor::from_cpu(stream3.clone(), &vec![0.0f32; 8], vec![2, 4])?;
    combine_heads(&mut combined, &head0, 2, 4, 2, 0)?;
    combine_heads(&mut combined, &head1, 2, 4, 2, 1)?;

    println!("combined: {:?}", combined.to_cpu());

    let seq_len = 2;


    let mut a = GpuTensor::from_cpu(stream3.clone(), &vec![1.0f32, 2.0, 3.0, 4.0], vec![4])?;
    let b = GpuTensor::from_cpu(stream3.clone(), &vec![10.0f32, 20.0, 30.0, 40.0], vec![4])?;
    residual_add(&mut a, &b)?;
    println!("Result of multi-head residual attention {:?}", a.to_cpu()?);

    let mut x = GpuTensor::from_cpu(stream3.clone(), &vec![-2.0f32, -1.0, 0.0, 1.0, 2.0], vec![5])?;
    gelu(&mut x)?;
    println!("Result of gelu {:?}", x.to_cpu()?);

    let hidden_dim = 8;

    let x = GpuTensor::from_cpu(stream3.clone(), &vec![1.0f32; seq_len * dim], vec![seq_len, dim])?;
    let w1 = GpuTensor::from_cpu(stream3.clone(), &vec![0.1f32; dim * hidden_dim], vec![dim, hidden_dim])?;
    let w2 = GpuTensor::from_cpu(stream3.clone(), &vec![0.1f32; hidden_dim * dim], vec![hidden_dim, dim])?;

    let output = ffn(&x, &w1, &w2, seq_len, dim, hidden_dim)?;
    println!("ffn: {:?}", output.to_cpu()?);

    let mut test = GpuTensor::from_cpu(stream3.clone(), &vec![0.4f32], vec![1])?;
    gelu(&mut test)?;
    println!("gelu(0.4): {:?}", test.to_cpu()?);

    let x = GpuTensor::from_cpu(stream3.clone(), &vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![seq_len, dim])?;
    let gamma = GpuTensor::from_cpu(stream3.clone(), &vec![1.0f32; dim], vec![dim])?;
    let beta = GpuTensor::from_cpu(stream3.clone(), &vec![0.0f32; dim], vec![dim])?;

    let output = layer_norm(&x, &gamma, &beta, 1e-5)?;
    println!("Layer norm {:?}", output.to_cpu());

    let x = GpuTensor::from_cpu(stream3.clone(), &vec![1.0f32, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0], vec![seq_len, dim])?;
    let gamma = GpuTensor::from_cpu(stream3.clone(), &vec![1.0f32; dim], vec![dim])?;
    let beta = GpuTensor::from_cpu(stream3.clone(), &vec![0.0f32; dim], vec![dim])?;
    let vocab_size = 8;
    let w_vocab = GpuTensor::from_cpu(stream3.clone(), &vec![
        0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
        0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1,
        0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2,
        0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3,
    ], vec![dim, vocab_size])?;

    let logits = output_projections(&x, &gamma, &beta, &w_vocab, seq_len, dim, vocab_size)?;

    let last_logits = get_last_logits(&logits)?;

    let next_token = arg_max(last_logits.as_slice());

    println!("logits shape: {:?}", logits.shape);
    println!("next token: {}", next_token);

    let vocab_size = 4;
    let dim2 = 3;

    let embed_table = GpuTensor::from_cpu(stream3.clone(), &vec![
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6,
        0.7, 0.8, 0.9,
        1.0, 1.1, 1.2
    ], vec![vocab_size, dim2])?;

    let token_ids = vec![2, 0, 3];

    let embedded = embeddings(&token_ids, &embed_table)?;
    println!("token_ids: {:?}", token_ids);
    println!("embed_table: {:?}", embed_table.to_cpu()?);
    println!("embeddings shape: {:?}", embedded.shape);
    println!("embeddings: {:?}", embedded.to_cpu());

    let mut x = GpuTensor::from_cpu(stream.clone(), &vec![1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], vec![seq_len, dim])?;
    rope(&mut x)?;

    println!("rope: {:?}", x.to_cpu()?);

    let x = GpuTensor::from_cpu(stream.clone(), &vec![1.0f32, 2.0, 3.0, 4.0,   2.0, 4.0, 6.0, 8.0], vec![2, 4])?;
    let gamma = GpuTensor::from_cpu(stream.clone(), &vec![1.0f32, 1.0, 1.0, 1.0], vec![4])?;
    let out = rms_norm(&x, &gamma)?;

    println!("Result of rms_norm {:?}", out.to_cpu()?);

    let mut x = GpuTensor::from_cpu(stream3.clone(), &vec![-2.0f32, -1.0, 0.0, 1.0, 2.0], vec![5])?;
    silu(&mut x)?;
    println!("Result of silu {:?}", x.to_cpu()?);


    let x = GpuTensor::from_cpu(stream3.clone(), &vec![
        1.0f32, 2.0, 3.0, 4.0,
        0.5, 1.0, 1.5, 2.0,
    ], vec![seq_len, dim])?;

    let w_gate = GpuTensor::from_cpu(stream3.clone(), &vec![0.1f32; dim*hidden_dim], vec![dim, hidden_dim])?;
    let w_up = GpuTensor::from_cpu(stream3.clone(), &vec![0.2f32; dim*hidden_dim], vec![dim, hidden_dim])?;
    let w_down = GpuTensor::from_cpu(stream3.clone(), &vec![0.1f32; dim*hidden_dim], vec![dim, hidden_dim])?;

    let output = ffn_swiglu(&x, &w_gate, &w_up, &w_down, seq_len, dim, hidden_dim)?;

    println!("output shape: {:?}", output.shape);
    println!("ffn_swiglu {:?}", output.to_cpu()?);

    test_layer0()?;

    test_rope_kernel()?;

    Ok(())
}