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
mod model;

use std::error::Error;
use std::fs::File;
use cudarc::driver::CudaContext;
use crate::attention::{attention_gpu, attention_gqa};
use crate::gguf::gguf::read_gguf_header;
use crate::gguf::tensor_data::{read_tensor_f32, read_tensor_q4_0};
use crate::gpu_tensor::GpuTensor;
use crate::matmul::{matmul_gpu, matmul_gpu_bt};
use crate::model::load_gguf;
use crate::residual::residual_add;
use crate::rms_norm::rms_norm;
use crate::rope::rope;
use crate::transformer_block::{transformer_block_llama};
use crate::transpose::{transpose_2d};


fn test_block_vs_hf() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let seq_len = 4usize;
    let dim = 2048usize;
    let num_heads = 32usize;
    let num_kv_heads = 4usize;
    let hidden_dim = 5632usize;
    let kv_dim = num_kv_heads * (dim / num_heads);

    let x = GpuTensor::from_cpu(stream.clone(), &load_f32_bin("python/test_data/hf_block_input.bin"), vec![seq_len, dim])?;
    let gamma1 = GpuTensor::from_cpu(stream.clone(), &load_f32_bin("python/test_data/hf_block_gamma1.bin"), vec![dim])?;
    let gamma2 = GpuTensor::from_cpu(stream.clone(), &load_f32_bin("python/test_data/hf_block_gamma2.bin"), vec![dim])?;
    let w_q = GpuTensor::from_cpu(stream.clone(), &load_f32_bin("python/test_data/hf_block_wq.bin"), vec![dim, dim])?;
    let w_k = GpuTensor::from_cpu(stream.clone(), &load_f32_bin("python/test_data/hf_block_wk.bin"), vec![dim, kv_dim])?;
    let w_v = GpuTensor::from_cpu(stream.clone(), &load_f32_bin("python/test_data/hf_block_wv.bin"), vec![dim, kv_dim])?;
    let w_o = GpuTensor::from_cpu(stream.clone(), &load_f32_bin("python/test_data/hf_block_wo.bin"), vec![dim, dim])?;
    let w_gate = GpuTensor::from_cpu(stream.clone(), &load_f32_bin("python/test_data/hf_block_wgate.bin"), vec![dim, hidden_dim])?;
    let w_up = GpuTensor::from_cpu(stream.clone(), &load_f32_bin("python/test_data/hf_block_wup.bin"), vec![dim, hidden_dim])?;
    let w_down = GpuTensor::from_cpu(stream.clone(), &load_f32_bin("python/test_data/hf_block_wdown.bin"), vec![hidden_dim, dim])?;
    let expected = load_f32_bin("python/test_data/hf_block_output.bin");

    let output = transformer_block_llama(
        &x, &gamma1, &gamma2,
        &w_q, &w_k, &w_v, &w_o,
        &w_gate, &w_up, &w_down,
        seq_len, dim, num_heads, num_kv_heads, hidden_dim
    )?;
    let result = output.to_cpu()?;

    println!("Expected first 8: {:?}", &expected[..8]);
    println!("Got      first 8: {:?}", &result[..8]);

    let max_diff: f32 = result.iter().zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Max diff: {}", max_diff);

    for tol in [1e-4, 1e-3, 1e-2] {
        let count = result.iter().zip(expected.iter())
            .filter(|(a, b)| ((*a) - (*b)).abs() > tol)
            .count();
        println!("Values exceeding {}: {}/{}", tol, count, result.len());
    }

    assert!(max_diff < 1e-2, "Block vs HF FAILED: max_diff={}", max_diff);
    println!("PASSED");

    Ok(())
}
fn test_attention_vs_hf() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let seq_len = 4usize;
    let dim = 2048usize;
    let num_heads = 32usize;
    let num_kv_heads = 4usize;
    let head_dim = dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;

    let attn_input = load_f32_bin("python/test_data/hf_attn_input.bin");
    let wq_data = load_f32_bin("python/test_data/hf_attn_wq.bin");
    let wk_data = load_f32_bin("python/test_data/hf_attn_wk.bin");
    let wv_data = load_f32_bin("python/test_data/hf_attn_wv.bin");
    let wo_data = load_f32_bin("python/test_data/hf_attn_wo.bin");
    let expected = load_f32_bin("python/test_data/hf_attn_expected.bin");

    let x = GpuTensor::from_cpu(stream.clone(), &attn_input, vec![seq_len, dim])?;
    let w_q = GpuTensor::from_cpu(stream.clone(), &wq_data, vec![dim, dim])?;
    let w_k = GpuTensor::from_cpu(stream.clone(), &wk_data, vec![dim, kv_dim])?;
    let w_v = GpuTensor::from_cpu(stream.clone(), &wv_data, vec![dim, kv_dim])?;
    let w_o = GpuTensor::from_cpu(stream.clone(), &wo_data, vec![dim, dim])?;

    let output = attention_gqa(&x, &w_q, &w_k, &w_v, &w_o, seq_len, dim, num_heads, num_kv_heads)?;
    let result = output.to_cpu()?;

    println!("Expected first 8: {:?}", &expected[..8]);
    println!("Got      first 8: {:?}", &result[..8]);

    let max_diff: f32 = result.iter().zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Max diff: {}", max_diff);

    // Count mismatches at various thresholds
    for tol in [1e-3, 1e-2, 1e-1] {
        let count = result.iter().zip(expected.iter())
            .filter(|(a, b)| ((*a) - (*b)).abs() > tol)
            .count();
        println!("Values exceeding {}: {}/{}", tol, count, result.len());
    }

    assert!(max_diff < 1e-2, "Attention vs HF FAILED: max_diff={}", max_diff);
    println!("PASSED");

    Ok(())
}

fn test_rope_vs_hf() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let seq_len = 4usize;
    let head_dim = 64usize;

    // Test Q head 0
    let q_pre = load_f32_bin("python/test_data/hf_q_head0_pre_rope.bin");
    let q_expected = load_f32_bin("python/test_data/hf_q_head0_post_rope.bin");

    let mut q_gpu = GpuTensor::from_cpu(stream.clone(), &q_pre, vec![seq_len, head_dim])?;
    rope(&mut q_gpu)?;
    let q_result = q_gpu.to_cpu()?;

    println!("Q RoPE expected pos0 first 8: {:?}", &q_expected[..8]);
    println!("Q RoPE got      pos0 first 8: {:?}", &q_result[..8]);
    println!("Q RoPE expected pos1 first 8: {:?}", &q_expected[head_dim..head_dim+8]);
    println!("Q RoPE got      pos1 first 8: {:?}", &q_result[head_dim..head_dim+8]);

    let q_diff: f32 = q_result.iter().zip(q_expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Q RoPE max diff: {}", q_diff);
    assert!(q_diff < 1e-5, "Q RoPE FAILED: max_diff={}", q_diff);
    println!("Q RoPE PASSED");

    // Test K head 0
    let k_pre = load_f32_bin("python/test_data/hf_k_head0_pre_rope.bin");
    let k_expected = load_f32_bin("python/test_data/hf_k_head0_post_rope.bin");

    let mut k_gpu = GpuTensor::from_cpu(stream.clone(), &k_pre, vec![seq_len, head_dim])?;
    rope(&mut k_gpu)?;
    let k_result = k_gpu.to_cpu()?;

    println!("\nK RoPE expected pos1 first 8: {:?}", &k_expected[head_dim..head_dim+8]);
    println!("K RoPE got      pos1 first 8: {:?}", &k_result[head_dim..head_dim+8]);

    let k_diff: f32 = k_result.iter().zip(k_expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("K RoPE max diff: {}", k_diff);
    assert!(k_diff < 1e-5, "K RoPE FAILED: max_diff={}", k_diff);
    println!("K RoPE PASSED");

    Ok(())
}
fn test_qkv_proj_vs_hf() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let seq_len = 4usize;
    let dim = 2048usize;
    let num_kv_heads = 4usize;
    let head_dim = 64usize;
    let kv_dim = num_kv_heads * head_dim;

    let rms_data = load_f32_bin("python/test_data/hf_rms_output.bin");
    let wq_data = load_f32_bin("python/test_data/hf_wq.bin");
    let wk_data = load_f32_bin("python/test_data/hf_wk.bin");
    let wv_data = load_f32_bin("python/test_data/hf_wv.bin");
    let q_expected = load_f32_bin("python/test_data/hf_q_expected.bin");
    let k_expected = load_f32_bin("python/test_data/hf_k_expected.bin");
    let v_expected = load_f32_bin("python/test_data/hf_v_expected.bin");

    let rms_out = GpuTensor::from_cpu(stream.clone(), &rms_data, vec![seq_len, dim])?;
    let w_q = GpuTensor::from_cpu(stream.clone(), &wq_data, vec![dim, dim])?;
    let w_k = GpuTensor::from_cpu(stream.clone(), &wk_data, vec![dim, kv_dim])?;
    let w_v = GpuTensor::from_cpu(stream.clone(), &wv_data, vec![dim, kv_dim])?;

    let q = matmul_gpu(&rms_out, &w_q, seq_len as i32, dim as i32, dim as i32)?;
    let k = matmul_gpu(&rms_out, &w_k, seq_len as i32, kv_dim as i32, dim as i32)?;
    let v = matmul_gpu(&rms_out, &w_v, seq_len as i32, kv_dim as i32, dim as i32)?;

    let q_result = q.to_cpu()?;
    let k_result = k.to_cpu()?;
    let v_result = v.to_cpu()?;

    for (name, result, expected) in [("Q", &q_result, &q_expected), ("K", &k_result, &k_expected), ("V", &v_result, &v_expected)] {
        let max_diff: f32 = result.iter().zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("{} expected first 8: {:?}", name, &expected[..8]);
        println!("{} got      first 8: {:?}", name, &result[..8]);
        println!("{} max diff: {}", name, max_diff);
        assert!(max_diff < 1e-3, "{} projection FAILED: max_diff={}", name, max_diff);
        println!("{} PASSED", name);
    }

    Ok(())
}
fn test_rms_norm_vs_hf() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let seq_len = 4usize;
    let dim = 2048usize;

    let x_data = load_f32_bin("python/test_data/hf_rms_input.bin");
    let gamma_data = load_f32_bin("python/test_data/hf_rms_gamma.bin");
    let expected = load_f32_bin("python/test_data/hf_rms_expected.bin");

    let x = GpuTensor::from_cpu(stream.clone(), &x_data, vec![seq_len, dim])?;
    let gamma = GpuTensor::from_cpu(stream.clone(), &gamma_data, vec![dim])?;

    let output = rms_norm(&x, &gamma)?;
    let result = output.to_cpu()?;

    println!("Expected first 8: {:?}", &expected[..8]);
    println!("Got      first 8: {:?}", &result[..8]);

    let max_diff: f32 = result.iter().zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Max diff: {}", max_diff);
    assert!(max_diff < 1e-4, "RMS norm vs HF FAILED: max_diff={}", max_diff);
    println!("PASSED");

    Ok(())
}
fn test_attention_gqa_kernel() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // Test 1: small
    println!("\n=== GQA Small Test ===");
    let seq_len = 4usize;
    let dim = 32usize;
    let num_heads = 4usize;
    let num_kv_heads = 2usize;
    let kv_dim = num_kv_heads * (dim / num_heads);

    let x_data = load_f32_bin("python/test_data/gqa_x_small.bin");
    let wq_data = load_f32_bin("python/test_data/gqa_wq_small.bin");
    let wk_data = load_f32_bin("python/test_data/gqa_wk_small.bin");
    let wv_data = load_f32_bin("python/test_data/gqa_wv_small.bin");
    let wo_data = load_f32_bin("python/test_data/gqa_wo_small.bin");
    let expected = load_f32_bin("python/test_data/gqa_expected_small.bin");

    let x = GpuTensor::from_cpu(stream.clone(), &x_data, vec![seq_len, dim])?;
    let w_q = GpuTensor::from_cpu(stream.clone(), &wq_data, vec![dim, dim])?;
    let w_k = GpuTensor::from_cpu(stream.clone(), &wk_data, vec![dim, kv_dim])?;
    let w_v = GpuTensor::from_cpu(stream.clone(), &wv_data, vec![dim, kv_dim])?;
    let w_o = GpuTensor::from_cpu(stream.clone(), &wo_data, vec![dim, dim])?;

    let output = attention_gqa(&x, &w_q, &w_k, &w_v, &w_o, seq_len, dim, num_heads, num_kv_heads)?;
    let result = output.to_cpu()?;

    println!("Expected first 8: {:?}", &expected[..8]);
    println!("Got      first 8: {:?}", &result[..8]);

    let max_diff: f32 = result.iter().zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Max diff: {}", max_diff);
    assert!(max_diff < 1e-4, "GQA small test FAILED: max_diff={}", max_diff);
    println!("PASSED");

    // Test 2: realistic
    println!("\n=== GQA Realistic Test ===");
    let seq_len = 4usize;
    let dim = 64usize;
    let num_heads = 4usize;
    let num_kv_heads = 2usize;
    let kv_dim = num_kv_heads * (dim / num_heads);

    let x_data = load_f32_bin("python/test_data/gqa_x_real.bin");
    let wq_data = load_f32_bin("python/test_data/gqa_wq_real.bin");
    let wk_data = load_f32_bin("python/test_data/gqa_wk_real.bin");
    let wv_data = load_f32_bin("python/test_data/gqa_wv_real.bin");
    let wo_data = load_f32_bin("python/test_data/gqa_wo_real.bin");
    let expected = load_f32_bin("python/test_data/gqa_expected_real.bin");

    let x = GpuTensor::from_cpu(stream.clone(), &x_data, vec![seq_len, dim])?;
    let w_q = GpuTensor::from_cpu(stream.clone(), &wq_data, vec![dim, dim])?;
    let w_k = GpuTensor::from_cpu(stream.clone(), &wk_data, vec![dim, kv_dim])?;
    let w_v = GpuTensor::from_cpu(stream.clone(), &wv_data, vec![dim, kv_dim])?;
    let w_o = GpuTensor::from_cpu(stream.clone(), &wo_data, vec![dim, dim])?;

    let output = attention_gqa(&x, &w_q, &w_k, &w_v, &w_o, seq_len, dim, num_heads, num_kv_heads)?;
    let result = output.to_cpu()?;

    println!("Expected first 8: {:?}", &expected[..8]);
    println!("Got      first 8: {:?}", &result[..8]);

    let max_diff: f32 = result.iter().zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Max diff: {}", max_diff);
    assert!(max_diff < 1e-4, "GQA realistic test FAILED: max_diff={}", max_diff);
    println!("PASSED");

    Ok(())
}
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




    let mut file = File::open("tinyllama-1.1b-chat-v1.0.Q4_0.gguf")?;
    let (_, tensor_metadata_map, data_start_offset) = read_gguf_header(&mut file)?;

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

    let mut file = File::open("tinyllama-1.1b-chat-v1.0.Q4_0.gguf")?;
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    load_gguf(&mut file, stream)?;
    Ok(())
}
