use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::sync::Arc;
use cudarc::driver::{CudaContext, CudaStream};
use crate::gguf::gguf::read_gguf_header;
use crate::gguf::metadata::{get_metadata_u32, GgufMetadataValue, TensorMetadata};
use crate::gguf::tensor_data::{load_tensor, read_tensor_f32, read_tensor_q4_0_transposed, read_tensor_q6_k};
use crate::gpu_tensor::GpuTensor;

#[derive(Debug)]
pub struct ModelConfig {
    pub dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
}

impl ModelConfig {
    pub fn from_gguf_metadata(metadata: &HashMap<String, GgufMetadataValue>) -> Result<Self, Box<dyn Error>> {

        let vocab_size = match metadata.get("tokenizer.ggml.tokens") {
            Some(GgufMetadataValue::ArrayString(tokens)) => tokens.len(),
            _ => return Err("Cannot determine vocab_size from tokenizer.ggml.tokens".into()),
        };

        Ok(Self {
            dim: get_metadata_u32(&metadata, "llama.embedding_length")?,
            num_heads: get_metadata_u32(&metadata, "llama.attention.head_count")?,
            num_kv_heads: get_metadata_u32(&metadata, "llama.attention.head_count_kv")?,
            hidden_dim: get_metadata_u32(&metadata, "llama.feed_forward_length")?,
            num_layers: get_metadata_u32(&metadata, "llama.block_count")?,
            vocab_size
        })
    }
}

pub struct LayerWeights {
    pub attn_norm: GpuTensor,
    pub ffn_norm: GpuTensor,
    pub w_q: GpuTensor,
    pub w_k: GpuTensor,
    pub w_v: GpuTensor,
    pub w_o: GpuTensor,
    pub w_gate: GpuTensor,
    pub w_up: GpuTensor,
    pub w_down: GpuTensor,
}

impl LayerWeights {
    pub fn load_layer(file: &mut File, stream: Arc<CudaStream>, tensor_map: &HashMap<String, TensorMetadata>, layer_id: usize, data_start_offset: u64, config: &ModelConfig) -> Result<Self, Box<dyn Error>> {
        let head_dim = config.dim / config.num_heads;
        let kv_dim = config.num_kv_heads * head_dim;

        let attn_norm_tensor = get_tensor_for_layer(&tensor_map, layer_id, "attn_norm.weight")?;
        let attn_norm_data = read_tensor_f32(file, attn_norm_tensor, data_start_offset)?;
        let attn_norm = GpuTensor::from_cpu(stream.clone(), &attn_norm_data, vec![config.dim])?;

        let ffn_norm_tensor = get_tensor_for_layer(&tensor_map, layer_id, "ffn_norm.weight")?;
        let ffn_norm_data = read_tensor_f32(file, ffn_norm_tensor, data_start_offset)?;
        let ffn_norm = GpuTensor::from_cpu(stream.clone(), &ffn_norm_data, vec![config.dim])?;

        let w_q_tensor = get_tensor_for_layer(&tensor_map, layer_id, "attn_q.weight")?;
        let w_q_data = read_tensor_q4_0_transposed(file, w_q_tensor, data_start_offset, config.dim, config.dim)?;
        let w_q= GpuTensor::from_cpu(stream.clone(), &w_q_data, vec![config.dim, config.dim])?;

        let w_k_tensor = get_tensor_for_layer(&tensor_map, layer_id, "attn_k.weight")?;
        let w_k_data = read_tensor_q4_0_transposed(file, w_k_tensor, data_start_offset, kv_dim, config.dim)?;
        let w_k= GpuTensor::from_cpu(stream.clone(), &w_k_data, vec![kv_dim, config.dim])?;

        let w_v_tensor = get_tensor_for_layer(&tensor_map, layer_id, "attn_v.weight")?;
        let w_v_data = read_tensor_q4_0_transposed(file, w_v_tensor, data_start_offset, kv_dim, config.dim)?;
        let w_v= GpuTensor::from_cpu(stream.clone(), &w_v_data, vec![kv_dim, config.dim])?;

        let w_o_tensor = get_tensor_for_layer(&tensor_map, layer_id, "attn_output.weight")?;
        let w_o_data = read_tensor_q4_0_transposed(file, w_o_tensor, data_start_offset, config.dim, config.dim)?;
        let w_o= GpuTensor::from_cpu(stream.clone(), &w_o_data, vec![config.dim, config.dim])?;

        let w_gate_tensor = get_tensor_for_layer(&tensor_map, layer_id, "ffn_gate.weight")?;
        let w_gate_data = read_tensor_q4_0_transposed(file, w_gate_tensor, data_start_offset, config.hidden_dim, config.dim)?;
        let w_gate= GpuTensor::from_cpu(stream.clone(), &w_gate_data, vec![config.hidden_dim, config.dim])?;

        let w_up_tensor = get_tensor_for_layer(&tensor_map, layer_id, "ffn_up.weight")?;
        let w_up_data = read_tensor_q4_0_transposed(file, w_up_tensor, data_start_offset, config.hidden_dim, config.dim)?;
        let w_up= GpuTensor::from_cpu(stream.clone(), &w_up_data, vec![config.hidden_dim, config.dim])?;

        let w_down_tensor = get_tensor_for_layer(&tensor_map, layer_id, "ffn_down.weight")?;
        let w_down_data = read_tensor_q4_0_transposed(file, w_down_tensor, data_start_offset, config.dim, config.hidden_dim)?;
        let w_down= GpuTensor::from_cpu(stream.clone(), &w_down_data, vec![config.dim, config.hidden_dim])?;

        let layer = Self {
            attn_norm, ffn_norm, w_q, w_k, w_v, w_o, w_gate, w_up, w_down
        };

        Ok(layer)
    }
}

pub struct ModelWeights {
    pub embed_tokens: GpuTensor,
    pub final_norm: GpuTensor,
    pub output_weights: GpuTensor,
    pub layers: Vec<LayerWeights>,
}


impl ModelWeights {
}

fn get_tensor<'a>(tensor_map: &'a HashMap<String, TensorMetadata>, key: &str) -> Result<&'a TensorMetadata, Box<dyn Error>> {
    let tensor = tensor_map.get(key).ok_or(format!("unknown tensor key {}", key))?;
    Ok(tensor)
}

fn get_tensor_for_layer<'a>(tensor_map: &'a HashMap<String, TensorMetadata>, layer_id: usize, key_pattern: &str) -> Result<&'a TensorMetadata, Box<dyn Error>> {
    let key = format!("blk.{}.{}", layer_id, key_pattern);
    let tensor = get_tensor(tensor_map, key.as_str())?;
    Ok(tensor)
}

pub fn load_gguf(file: &mut File, stream: Arc<CudaStream>) -> Result<(ModelConfig, ModelWeights), Box<dyn Error>> {
    println!("====Loading model from gguf =========");

    let (metadata, tensor_map, data_start_offset) = read_gguf_header(file)?;
    let config = ModelConfig::from_gguf_metadata(&metadata)?;
    println!("Model config {:?}", &config);

    println!("Loading embed_tokens");
    let embed_tensor = get_tensor(&tensor_map, "token_embd.weight")?;
    let embed_data = read_tensor_f32(file, &embed_tensor, data_start_offset)?;
    let embed_tokens = GpuTensor::from_cpu(stream.clone(), &embed_data, vec![config.vocab_size, config.dim])?;

    println!("Loading output_norm");
    let norm_tensor = get_tensor(&tensor_map, "output_norm.weight")?;
    let norm_data = read_tensor_f32(file, &norm_tensor, data_start_offset)?;
    let final_norm = GpuTensor::from_cpu(stream.clone(), &norm_data, vec![config.dim])?;

    println!("Loading output_weights");
    let output_tensor = get_tensor(&tensor_map, "output.weight")?;
    let output_data = read_tensor_q6_k(file, &output_tensor, data_start_offset)?;
    let output_weights = GpuTensor::from_cpu(stream.clone(), &output_data, vec![config.dim, config.vocab_size])?;

    println!("Weights loaded");

    let mut layers = Vec::with_capacity(config.num_layers);

    for i in 0..config.num_layers {
        if i % 5 == 0 {
            println!("Loading layer {}", i);
        }
        let layer = LayerWeights::load_layer(file, stream.clone(), &tensor_map, i, data_start_offset, &config)?;
        layers.push(layer);
    }


    let model_weight = ModelWeights {
        embed_tokens, final_norm, output_weights, layers
    };

    println!("Model layers loaded {}", model_weight.layers.len());

    Ok((config, model_weight))
}

