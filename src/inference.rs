use anyhow::{Context, Result};
use ndarray::{ArrayD, IxDyn};
use ort::session::Session;
use ort::value::Tensor;

/// Create an ONNX inference session from model bytes.
///
/// This should be called once and the session reused for all files.
pub fn create_session(model_bytes: &[u8]) -> Result<Session> {
    let session = Session::builder()
        .context("Failed to create session builder")?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .context("Failed to set optimization level")?
        .commit_from_memory(model_bytes)
        .context("Failed to load ONNX model from memory")?;

    Ok(session)
}

/// Run inference on a pre-built session.
///
/// Input: 3D f32 array (H, W, D) — will be reshaped to (1, 1, H, W, D).
/// Output: raw logits array with shape matching the ONNX model output.
pub fn run_onnx(session: &mut Session, input: &ArrayD<f32>) -> Result<ArrayD<f32>> {
    // Reshape input from (H, W, D) to (1, 1, H, W, D)
    let shape = input.shape();
    let shape_5d: Vec<usize> = vec![1, 1, shape[0], shape[1], shape[2]];
    let flat_data: Vec<f32> = input.iter().cloned().collect();

    // Create ort Tensor from (shape, data) tuple
    let input_tensor = Tensor::from_array((shape_5d.as_slice(), flat_data))
        .context("Failed to create input tensor")?;

    // Get input name
    let input_name = session.inputs()[0].name().to_string();

    // Run inference
    let outputs = session
        .run(ort::inputs![input_name.as_str() => input_tensor])
        .context("ONNX inference failed")?;

    // Extract output tensor — returns (&Shape, &[f32])
    let (out_shape, out_data) = outputs[0]
        .try_extract_tensor::<f32>()
        .context("Failed to extract output tensor")?;

    // Reconstruct ArrayD from shape and data
    let shape_vec: Vec<usize> = out_shape.iter().map(|&s| s as usize).collect();
    let output = ArrayD::from_shape_vec(IxDyn(&shape_vec), out_data.to_vec())
        .context("Failed to reconstruct output array")?;

    Ok(output)
}
