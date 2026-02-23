use anyhow::{Context, Result};
use ndarray::{ArrayD, IxDyn};
use ort::session::Session;

use crate::inference;
use crate::postprocess;

/// Run the full brain-extraction pipeline on input data.
///
/// `data`: 3D f32 array from NIfTI (H, W, D)
/// `session`: pre-built ONNX inference session
///
/// Returns `(brain_mask, brain_extracted)`:
/// - `brain_mask`: u8 binary mask (0 or 1)
/// - `brain_extracted`: f32 input × mask
pub fn run_bx(data: &ArrayD<f32>, session: &mut Session) -> Result<(ArrayD<u8>, ArrayD<f32>)> {
    let shape = data.shape();
    assert_eq!(shape.len(), 3, "Expected 3D input");

    // 1. Normalize to [0, 1]
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let normalized = if max_val > 0.0 {
        data.mapv(|v| v / max_val)
    } else {
        data.clone()
    };

    println!("  Normalized max: {:.4}", max_val);

    // 2. ONNX inference
    println!("  Running ONNX inference...");
    let logits_5d = inference::run_onnx(session, &normalized)
        .context("ONNX inference failed")?;

    // Output shape is (1, 1, H, W, D) — sigmoid mode
    // Extract the single-channel 3D logits
    let logits_shape: Vec<usize> = logits_5d.shape().to_vec();
    println!("  Model output shape: {:?}", logits_shape);

    let logits_3d = logits_5d
        .into_shape_with_order(IxDyn(&[logits_shape[2], logits_shape[3], logits_shape[4]]))
        .context("Failed to reshape logits to 3D")?;

    // 3. Sigmoid → probability
    let prob = postprocess::sigmoid(&logits_3d);

    // 4. Threshold at 0.5
    let mask = postprocess::threshold_mask(&prob, 0.5);
    let mask_count: usize = mask.iter().filter(|&&v| v > 0).count();
    println!("  Mask voxels before cleanup: {}", mask_count);

    // 5. Keep largest connected component
    let mask = postprocess::largest_connected_component_3d(&mask);
    let mask_count: usize = mask.iter().filter(|&&v| v > 0).count();
    println!("  Mask voxels after cleanup: {}", mask_count);

    // 6. Brain extracted = input × mask
    let brain = ndarray::Zip::from(data)
        .and(&mask)
        .map_collect(|&d, &m| if m > 0 { d } else { 0.0 });

    Ok((mask, brain))
}
