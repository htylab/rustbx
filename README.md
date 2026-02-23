# rustbx

A fast brain extraction CLI tool written in Rust. Single binary, no dependencies, no Python required.

This is a side project of [TigerBx](https://github.com/htylab/tigerbx), providing the same deep-learning-based brain extraction as a standalone native executable with minimal overhead.

## Features

- Single self-contained binary (~40 MB, ONNX model embedded)
- Supports `.nii` and `.nii.gz` input
- Batch processing with glob patterns, directories, or multiple files
- ONNX Runtime inference (CPU)
- Cross-platform: Windows and Linux

## Usage

```bash
# Single file
rustbx T1w.nii.gz

# Multiple files
rustbx sub01_T1w.nii.gz sub02_T1w.nii.gz

# Glob pattern
rustbx data/*.nii.gz

# Directory (scans for NIfTI files, non-recursive)
rustbx data/

# Specify output directory
rustbx T1w.nii.gz -o output/

# Use external model instead of embedded one
rustbx T1w.nii.gz -m custom_model.onnx
```

## Output

For each input file, two outputs are generated:

| Suffix | Description |
|--------|-------------|
| `_tbet.nii.gz` | Brain-extracted image (input masked by brain mask) |
| `_tbetmask.nii.gz` | Binary brain mask (0 or 1) |

## Download

Pre-built binaries are available from [GitHub Actions](https://github.com/htylab/rustbx/actions) artifacts:

- **rustbx-win-x64** — Windows x64
- **rustbx-linux-x64** — Linux x64

## Build from Source

Requirements: [Rust](https://rustup.rs/) toolchain.

```bash
# Clone
git clone https://github.com/htylab/rustbx.git
cd rustbx

# Download model
mkdir models
curl -L -o models/mprage_bet_v005_mixsynthv4.onnx \
  "https://github.com/htylab/tigerbx/releases/download/modelhub/mprage_bet_v005_mixsynthv4.onnx"

# Build
cargo build --release
```

The binary will be at `target/release/rustbx` (Linux) or `target\release\rustbx.exe` (Windows).

## License

See [TigerBx](https://github.com/htylab/tigerbx) for license information.
