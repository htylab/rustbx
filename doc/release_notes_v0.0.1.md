# Release Notes

## v0.0.1 — Initial Release

First public release of rustbx, a fast brain extraction CLI tool written in Rust.
Side project of [TigerBx](https://github.com/htylab/tigerbx).

### Features

- Single self-contained binary with ONNX model embedded (~40 MB)
- Brain extraction pipeline: normalize → ONNX inference → sigmoid → threshold → largest connected component
- Supports single files, multiple files, glob patterns, and directory input
- Batch processing with session caching (ONNX session built once, reused for all files)
- Per-file processing time displayed in output
- Output files: `_tbet.nii.gz` (brain-extracted) and `_tbetmask.nii.gz` (binary mask)
- Optional `-o` flag for output directory
- Optional `-m` flag for external ONNX model

### Platforms

- Windows x64
- Linux x64 (Ubuntu 24.04+)

### CI/CD

- GitHub Actions: auto build on push (Windows + Linux)
- Tag-triggered release workflow (`git push origin v*` creates draft GitHub Release with zips)
