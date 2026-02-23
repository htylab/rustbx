# Build release — single self-contained exe with embedded ONNX model

param(
    [switch]$Package  # Create a zip package
)

$env:Path += ";$env:USERPROFILE\.cargo\bin"
$env:RUSTFLAGS = ""

Write-Host "=== Building release ===" -ForegroundColor Cyan
cargo build --release 2>&1 | ForEach-Object { $_.ToString() }
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build FAILED" -ForegroundColor Red
    exit 1
}

# Copy exe to project root
Copy-Item target\release\rustbx.exe -Destination "rustbx.exe" -Force
$exe = Get-Item "rustbx.exe"
Write-Host "  -> rustbx.exe ($([math]::Round($exe.Length/1MB, 1)) MB)" -ForegroundColor Green

# Optionally create zip package
if ($Package) {
    $version = (cargo metadata --format-version=1 --no-deps 2>$null | ConvertFrom-Json).packages[0].version
    $zipName = "rustbx-v${version}-win-x64.zip"
    Compress-Archive -Path "rustbx.exe" -DestinationPath $zipName -Force
    $zip = Get-Item $zipName
    Write-Host "  => $zipName ($([math]::Round($zip.Length/1MB, 1)) MB)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Yellow
Write-Host "  Single file: rustbx.exe (model embedded)"
