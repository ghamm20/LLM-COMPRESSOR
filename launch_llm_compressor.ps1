$ErrorActionPreference = "Stop"

$Root = "D:\AI\airllm"
$VenvActivate = Join-Path $Root ".venv\Scripts\Activate.ps1"

Set-Location -LiteralPath $Root

New-Item -ItemType Directory -Force -Path (Join-Path $Root "models\huggingface\hub") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root "models\huggingface\transformers") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root "cache\torch") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root "cache\temp") | Out-Null

$env:HF_HOME = Join-Path $Root "models\huggingface"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $Root "models\huggingface\hub"
$env:TRANSFORMERS_CACHE = Join-Path $Root "models\huggingface\transformers"
$env:TORCH_HOME = Join-Path $Root "cache\torch"
$env:TEMP = Join-Path $Root "cache\temp"
$env:TMP = Join-Path $Root "cache\temp"

if (-not (Test-Path -LiteralPath $VenvActivate)) {
    throw "Virtual environment activation script not found: $VenvActivate"
}

& $VenvActivate

Clear-Host
Write-Host "================================="
Write-Host "LLM-COMPRESSOR"
Write-Host "Version: v0.1.0-yi34b-airllm-pass"
Write-Host "Repo: ghamm20/LLM-COMPRESSOR"
Write-Host "============================"
Write-Host ""

$InfoScriptPath = Join-Path $env:TEMP "llm_compressor_status.py"
$InfoScript = @'
import shutil
import sys

print(f"Python version: {sys.version.split()[0]}")

try:
    import torch
    print(f"Torch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    print(f"GPU name: {torch.cuda.get_device_name(0) if cuda_available else 'none'}")
except Exception as exc:
    print(f"Torch check failed: {exc!r}")
    print("CUDA available: unknown")
    print("GPU name: unknown")

total, used, free = shutil.disk_usage("D:\\")
print(f"D: free disk space: {free / (1024 ** 3):.2f} GiB")
'@

Set-Content -LiteralPath $InfoScriptPath -Value $InfoScript -Encoding UTF8
& python $InfoScriptPath

Write-Host ""
Write-Host "Workspace: $Root"
Write-Host "Environment is active. This PowerShell session will remain open."
