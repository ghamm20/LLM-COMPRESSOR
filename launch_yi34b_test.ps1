$ErrorActionPreference = "Stop"

$Root = "D:\AI\airllm"
$VenvActivate = Join-Path $Root ".venv\Scripts\Activate.ps1"
$RerunScript = Join-Path $Root "rerun_yi34b_generation.ps1"

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
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

if (-not (Test-Path -LiteralPath $VenvActivate)) {
    throw "Virtual environment activation script not found: $VenvActivate"
}
if (-not (Test-Path -LiteralPath $RerunScript)) {
    throw "Yi-34B rerun script not found: $RerunScript"
}

& $VenvActivate

Clear-Host
Write-Host "================================="
Write-Host "LLM-COMPRESSOR Yi-34B Test"
Write-Host "Version: v0.1.1-yi-family-compat"
Write-Host "Repo: ghamm20/LLM-COMPRESSOR"
Write-Host "============================"
Write-Host ""
Write-Host "Running: $RerunScript"
Write-Host ""

& $RerunScript
$ExitCode = $LASTEXITCODE

Write-Host ""
Write-Host "Yi-34B test finished with exit code: $ExitCode"
Write-Host ""
Write-Host "Receipt paths:"
Write-Host "  D:\AI\airllm\receipts\yi34b_generation_rerun.txt"
Write-Host "  D:\AI\airllm\receipts\yi34b_generation_rerun.json"
Write-Host "  D:\AI\airllm\receipts\yi34b_generation_rerun_stdout.log"
Write-Host "  D:\AI\airllm\receipts\yi34b_generation_rerun_stderr.log"
Write-Host ""
Write-Host "This PowerShell session will remain open."
