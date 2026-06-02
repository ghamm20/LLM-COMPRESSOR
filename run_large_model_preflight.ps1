$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$Script = Join-Path $Root "scripts\preflight_large_models.py"
$CacheDir = Join-Path $Root "cache"
$ModelsDir = Join-Path $Root "models"

New-Item -ItemType Directory -Force -Path `
    (Join-Path $CacheDir "pip"), `
    (Join-Path $CacheDir "torch"), `
    (Join-Path $CacheDir "temp"), `
    (Join-Path $CacheDir "huggingface\datasets"), `
    (Join-Path $ModelsDir "huggingface\hub"), `
    (Join-Path $ModelsDir "huggingface\transformers"), `
    (Join-Path $Root "receipts") | Out-Null

$env:PIP_CACHE_DIR = Join-Path $CacheDir "pip"
$env:TEMP = Join-Path $CacheDir "temp"
$env:TMP = Join-Path $CacheDir "temp"
$env:TORCH_HOME = Join-Path $CacheDir "torch"
$env:XDG_CACHE_HOME = $CacheDir
$env:HF_DATASETS_CACHE = Join-Path $CacheDir "huggingface\datasets"
$env:HF_HOME = Join-Path $ModelsDir "huggingface"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $ModelsDir "huggingface\hub"
$env:TRANSFORMERS_CACHE = Join-Path $ModelsDir "huggingface\transformers"
$env:AIRLLM_LOCAL_MODELS = $ModelsDir

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Missing venv python: $VenvPython"
}
if (-not (Test-Path -LiteralPath $Script)) {
    throw "Missing preflight script: $Script"
}

& $VenvPython $Script
exit $LASTEXITCODE
