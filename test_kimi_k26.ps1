$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$Script = Join-Path $Root "scripts\test_airllm_large_model.py"
$PreflightJson = Join-Path $Root "receipts\large_model_preflight.json"
$StepModelId = "stepfun-ai/Step-3.5-Flash"
$StepSafeName = "stepfun-ai__Step-3.5-Flash"
$StepReceipt = Join-Path $Root "receipts\large_model_test_$StepSafeName.json"
$ModelId = "moonshotai/Kimi-K2.6"
$SafeName = "moonshotai__Kimi-K2.6"
$ShardDir = Join-Path $Root "models\airllm_shards\$SafeName"
$CacheDir = Join-Path $Root "cache"
$ModelsDir = Join-Path $Root "models"

if (-not (Test-Path -LiteralPath $PreflightJson)) {
    throw "Preflight receipt missing. Run .\run_large_model_preflight.ps1 before downloading weights."
}

$preflight = Get-Content -Path $PreflightJson -Raw | ConvertFrom-Json
$step = $preflight.models | Where-Object { $_.model_id -eq $StepModelId } | Select-Object -First 1
$model = $preflight.models | Where-Object { $_.model_id -eq $ModelId } | Select-Object -First 1
if (-not $model) {
    throw "No preflight result found for $ModelId"
}
if (-not $step) {
    throw "No preflight result found for $StepModelId"
}

$stepClearFailure = $step.airllm_support.compatible -eq "no"
if (-not (Test-Path -LiteralPath $StepReceipt) -and -not $stepClearFailure) {
    throw "Refusing Kimi test: Step-3.5-Flash has not passed or produced a clear compatibility failure yet."
}
if ($model.airllm_support.compatible -ne "yes") {
    throw "Refusing download: AirLLM compatibility is $($model.airllm_support.compatible). Reason: $($model.airllm_support.reason)"
}
if ($model.recommendation_before_download -notlike "Preflight passed*") {
    throw "Refusing download: $($model.recommendation_before_download)"
}

New-Item -ItemType Directory -Force -Path `
    (Join-Path $CacheDir "pip"), `
    (Join-Path $CacheDir "torch"), `
    (Join-Path $CacheDir "temp"), `
    (Join-Path $CacheDir "huggingface\datasets"), `
    (Join-Path $ModelsDir "huggingface\hub"), `
    (Join-Path $ModelsDir "huggingface\transformers"), `
    $ShardDir, `
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

& $VenvPython $Script --model-id $ModelId --shard-dir $ShardDir
exit $LASTEXITCODE
