$ErrorActionPreference = "Stop"

$Root = "D:\AI\airllm"
$VenvActivate = Join-Path $Root ".venv\Scripts\Activate.ps1"
$RerunScript = Join-Path $Root "rerun_yi34b_chat_minimal.ps1"

Set-Location -LiteralPath $Root

New-Item -ItemType Directory -Force -Path `
    (Join-Path $Root "models\huggingface\hub"), `
    (Join-Path $Root "models\huggingface\transformers"), `
    (Join-Path $Root "cache\torch"), `
    (Join-Path $Root "cache\temp"), `
    (Join-Path $Root "receipts") | Out-Null

$env:HF_HOME = Join-Path $Root "models\huggingface"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $Root "models\huggingface\hub"
$env:TRANSFORMERS_CACHE = Join-Path $Root "models\huggingface\transformers"
$env:TORCH_HOME = Join-Path $Root "cache\torch"
$env:TEMP = Join-Path $Root "cache\temp"
$env:TMP = Join-Path $Root "cache\temp"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

if (-not (Test-Path -LiteralPath $VenvActivate)) {
    throw "Virtual environment activation script not found: $VenvActivate"
}
if (-not (Test-Path -LiteralPath $RerunScript)) {
    throw "Yi-34B-Chat 1-token rerun script not found: $RerunScript"
}

. $VenvActivate

Write-Host "================================="
Write-Host "LLM-COMPRESSOR Yi-34B-Chat 1-token"
Write-Host "Version: v0.1.1-yi-family-compat"
Write-Host "Model: 01-ai/Yi-34B-Chat"
Write-Host "Mode: local snapshot and existing AirLLM shards only"
Write-Host "================================="
Write-Host ""

& $RerunScript
$ExitCode = $LASTEXITCODE

Write-Host ""
Write-Host "Yi-34B-Chat 1-token test finished with exit code: $ExitCode"
Write-Host ""
Write-Host "Receipt paths:"
Write-Host "  D:\AI\airllm\receipts\yi34b_chat_minimal_rerun.txt"
Write-Host "  D:\AI\airllm\receipts\yi34b_chat_minimal_rerun.json"
Write-Host "  D:\AI\airllm\receipts\yi34b_chat_minimal_stdout.log"
Write-Host "  D:\AI\airllm\receipts\yi34b_chat_minimal_stderr.log"
Write-Host ""

exit $ExitCode
