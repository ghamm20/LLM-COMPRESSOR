$ErrorActionPreference = "Stop"

$Root = "D:\AI\airllm"
$Template = Join-Path $Root "rerun_yi34b_chat_minimal.ps1"
$Generated = Join-Path $Root "cache\temp\yi34b_chat_4token_generated.ps1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

if (-not (Test-Path -LiteralPath $Template)) {
    throw "Missing local-only template script: $Template"
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Generated) | Out-Null

$script = Get-Content -LiteralPath $Template -Raw
$replacements = [ordered]@{
    "yi34b_chat_minimal_runner.py" = "yi34b_chat_4token_runner.py"
    "yi34b_chat_minimal_stdout.log" = "yi34b_chat_4token_stdout.log"
    "yi34b_chat_minimal_stderr.log" = "yi34b_chat_4token_stderr.log"
    "yi34b_chat_minimal_rerun.txt" = "yi34b_chat_4token_rerun.txt"
    "yi34b_chat_minimal_rerun.json" = "yi34b_chat_4token_rerun.json"
    "MAX_NEW_TOKENS = 1" = "MAX_NEW_TOKENS = 4"
    "Yi-34B-Chat minimal local-only rerun" = "Yi-34B-Chat 4-token local-only rerun"
    "Yi-34B-Chat minimal rerun using local snapshot" = "Yi-34B-Chat 4-token rerun using local snapshot"
}

foreach ($pair in $replacements.GetEnumerator()) {
    $script = $script.Replace($pair.Key, $pair.Value)
}

Set-Content -LiteralPath $Generated -Value $script -Encoding UTF8
& powershell -NoProfile -ExecutionPolicy Bypass -File $Generated
exit $LASTEXITCODE
