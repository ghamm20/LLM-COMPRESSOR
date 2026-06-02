$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $Root ".venv"
$CacheDir = Join-Path $Root "cache"
$ModelsDir = Join-Path $Root "models"
$ReceiptsDir = Join-Path $Root "receipts"
$VerifyLog = Join-Path $ReceiptsDir "verify_log.txt"
$PipFreeze = Join-Path $ReceiptsDir "pip_freeze.txt"
$VerifySummary = Join-Path $ReceiptsDir "verify_summary.json"
$SmokeScript = Join-Path $Root "run_smoke.py"

function Get-DriveFreeGB {
    param([string]$DriveName = "D")
    $drive = Get-PSDrive -Name $DriveName
    return [math]::Round($drive.Free / 1GB, 2)
}

function Set-LocalEnvironment {
    New-Item -ItemType Directory -Force -Path $CacheDir, $ModelsDir, $ReceiptsDir | Out-Null
    New-Item -ItemType Directory -Force -Path `
        (Join-Path $CacheDir "pip"), `
        (Join-Path $CacheDir "torch"), `
        (Join-Path $CacheDir "temp"), `
        (Join-Path $CacheDir "huggingface\datasets"), `
        (Join-Path $ModelsDir "huggingface\hub"), `
        (Join-Path $ModelsDir "huggingface\transformers") | Out-Null

    $env:TEMP = Join-Path $CacheDir "temp"
    $env:TMP = Join-Path $CacheDir "temp"
    $env:PIP_CACHE_DIR = Join-Path $CacheDir "pip"
    $env:TORCH_HOME = Join-Path $CacheDir "torch"
    $env:XDG_CACHE_HOME = $CacheDir
    $env:HF_DATASETS_CACHE = Join-Path $CacheDir "huggingface\datasets"

    $env:HF_HOME = Join-Path $ModelsDir "huggingface"
    $env:HUGGINGFACE_HUB_CACHE = Join-Path $ModelsDir "huggingface\hub"
    $env:TRANSFORMERS_CACHE = Join-Path $ModelsDir "huggingface\transformers"
    $env:AIRLLM_LOCAL_MODELS = $ModelsDir
}

function Get-NvidiaCudaVersion {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        return $null
    }

    $smiText = (& nvidia-smi 2>$null | Out-String)
    if ($smiText -match "CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)") {
        return $Matches[1]
    }

    return $null
}

function Get-GpuSummary {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        return "nvidia-smi not found"
    }

    $gpu = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $gpu) {
        return ($gpu -join "; ")
    }

    return "nvidia-smi available, GPU query failed"
}

New-Item -ItemType Directory -Force -Path $ReceiptsDir | Out-Null
$driveFreeAtVerify = Get-DriveFreeGB
$pythonVersion = $null
$torchInfo = $null
$airllmImportStatus = "fail"
$smokeStatus = "fail"
$cudaVersionDetected = $null
$gpuDetected = $null

Start-Transcript -Path $VerifyLog -Force | Out-Null
try {
    Write-Host "AirLLM local verification"
    Write-Host "Root: $Root"
    Write-Host "D: free at verification: $driveFreeAtVerify GB"

    Set-LocalEnvironment

    $venvPython = Join-Path $VenvDir "Scripts\python.exe"
    $activate = Join-Path $VenvDir "Scripts\Activate.ps1"
    if (-not (Test-Path -LiteralPath $venvPython)) {
        throw "Venv python missing: $venvPython"
    }
    if (-not (Test-Path -LiteralPath $activate)) {
        throw "Activation script missing: $activate"
    }

    . $activate
    Write-Host "Activated venv: $VenvDir"

    Write-Host ""
    Write-Host "=== Python version ==="
    $pythonVersion = (& $venvPython --version)
    Write-Host $pythonVersion
    & $venvPython -c "import sys; print(sys.executable)"

    Write-Host ""
    Write-Host "=== pip freeze ==="
    & $venvPython -m pip freeze | Set-Content -Path $PipFreeze -Encoding UTF8
    Write-Host "Wrote $PipFreeze"

    $gpuDetected = Get-GpuSummary
    $cudaVersionDetected = Get-NvidiaCudaVersion
    Write-Host "GPU detected: $gpuDetected"
    Write-Host "CUDA version detected by nvidia-smi: $cudaVersionDetected"

    Write-Host ""
    Write-Host "=== Torch and AirLLM import probe ==="
    $probe = @'
import json
import sys

info = {
    "torch": {"import_ok": False},
    "airllm": {"import_ok": False},
}

try:
    import torch
    info["torch"] = {
        "import_ok": True,
        "version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
except Exception as exc:
    info["torch"]["error"] = repr(exc)

try:
    import airllm
    from airllm import AutoModel, AirLLMLlama2
    info["airllm"] = {
        "import_ok": True,
        "module_file": getattr(airllm, "__file__", None),
        "automodel_callable": callable(getattr(AutoModel, "from_pretrained", None)),
        "llama_class": AirLLMLlama2.__name__,
    }
except Exception as exc:
    info["airllm"]["error"] = repr(exc)

print(json.dumps(info, indent=2))
sys.exit(0 if info["torch"]["import_ok"] and info["airllm"]["import_ok"] else 1)
'@
    $probeOutput = $probe | & $venvPython -
    $probeExit = $LASTEXITCODE
    $probeOutput | ForEach-Object { Write-Host $_ }

    $jsonStart = ($probeOutput | Select-String -Pattern "^\{" | Select-Object -First 1).LineNumber
    if ($jsonStart) {
        $torchInfo = (($probeOutput | Select-Object -Skip ($jsonStart - 1)) -join [Environment]::NewLine | ConvertFrom-Json)
        if ($torchInfo.airllm.import_ok) {
            $airllmImportStatus = "pass"
        }
    }

    if ($probeExit -ne 0) {
        throw "Torch/AirLLM probe failed."
    }

    Write-Host ""
    Write-Host "=== Smoke test ==="
    & $venvPython $SmokeScript
    if ($LASTEXITCODE -eq 0) {
        $smokeStatus = "pass"
    } else {
        throw "Smoke test failed with exit code $LASTEXITCODE"
    }
} finally {
    $summary = [ordered]@{
        root = $Root
        verify_log = $VerifyLog
        pip_freeze = $PipFreeze
        d_free_gb_at_verification = $driveFreeAtVerify
        python_version = $pythonVersion
        gpu_detected = $gpuDetected
        cuda_version_detected = $cudaVersionDetected
        torch = $torchInfo.torch
        airllm = $torchInfo.airllm
        airllm_import_status = $airllmImportStatus
        smoke_status = $smokeStatus
    }
    $summary | ConvertTo-Json -Depth 8 | Set-Content -Path $VerifySummary -Encoding UTF8
    Write-Host "Verification summary: $VerifySummary"
    Stop-Transcript | Out-Null
}
