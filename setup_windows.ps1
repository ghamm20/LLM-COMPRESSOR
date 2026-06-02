$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = Join-Path $Root "repo"
$PackageDir = Join-Path $RepoDir "air_llm"
$VenvDir = Join-Path $Root ".venv"
$CacheDir = Join-Path $Root "cache"
$ModelsDir = Join-Path $Root "models"
$ReceiptsDir = Join-Path $Root "receipts"
$SetupLog = Join-Path $ReceiptsDir "setup_log.txt"
$SetupSummary = Join-Path $ReceiptsDir "setup_summary.json"

function Get-DriveFreeGB {
    param([string]$DriveName = "D")
    $drive = Get-PSDrive -Name $DriveName
    return [math]::Round($drive.Free / 1GB, 2)
}

function Invoke-Step {
    param(
        [string]$Label,
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host "=== $Label ==="
    & $Command
    if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
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

function Get-TorchInstallPlan {
    param([string]$CudaVersion)

    if ([string]::IsNullOrWhiteSpace($CudaVersion)) {
        return [pscustomobject]@{
            Label = "cpu"
            Url = "https://download.pytorch.org/whl/cpu"
            Reason = "No CUDA version reported by nvidia-smi."
        }
    }

    $version = [version]$CudaVersion
    if ($version -ge [version]"12.8") {
        return [pscustomobject]@{ Label = "cu128"; Url = "https://download.pytorch.org/whl/cu128"; Reason = "Detected CUDA $CudaVersion; choosing the newest supported wheel at or below the detected runtime." }
    }
    if ($version -ge [version]"12.6") {
        return [pscustomobject]@{ Label = "cu126"; Url = "https://download.pytorch.org/whl/cu126"; Reason = "Detected CUDA $CudaVersion." }
    }
    if ($version -ge [version]"12.4") {
        return [pscustomobject]@{ Label = "cu124"; Url = "https://download.pytorch.org/whl/cu124"; Reason = "Detected CUDA $CudaVersion." }
    }
    if ($version -ge [version]"12.1") {
        return [pscustomobject]@{ Label = "cu121"; Url = "https://download.pytorch.org/whl/cu121"; Reason = "Detected CUDA $CudaVersion." }
    }
    if ($version -ge [version]"11.8") {
        return [pscustomobject]@{ Label = "cu118"; Url = "https://download.pytorch.org/whl/cu118"; Reason = "Detected CUDA $CudaVersion." }
    }

    return [pscustomobject]@{
        Label = "cpu"
        Url = "https://download.pytorch.org/whl/cpu"
        Reason = "Detected CUDA $CudaVersion, below the supported CUDA wheel baseline."
    }
}

function Install-Torch {
    param(
        [string]$PythonExe,
        [string]$IndexUrl,
        [string]$Label
    )

    Write-Host "Installing torch from $IndexUrl ($Label)"
    & $PythonExe -m pip install --upgrade torch --index-url $IndexUrl
    if ($LASTEXITCODE -ne 0) {
        throw "torch install failed for $Label"
    }
}

function Get-TorchProbe {
    param([string]$PythonExe)

    $probe = @'
import json
import sys

info = {"import_ok": False}
try:
    import torch
    info.update({
        "import_ok": True,
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    })
except Exception as exc:
    info["error"] = repr(exc)

print(json.dumps(info))
sys.exit(0 if info.get("import_ok") else 1)
'@

    $output = $probe | & $PythonExe -
    if ($LASTEXITCODE -ne 0) {
        throw "torch probe failed: $output"
    }

    return ($output | Select-Object -Last 1 | ConvertFrom-Json)
}

New-Item -ItemType Directory -Force -Path $ReceiptsDir | Out-Null
$driveFreeBefore = Get-DriveFreeGB
$cudaVersionDetected = $null
$gpuDetected = $null
$torchInstallLabel = $null
$torchProbe = $null
$pipCheckStatus = "not_run"
$runtimeCompatibilityPins = @(
    "optimum==1.27.0",
    "transformers==4.48.3",
    "sentencepiece==0.2.1"
)
$setupStatus = "started"

Start-Transcript -Path $SetupLog -Force | Out-Null
try {
    Write-Host "AirLLM local setup"
    Write-Host "Root: $Root"
    Write-Host "Repo: $RepoDir"
    Write-Host "Package: $PackageDir"
    Write-Host "D: free before install: $driveFreeBefore GB"

    if (-not (Test-Path -LiteralPath $RepoDir)) {
        throw "Repository directory is missing: $RepoDir"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $PackageDir "setup.py"))) {
        throw "Installable package setup.py is missing: $(Join-Path $PackageDir 'setup.py')"
    }

    Set-LocalEnvironment
    Write-Host "PIP_CACHE_DIR=$env:PIP_CACHE_DIR"
    Write-Host "TEMP=$env:TEMP"
    Write-Host "TMP=$env:TMP"
    Write-Host "HF_HOME=$env:HF_HOME"
    Write-Host "HUGGINGFACE_HUB_CACHE=$env:HUGGINGFACE_HUB_CACHE"
    Write-Host "TORCH_HOME=$env:TORCH_HOME"

    $venvPython = Join-Path $VenvDir "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $venvPython)) {
        Invoke-Step "Create local virtual environment" {
            & python -m venv $VenvDir
        }
    } else {
        Write-Host "Existing local virtual environment found: $VenvDir"
    }

    $activate = Join-Path $VenvDir "Scripts\Activate.ps1"
    if (Test-Path -LiteralPath $activate) {
        . $activate
        Write-Host "Activated venv: $VenvDir"
    } else {
        throw "Activation script missing: $activate"
    }

    Invoke-Step "Python version" {
        & $venvPython --version
    }

    Invoke-Step "Upgrade pip, setuptools, wheel within runtime constraints" {
        & $venvPython -m pip install --upgrade pip "setuptools<82" wheel
    }

    $gpuDetected = Get-GpuSummary
    $cudaVersionDetected = Get-NvidiaCudaVersion
    Write-Host "GPU detected: $gpuDetected"
    Write-Host "CUDA version detected by nvidia-smi: $cudaVersionDetected"

    $torchPlan = Get-TorchInstallPlan -CudaVersion $cudaVersionDetected
    Write-Host "Torch install plan: $($torchPlan.Label) - $($torchPlan.Reason)"

    try {
        Install-Torch -PythonExe $venvPython -IndexUrl $torchPlan.Url -Label $torchPlan.Label
        $torchProbe = Get-TorchProbe -PythonExe $venvPython
        $torchInstallLabel = $torchPlan.Label

        if ($torchPlan.Label -ne "cpu" -and -not $torchProbe.cuda_available) {
            throw "CUDA torch wheel installed, but torch.cuda.is_available() returned false."
        }
    } catch {
        Write-Warning "CUDA torch install/validation failed: $($_.Exception.Message)"
        Write-Warning "Falling back to CPU torch wheel."
        & $venvPython -m pip uninstall -y torch
        Install-Torch -PythonExe $venvPython -IndexUrl "https://download.pytorch.org/whl/cpu" -Label "cpu"
        $torchProbe = Get-TorchProbe -PythonExe $venvPython
        $torchInstallLabel = "cpu"
    }

    Write-Host "Torch probe:"
    $torchProbe | ConvertTo-Json -Depth 6

    Write-Host ""
    Write-Host "Repo top-level requirements.txt is present, but not installed by default."
    Write-Host "Reason: it appears to be training extras with old Windows/Python-sensitive pins."
    Get-Content -Path (Join-Path $RepoDir "requirements.txt") | ForEach-Object { Write-Host "  $_" }

    Invoke-Step "Install AirLLM package in editable mode" {
        & $venvPython -m pip install --editable $PackageDir
    }

    Write-Host ""
    Write-Host "Applying runtime compatibility pins required by AirLLM imports:"
    $runtimeCompatibilityPins | ForEach-Object { Write-Host "  $_" }
    Invoke-Step "Install AirLLM runtime compatibility pins" {
        & $venvPython -m pip install --upgrade @runtimeCompatibilityPins
    }

    Invoke-Step "Reconcile setuptools to PyTorch constraint" {
        & $venvPython -m pip install --upgrade "setuptools<82"
    }

    Invoke-Step "Validate dependency consistency with pip check" {
        & $venvPython -m pip check
    }
    $pipCheckStatus = "pass"

    $setupStatus = "completed"
} finally {
    $driveFreeAfter = Get-DriveFreeGB
    $summary = [ordered]@{
        status = $setupStatus
        root = $Root
        repo = $RepoDir
        venv = $VenvDir
        setup_log = $SetupLog
        d_free_gb_before_install = $driveFreeBefore
        d_free_gb_after_install = $driveFreeAfter
        gpu_detected = $gpuDetected
        cuda_version_detected = $cudaVersionDetected
        torch_install_label = $torchInstallLabel
        torch_probe = $torchProbe
        pip_check_status = $pipCheckStatus
        runtime_compatibility_pins = $runtimeCompatibilityPins
        top_level_requirements_installed = $false
        top_level_requirements_note = "Skipped by default; appears to be training extras and may conflict on Windows/Python 3.12."
    }
    $summary | ConvertTo-Json -Depth 8 | Set-Content -Path $SetupSummary -Encoding UTF8
    Write-Host "D: free after install: $driveFreeAfter GB"
    Write-Host "Setup summary: $SetupSummary"
    Stop-Transcript | Out-Null
}
