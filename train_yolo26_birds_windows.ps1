param(
    [string]$ProjectRoot = "",
    [string]$WorkDir = "",
    [string]$DatasetDir = "",
    [string]$RoboflowDownloadUrl = "",
    [string]$DatasetZip = "",
    [string]$PythonBin = "",
    [string]$TorchIndexUrl = "",
    [int]$InstallTorch = 1,
    [string]$Model = "yolo26x.pt",
    [int]$ImgSize = 640,
    [int]$Epochs = 300,
    [int]$Batch = 4,
    [int]$Nbs = 32,
    [int]$Patience = 75,
    [int]$Workers = 8,
    [string]$Device = "0",
    [string]$Cache = "disk",
    [string]$RunName = "",
    [int]$RunSmokeTest = 1,
    [int]$RunFullTrain = 1,
    [int]$RunPolish = 0,
    [int]$Resume = 0,
    [string]$ResumeLast = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ============================================================
# train_yolo26_birds_windows.ps1
#
# PowerShellowy odpowiednik skryptu Linux .sh.
#
# Scenariusz:
# - dataset ptaków z Roboflow,
# - YOLO26,
# - yolo26x.pt,
# - imgsz 640,
# - 300 epok,
# - smoke-test 1 epoki,
# - pełny trening,
# - opcjonalny resume i polishing run.
#
# Nie zapisuj linku Roboflow z kluczem API w repozytorium.
# ============================================================

function Get-Value {
    param(
        [string]$Current,
        [string]$EnvName,
        [string]$Default
    )

    if (-not [string]::IsNullOrWhiteSpace($Current)) {
        return $Current
    }

    $envValue = [Environment]::GetEnvironmentVariable($EnvName)
    if (-not [string]::IsNullOrWhiteSpace($envValue)) {
        return $envValue
    }

    return $Default
}

function Info {
    param([string]$Message)
    Write-Host ""
    Write-Host "============================================================"
    Write-Host $Message
    Write-Host "============================================================"
}

function Warn {
    param([string]$Message)
    Write-Host ""
    Write-Host "[UWAGA] $Message" -ForegroundColor Yellow
}

function Fail {
    param([string]$Message)
    throw "[BŁĄD] $Message"
}

function Check-LastExitCode {
    param([string]$StepName)

    if ($LASTEXITCODE -ne 0) {
        Fail "$StepName zakończył się błędem. LASTEXITCODE=$LASTEXITCODE"
    }
}

$ProjectRoot = Get-Value $ProjectRoot "PROJECT_ROOT" (Join-Path $HOME "yolo_train")
$WorkDir = Get-Value $WorkDir "WORKDIR" (Join-Path $ProjectRoot "birds_yolo26")
$DatasetDir = Get-Value $DatasetDir "DATASET_DIR" (Join-Path $WorkDir "dataset")
$RoboflowDownloadUrl = Get-Value $RoboflowDownloadUrl "ROBOFLOW_DOWNLOAD_URL" ""
$DatasetZip = Get-Value $DatasetZip "DATASET_ZIP" ""
$PythonBin = Get-Value $PythonBin "PYTHON_BIN" ""
$TorchIndexUrl = Get-Value $TorchIndexUrl "TORCH_INDEX_URL" "https://download.pytorch.org/whl/cu126"

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "birds_yolo26x_640_b$($Batch)_nbs$($Nbs)_cm10_e$($Epochs)"
}

if ([string]::IsNullOrWhiteSpace($ResumeLast)) {
    $ResumeLast = Join-Path $WorkDir "runs\detect\$RunName\weights\last.pt"
}

New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
$LogFile = Join-Path $WorkDir ("train_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

try {
    Start-Transcript -Path $LogFile -Append | Out-Null
}
catch {
    Warn "Nie udało się uruchomić Start-Transcript. Skrypt będzie działał, ale log może być niepełny."
}

try {
    Info "Start skryptu Windows PowerShell"
    Write-Host "WORKDIR:     $WorkDir"
    Write-Host "DATASET_DIR: $DatasetDir"
    Write-Host "LOG_FILE:    $LogFile"
    Write-Host "MODEL:       $Model"
    Write-Host "IMG_SIZE:    $ImgSize"
    Write-Host "EPOCHS:      $Epochs"
    Write-Host "BATCH:       $Batch"
    Write-Host "NBS:         $Nbs"
    Write-Host "DEVICE:      $Device"
    Write-Host "RUN_NAME:    $RunName"

    Info "Wybór Pythona"

    $PythonCmd = ""
    $PythonBaseArgs = @()

    if (-not [string]::IsNullOrWhiteSpace($PythonBin)) {
        $PythonCmd = $PythonBin
        $PythonBaseArgs = @()
    }
    elseif (Get-Command py -ErrorAction SilentlyContinue) {
        $PythonCmd = "py"
        $PythonBaseArgs = @("-3")
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $PythonCmd = "python"
        $PythonBaseArgs = @()
    }
    else {
        Fail "Nie znaleziono Pythona. Zainstaluj Python 3 i zaznacz 'Add Python to PATH'."
    }

    Write-Host "Python command: $PythonCmd $($PythonBaseArgs -join ' ')"
    & $PythonCmd @PythonBaseArgs --version
    Check-LastExitCode "Sprawdzenie Pythona"

    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        Info "Informacje o GPU"
        nvidia-smi
    }
    else {
        Warn "Nie znaleziono nvidia-smi. Jeśli komputer ma NVIDIA GPU, sprawdź sterowniki."
    }

    Info "Tworzenie / użycie venv"

    $VenvDir = Join-Path $WorkDir ".venv"
    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"
    $YoloExe = Join-Path $VenvDir "Scripts\yolo.exe"

    if (-not (Test-Path $VenvPython)) {
        & $PythonCmd @PythonBaseArgs -m venv $VenvDir
        Check-LastExitCode "Tworzenie venv"
    }

    & $VenvPython -m pip install --upgrade pip setuptools wheel
    Check-LastExitCode "Aktualizacja pip/setuptools/wheel"

    if ($InstallTorch -eq 1) {
        Info "Instalacja / aktualizacja PyTorch"
        Write-Host "TorchIndexUrl: $TorchIndexUrl"
        & $VenvPython -m pip install --upgrade torch torchvision torchaudio --index-url $TorchIndexUrl
        Check-LastExitCode "Instalacja PyTorch"
    }
    else {
        Warn "Pominięto instalację PyTorch, bo -InstallTorch 0"
    }

    Info "Instalacja / aktualizacja Ultralytics"
    & $VenvPython -m pip install --upgrade ultralytics pyyaml
    Check-LastExitCode "Instalacja Ultralytics"

    if (-not (Test-Path $YoloExe)) {
        Fail "Nie znaleziono yolo.exe w venv: $YoloExe"
    }

    Info "Kontrola Python / Torch / CUDA"

    $TorchCheckPy = Join-Path $WorkDir "_check_torch.py"
    @'
import sys
print("Python:", sys.version)

try:
    import torch
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version in torch:", torch.version.cuda)
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device 0:", torch.cuda.get_device_name(0))
except Exception as e:
    print("Torch check error:", repr(e))
    raise
'@ | Set-Content -Encoding UTF8 -Path $TorchCheckPy

    & $VenvPython $TorchCheckPy
    Check-LastExitCode "Kontrola Torch/CUDA"

    function Invoke-Yolo {
        param([string[]]$YoloArgs)

        Write-Host ""
        Write-Host "YOLO command:"
        Write-Host "$YoloExe $($YoloArgs -join ' ')"
        Write-Host ""

        & $YoloExe @YoloArgs
        Check-LastExitCode "Komenda YOLO"
    }

    if ($Resume -eq 1) {
        Info "Tryb RESUME"

        if (-not (Test-Path $ResumeLast)) {
            Fail "Nie znaleziono checkpointu last.pt: $ResumeLast"
        }

        Invoke-Yolo @(
            "task=detect",
            "mode=train",
            "model=$ResumeLast",
            "resume=True"
        )

        Info "Resume zakończony"
        Write-Host "Log: $LogFile"
        return
    }

    Info "Przygotowanie datasetu"

    $DataYaml = Join-Path $DatasetDir "data.yaml"

    if (Test-Path $DataYaml) {
        Write-Host "Znaleziono istniejący dataset: $DataYaml"
    }
    else {
        New-Item -ItemType Directory -Force -Path $DatasetDir | Out-Null
        $RawDir = Join-Path $WorkDir "dataset_raw"

        if (Test-Path $RawDir) {
            Remove-Item -Recurse -Force $RawDir
        }

        New-Item -ItemType Directory -Force -Path $RawDir | Out-Null

        if (-not [string]::IsNullOrWhiteSpace($DatasetZip)) {
            if (-not (Test-Path $DatasetZip)) {
                Fail "DatasetZip wskazuje na nieistniejący plik: $DatasetZip"
            }

            Info "Rozpakowanie datasetu ZIP"
            Expand-Archive -Force -Path $DatasetZip -DestinationPath $RawDir
        }
        elseif (-not [string]::IsNullOrWhiteSpace($RoboflowDownloadUrl)) {
            Info "Pobieranie datasetu z Roboflow"
            Warn "Link Roboflow może zawierać klucz API. Nie zapisuj go w publicznym repo."

            $ZipOut = Join-Path $WorkDir "roboflow_dataset.zip"

            Invoke-WebRequest -Uri $RoboflowDownloadUrl -OutFile $ZipOut
            Expand-Archive -Force -Path $ZipOut -DestinationPath $RawDir
        }
        else {
            Fail "Nie znaleziono data.yaml i nie podano danych. Użyj -RoboflowDownloadUrl, -DatasetZip albo -DatasetDir."
        }

        $FoundYaml = Get-ChildItem -Path $RawDir -Filter "data.yaml" -Recurse | Select-Object -First 1

        if ($null -eq $FoundYaml) {
            Fail "Nie znaleziono data.yaml po rozpakowaniu datasetu."
        }

        $RawRoot = $FoundYaml.Directory.FullName

        if (Test-Path $DatasetDir) {
            Remove-Item -Recurse -Force $DatasetDir
        }

        New-Item -ItemType Directory -Force -Path $DatasetDir | Out-Null
        Copy-Item -Path (Join-Path $RawRoot "*") -Destination $DatasetDir -Recurse -Force

        $DataYaml = Join-Path $DatasetDir "data.yaml"
    }

    if (-not (Test-Path $DataYaml)) {
        Fail "Brak data.yaml: $DataYaml"
    }

    Write-Host "DATA_YAML: $DataYaml"

    Info "Kontrola data.yaml i struktury datasetu"

    $CheckDatasetPy = Join-Path $WorkDir "_check_dataset.py"
    @'
from pathlib import Path
import sys
import yaml

data_yaml = Path(sys.argv[1]).resolve()
root = data_yaml.parent

with open(data_yaml, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

print("data.yaml:", data_yaml)
print("root:", root)

required = ["train", "val", "names"]
missing = [k for k in required if k not in data]
if missing:
    raise SystemExit(f"Brakuje wymaganych pól w data.yaml: {missing}")

names = data["names"]
print("Liczba klas:", len(names) if isinstance(names, (list, dict)) else "nieznana")
print("Klasy:", names)

def resolve_path(value):
    p = Path(str(value))
    if not p.is_absolute():
        p = (root / p).resolve()
    return p

for key in ["train", "val", "test"]:
    if key not in data:
        continue
    p = resolve_path(data[key])
    print(f"{key}: {p}")
    if not p.exists():
        print(f"[UWAGA] Ścieżka {key} nie istnieje: {p}")

for folder_name in ["train", "valid", "val", "test"]:
    candidates = [
        root / folder_name / "images",
        root / folder_name,
    ]
    for c in candidates:
        if c.exists():
            imgs = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                imgs.extend(c.glob(ext))
            if imgs:
                print(f"Obrazy w {c}: {len(imgs)}")
'@ | Set-Content -Encoding UTF8 -Path $CheckDatasetPy

    & $VenvPython $CheckDatasetPy $DataYaml
    Check-LastExitCode "Kontrola datasetu"

    $RunsProject = Join-Path $WorkDir "runs\detect"

    if ($RunSmokeTest -eq 1) {
        Info "Smoke-test: mini-trening 1 epoki na yolo26n.pt"

        Invoke-Yolo @(
            "task=detect",
            "mode=train",
            "model=yolo26n.pt",
            "data=$DataYaml",
            "device=$Device",
            "imgsz=$ImgSize",
            "batch=2",
            "epochs=1",
            "workers=2",
            "cache=False",
            "plots=False",
            "project=$RunsProject",
            "name=smoke_test_yolo26n_e1"
        )

        Info "Smoke-test zakończony"
    }
    else {
        Warn "Pominięto smoke-test, bo -RunSmokeTest 0"
    }

    if ($RunFullTrain -eq 1) {
        Info "Pełny trening YOLO26"

        Invoke-Yolo @(
            "task=detect",
            "mode=train",
            "model=$Model",
            "data=$DataYaml",
            "device=$Device",
            "imgsz=$ImgSize",
            "batch=$Batch",
            "nbs=$Nbs",
            "epochs=$Epochs",
            "patience=$Patience",
            "optimizer=auto",
            "cos_lr=True",
            "mosaic=1.0",
            "close_mosaic=10",
            "workers=$Workers",
            "cache=$Cache",
            "save_period=10",
            "plots=True",
            "seed=42",
            "project=$RunsProject",
            "name=$RunName"
        )

        Info "Pełny trening zakończony"

        $BestModel = Join-Path $RunsProject "$RunName\weights\best.pt"
        $LastModel = Join-Path $RunsProject "$RunName\weights\last.pt"

        Write-Host "BEST_MODEL: $BestModel"
        Write-Host "LAST_MODEL: $LastModel"
    }
    else {
        Warn "Pominięto pełny trening, bo -RunFullTrain 0"
    }

    if ($RunPolish -eq 1) {
        Info "Polishing run: delikatny fine-tuning bez mosaic"

        $BestModel = Join-Path $RunsProject "$RunName\weights\best.pt"

        if (-not (Test-Path $BestModel)) {
            Fail "Nie znaleziono best.pt do polishing run: $BestModel"
        }

        Invoke-Yolo @(
            "task=detect",
            "mode=train",
            "model=$BestModel",
            "data=$DataYaml",
            "device=$Device",
            "imgsz=$ImgSize",
            "batch=$Batch",
            "nbs=$Nbs",
            "epochs=60",
            "patience=20",
            "optimizer=auto",
            "lr0=0.0005",
            "lrf=0.1",
            "cos_lr=True",
            "mosaic=0",
            "close_mosaic=0",
            "workers=$Workers",
            "cache=$Cache",
            "plots=True",
            "seed=42",
            "project=$RunsProject",
            "name=$($RunName)_polish_lr0005_mos0_e60"
        )

        Info "Polishing run zakończony"
    }
    else {
        Warn "Polishing run pominięty. To normalne. Możesz go uruchomić później przez -RunPolish 1."
    }

    Info "Gotowe"

    Write-Host "Wyniki:"
    Write-Host "  $RunsProject"
    Write-Host ""
    Write-Host "Główny model po treningu:"
    Write-Host "  $(Join-Path $RunsProject "$RunName\weights\best.pt")"
    Write-Host ""
    Write-Host "Log:"
    Write-Host "  $LogFile"
    Write-Host ""
    Write-Host "Jeśli trening przerwał się w trakcie, kontynuuj tak:"
    Write-Host "  .\train_yolo26_birds_windows.ps1 -Resume 1"
    Write-Host ""
    Write-Host "Jeśli zabraknie pamięci GPU, spróbuj:"
    Write-Host "  .\train_yolo26_birds_windows.ps1 -Batch 2"
    Write-Host "albo:"
    Write-Host "  .\train_yolo26_birds_windows.ps1 -Model yolo26l.pt -Batch 4"
}
finally {
    try {
        Stop-Transcript | Out-Null
    }
    catch {
        # Bez znaczenia, jeśli transcript nie był aktywny.
    }
}
