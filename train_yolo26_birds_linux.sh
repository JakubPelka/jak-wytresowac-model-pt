#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# train_yolo26_birds_linux.sh
# Ultra-prosty skrypt do treningu modelu YOLO26 na Linuxie.
#
# Docelowy scenariusz:
# - dataset ptaków z Roboflow, np. ok. 9000 zannotowanych zdjęć,
# - trening detekcji obiektów YOLO,
# - start z wag pretrained: yolo26x.pt,
# - obraz 640 px,
# - 300 epok,
# - najpierw smoke-test 1 epoki, potem pełny trening.
#
# WAŻNE:
# Nie wklejaj linku Roboflow z kluczem API do README ani do publicznego repo.
# Uruchamiaj go jako zmienną środowiskową w terminalu.
# ============================================================

# ----------------------------
# 1. USTAWIENIA DOMYŚLNE
# Możesz je nadpisać przy uruchomieniu skryptu.
# ----------------------------

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/yolo_train}"
WORKDIR="${WORKDIR:-$PROJECT_ROOT/birds_yolo26}"
DATASET_DIR="${DATASET_DIR:-$WORKDIR/dataset}"

# Tryb pobierania danych:
# A) ROBOFLOW_DOWNLOAD_URL="https://..." ./train_yolo26_birds_linux.sh
# B) DATASET_ZIP="/ścieżka/do/dataset.zip" ./train_yolo26_birds_linux.sh
# C) DATASET_DIR="/ścieżka/do/gotowego/datasetu" ./train_yolo26_birds_linux.sh
ROBOFLOW_DOWNLOAD_URL="${ROBOFLOW_DOWNLOAD_URL:-}"
DATASET_ZIP="${DATASET_ZIP:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$WORKDIR/.venv}"

# Instalacja PyTorch:
# Domyślnie CUDA 12.6. Jeśli to nie pasuje, ustaw TORCH_INDEX_URL ręcznie.
# Przykład CPU-only:
# TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"

# Parametry treningu głównego
MODEL="${MODEL:-yolo26x.pt}"
IMG_SIZE="${IMG_SIZE:-640}"
EPOCHS="${EPOCHS:-300}"
BATCH="${BATCH:-4}"
NBS="${NBS:-32}"
PATIENCE="${PATIENCE:-75}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-0}"
CACHE="${CACHE:-disk}"
RUN_NAME="${RUN_NAME:-birds_yolo26x_640_b${BATCH}_nbs${NBS}_cm10_e${EPOCHS}}"

# Dodatkowe tryby
RUN_SMOKE_TEST="${RUN_SMOKE_TEST:-1}"
RUN_FULL_TRAIN="${RUN_FULL_TRAIN:-1}"
RUN_POLISH="${RUN_POLISH:-0}"

# Resume przerwanego treningu:
# RESUME=1 ./train_yolo26_birds_linux.sh
RESUME="${RESUME:-0}"
RESUME_LAST="${RESUME_LAST:-$WORKDIR/runs/detect/$RUN_NAME/weights/last.pt}"

# ----------------------------
# 2. FUNKCJE POMOCNICZE
# ----------------------------

info() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

warn() {
  echo
  echo "[UWAGA] $1"
}

fail() {
  echo
  echo "[BŁĄD] $1"
  exit 1
}

need_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Brakuje komendy: $1"
}

# ----------------------------
# 3. START I LOGI
# ----------------------------

mkdir -p "$WORKDIR"
LOG_FILE="$WORKDIR/train_$(date +%Y%m%d_%H%M%S).log"

# Zapisuj wszystko jednocześnie na ekran i do pliku logu.
exec > >(tee -a "$LOG_FILE") 2>&1

info "Start skryptu"
echo "WORKDIR:     $WORKDIR"
echo "DATASET_DIR: $DATASET_DIR"
echo "LOG_FILE:    $LOG_FILE"
echo "MODEL:       $MODEL"
echo "IMG_SIZE:    $IMG_SIZE"
echo "EPOCHS:      $EPOCHS"
echo "BATCH:       $BATCH"
echo "NBS:         $NBS"
echo "DEVICE:      $DEVICE"
echo "RUN_NAME:    $RUN_NAME"

need_command "$PYTHON_BIN"
need_command curl
need_command unzip

if command -v nvidia-smi >/dev/null 2>&1; then
  info "Informacje o GPU"
  nvidia-smi || true
else
  warn "Nie znaleziono nvidia-smi. Jeśli trenujesz na NVIDIA GPU, sprawdź sterowniki CUDA/NVIDIA."
fi

# ----------------------------
# 4. ŚRODOWISKO PYTHON
# ----------------------------

info "Tworzenie / aktywacja środowiska Python"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR" || fail "Nie udało się utworzyć venv. Zainstaluj python3-venv."
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [ "$INSTALL_TORCH" = "1" ]; then
  info "Instalacja / aktualizacja PyTorch"
  echo "TORCH_INDEX_URL: $TORCH_INDEX_URL"
  python -m pip install --upgrade torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
else
  warn "Pominięto instalację PyTorch, bo INSTALL_TORCH=0"
fi

info "Instalacja / aktualizacja Ultralytics"
python -m pip install --upgrade ultralytics pyyaml

info "Kontrola Python / Torch / CUDA"
python - <<'PY'
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
PY

# ----------------------------
# 5. RESUME, jeśli chcesz kontynuować przerwany trening
# ----------------------------

if [ "$RESUME" = "1" ]; then
  info "Tryb RESUME"
  [ -f "$RESUME_LAST" ] || fail "Nie znaleziono checkpointu last.pt: $RESUME_LAST"

  yolo task=detect mode=train \
    model="$RESUME_LAST" \
    resume=True

  info "Resume zakończony"
  echo "Log: $LOG_FILE"
  exit 0
fi

# ----------------------------
# 6. DATASET Z ROBOFLOW / ZIP / LOKALNEGO FOLDERU
# ----------------------------

info "Przygotowanie datasetu"

DATA_YAML=""

# Jeśli dataset już istnieje, nie pobieramy ponownie.
if [ -f "$DATASET_DIR/data.yaml" ]; then
  DATA_YAML="$DATASET_DIR/data.yaml"
  echo "Znaleziono istniejący dataset: $DATA_YAML"
else
  mkdir -p "$DATASET_DIR"

  if [ -n "$DATASET_ZIP" ]; then
    [ -f "$DATASET_ZIP" ] || fail "DATASET_ZIP wskazuje na nieistniejący plik: $DATASET_ZIP"

    info "Rozpakowanie datasetu ZIP"
    rm -rf "$WORKDIR/dataset_raw"
    mkdir -p "$WORKDIR/dataset_raw"
    unzip -q "$DATASET_ZIP" -d "$WORKDIR/dataset_raw"

  elif [ -n "$ROBOFLOW_DOWNLOAD_URL" ]; then
    info "Pobieranie datasetu z Roboflow"
    warn "Link Roboflow może zawierać klucz API. Nie zapisuj go w publicznym repo."

    rm -rf "$WORKDIR/dataset_raw"
    mkdir -p "$WORKDIR/dataset_raw"

    curl -L "$ROBOFLOW_DOWNLOAD_URL" -o "$WORKDIR/roboflow_dataset.zip"
    unzip -q "$WORKDIR/roboflow_dataset.zip" -d "$WORKDIR/dataset_raw"

  else
    fail "Nie znaleziono data.yaml i nie podano danych. Użyj ROBOFLOW_DOWNLOAD_URL albo DATASET_ZIP, albo ustaw DATASET_DIR na gotowy dataset."
  fi

  FOUND_YAML="$(find "$WORKDIR/dataset_raw" -name data.yaml | head -n 1 || true)"
  [ -n "$FOUND_YAML" ] || fail "Nie znaleziono data.yaml po rozpakowaniu datasetu."

  RAW_ROOT="$(dirname "$FOUND_YAML")"

  # Przenosimy właściwą zawartość datasetu do DATASET_DIR.
  rm -rf "$DATASET_DIR"
  mkdir -p "$DATASET_DIR"
  cp -a "$RAW_ROOT"/. "$DATASET_DIR"/

  DATA_YAML="$DATASET_DIR/data.yaml"
fi

[ -f "$DATA_YAML" ] || fail "Brak data.yaml: $DATA_YAML"

echo "DATA_YAML: $DATA_YAML"

# ----------------------------
# 7. KONTROLA DATASETU
# ----------------------------

info "Kontrola data.yaml i struktury datasetu"

python - "$DATA_YAML" <<'PY'
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

print("Liczba klas:", len(data["names"]) if isinstance(data["names"], (list, dict)) else "nieznana")
print("Klasy:", data["names"])

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

# Próba policzenia obrazów.
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
PY

# ----------------------------
# 8. SMOKE-TEST: 1 epoka na małym modelu
# ----------------------------

cd "$WORKDIR"

if [ "$RUN_SMOKE_TEST" = "1" ]; then
  info "Smoke-test: mini-trening 1 epoki na yolo26n.pt"

  yolo task=detect mode=train \
    model=yolo26n.pt \
    data="$DATA_YAML" \
    device="$DEVICE" \
    imgsz="$IMG_SIZE" \
    batch=2 \
    epochs=1 \
    workers=2 \
    cache=False \
    plots=False \
    project="$WORKDIR/runs/detect" \
    name=smoke_test_yolo26n_e1

  info "Smoke-test zakończony"
else
  warn "Pominięto smoke-test, bo RUN_SMOKE_TEST=0"
fi

# ----------------------------
# 9. PEŁNY TRENING
# ----------------------------

if [ "$RUN_FULL_TRAIN" = "1" ]; then
  info "Pełny trening YOLO26"

  yolo task=detect mode=train \
    model="$MODEL" \
    data="$DATA_YAML" \
    device="$DEVICE" \
    imgsz="$IMG_SIZE" \
    batch="$BATCH" \
    nbs="$NBS" \
    epochs="$EPOCHS" \
    patience="$PATIENCE" \
    optimizer=auto \
    cos_lr=True \
    mosaic=1.0 \
    close_mosaic=10 \
    workers="$WORKERS" \
    cache="$CACHE" \
    save_period=10 \
    plots=True \
    seed=42 \
    project="$WORKDIR/runs/detect" \
    name="$RUN_NAME"

  info "Pełny trening zakończony"

  BEST_MODEL="$WORKDIR/runs/detect/$RUN_NAME/weights/best.pt"
  LAST_MODEL="$WORKDIR/runs/detect/$RUN_NAME/weights/last.pt"

  echo "BEST_MODEL: $BEST_MODEL"
  echo "LAST_MODEL: $LAST_MODEL"

else
  warn "Pominięto pełny trening, bo RUN_FULL_TRAIN=0"
fi

# ----------------------------
# 10. OPCJONALNY POLISHING RUN
# Domyślnie wyłączony. Uruchom:
# RUN_POLISH=1 ./train_yolo26_birds_linux.sh
# ----------------------------

if [ "$RUN_POLISH" = "1" ]; then
  info "Polishing run: delikatny fine-tuning bez mosaic"

  BEST_MODEL="$WORKDIR/runs/detect/$RUN_NAME/weights/best.pt"
  [ -f "$BEST_MODEL" ] || fail "Nie znaleziono best.pt do polishing run: $BEST_MODEL"

  yolo task=detect mode=train \
    model="$BEST_MODEL" \
    data="$DATA_YAML" \
    device="$DEVICE" \
    imgsz="$IMG_SIZE" \
    batch="$BATCH" \
    nbs="$NBS" \
    epochs=60 \
    patience=20 \
    optimizer=auto \
    lr0=0.0005 \
    lrf=0.1 \
    cos_lr=True \
    mosaic=0 \
    close_mosaic=0 \
    workers="$WORKERS" \
    cache="$CACHE" \
    plots=True \
    seed=42 \
    project="$WORKDIR/runs/detect" \
    name="${RUN_NAME}_polish_lr0005_mos0_e60"

  info "Polishing run zakończony"
else
  warn "Polishing run pominięty. To normalne. Możesz go uruchomić później przez RUN_POLISH=1."
fi

# ----------------------------
# 11. KONIEC
# ----------------------------

info "Gotowe"

echo "Wyniki:"
echo "  $WORKDIR/runs/detect"
echo
echo "Główny model po treningu:"
echo "  $WORKDIR/runs/detect/$RUN_NAME/weights/best.pt"
echo
echo "Log:"
echo "  $LOG_FILE"
echo
echo "Jeśli trening przerwał się w trakcie, kontynuuj tak:"
echo "  RESUME=1 ./train_yolo26_birds_linux.sh"
echo
echo "Jeśli zabraknie pamięci GPU, spróbuj:"
echo "  BATCH=2 ./train_yolo26_birds_linux.sh"
echo "albo:"
echo "  MODEL=yolo26l.pt BATCH=4 ./train_yolo26_birds_linux.sh"
