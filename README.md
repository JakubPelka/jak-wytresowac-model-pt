# jak-wytresowac-model-pt
Instrukcja dla Linux - od zestawu annotowanych zdjęć do modelu .pt

# Jak wytrenować model `.pt` na Linuxie z datasetu Roboflow

Prosta instrukcja krok po kroku: od datasetu z Roboflow do własnego modelu YOLO `.pt`.

Instrukcja jest przygotowana pod przykład:

- system: Linux / Ubuntu albo podobny,
- GPU: NVIDIA z CUDA,
- framework: Ultralytics YOLO26,
- dataset: eksport z Roboflow,
- zadanie: detekcja ptaków,
- liczba klas: `1` (`bird`),
- rozmiar treningowy: `640 px`,
- pierwszy poważny trening: `300` epok,
- model startowy: `yolo26x.pt`.

> Ważne: `yolo26x.pt` oznacza start z gotowych wag pretrained i dalszy trening na własnym dataspecie. To jest praktyczne i zalecane. To nie jest „trening od zera” w sensie losowej inicjalizacji wag.

---

## 0. Docelowa struktura katalogów

Po wykonaniu instrukcji będziesz mieć mniej więcej taką strukturę:

```text
~/yolo_train/
├── .venv/
├── dataset_raw/          # oryginalny eksport z Roboflow
├── dataset_birds/        # robocza kopia datasetu do treningu
│   ├── data.yaml
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   ├── test/             # może istnieć, ale nie musi
│   │   ├── images/
│   │   └── labels/
│   └── runs/
│       └── detect/
└── roboflow.zip          # tylko tymczasowo, potem usuwany
```

---

## 1. Zasada bezpieczeństwa

Nie wrzucaj do GitHuba:

- datasetu z Roboflow,
- linku Roboflow z kluczem API,
- prywatnych danych,
- dużych plików `.pt`, jeśli nie chcesz ich świadomie publikować,
- folderu `runs/`,
- tymczasowych zipów.

Ten repozytorium powinno zawierać instrukcję, a nie dane treningowe.

---

## 2. Sprawdź, czy Linux widzi kartę NVIDIA

W terminalu:

```bash
nvidia-smi
```

Jeśli zobaczysz tabelę z kartą NVIDIA, jest dobrze.

Jeśli dostaniesz błąd typu `command not found` albo brak GPU, trening nadal może działać na CPU, ale będzie bardzo wolny. Do YOLO26X praktycznie potrzebujesz GPU.

---

## 3. Utwórz katalog projektu

```bash
mkdir -p ~/yolo_train
cd ~/yolo_train
```

---

## 4. Zainstaluj podstawowe pakiety systemowe

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv curl unzip rsync git
```

Sprawdź wersję Pythona:

```bash
python3 --version
```

Dobrze, jeśli masz Python `3.10`, `3.11`, `3.12`, `3.13` albo `3.14`.

---

## 5. Utwórz środowisko Python `.venv`

```bash
cd ~/yolo_train
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

Sprawdź, czy środowisko działa:

```bash
which python
python -V
```

Powinieneś zobaczyć ścieżkę podobną do:

```text
/home/TWOJ_UZYTKOWNIK/yolo_train/.venv/bin/python
```

> Gdy otworzysz nowy terminal, zawsze najpierw aktywuj środowisko:
>
> ```bash
> cd ~/yolo_train
> source .venv/bin/activate
> ```

---

## 6. Zainstaluj PyTorch z obsługą CUDA

Najpierw spróbuj tej wersji dla CUDA 12.6:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Sprawdź, czy PyTorch widzi GPU:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("UWAGA: PyTorch nie widzi GPU. Trening będzie bardzo wolny albo niepraktyczny.")
PY
```

Oczekiwany dobry wynik:

```text
CUDA available: True
GPU: NVIDIA ...
```

Jeśli CUDA nie działa, sprawdź aktualną komendę instalacyjną na stronie PyTorch:

- <https://pytorch.org/get-started/locally/>

Wybierz:

- OS: Linux,
- Package: Pip,
- Language: Python,
- Compute Platform: CUDA zgodne z Twoim systemem.

---

## 7. Zainstaluj Ultralytics YOLO i narzędzia

```bash
pip install -U ultralytics opencv-python-headless tqdm tensorboard pyyaml
```

Sprawdź, czy komenda `yolo` działa:

```bash
yolo version
```

Włącz TensorBoard:

```bash
yolo settings tensorboard=True
```

Sprawdź ustawienie:

```bash
yolo settings | grep -i tensorboard
```

---

## 8. Pobierz dataset z Roboflow

W Roboflow:

1. Otwórz projekt.
2. Utwórz albo wybierz konkretną wersję datasetu.
3. Kliknij `Download Dataset`.
4. Wybierz format eksportu zgodny z YOLO / Ultralytics, np. `YOLOv8`.
5. Wybierz opcję `curl` albo direct download link.
6. Skopiuj sam URL z komendy `curl`, czyli fragment wewnątrz cudzysłowu.

Przykład Roboflow może wyglądać tak:

```bash
curl -L "https://app.roboflow.com/ds/XXXXX?key=YYYYY" > roboflow.zip
```

Do następnego polecenia wklej tylko ten fragment:

```text
https://app.roboflow.com/ds/XXXXX?key=YYYYY
```

Teraz w terminalu:

```bash
cd ~/yolo_train
source .venv/bin/activate

rm -rf dataset_raw dataset_birds roboflow.zip
mkdir -p dataset_raw

export ROBOFLOW_URL='WKLEJ_TUTAJ_LINK_Z_ROBOFLOW'

curl -L "$ROBOFLOW_URL" -o roboflow.zip
unzip -q roboflow.zip -d dataset_raw
rm roboflow.zip
```

Sprawdź, co zostało pobrane:

```bash
find ~/yolo_train/dataset_raw -maxdepth 3 -type d | sort
```

Typowa struktura powinna wyglądać podobnie do:

```text
dataset_raw/train/images
dataset_raw/train/labels
dataset_raw/valid/images
dataset_raw/valid/labels
dataset_raw/test/images
dataset_raw/test/labels
```

Jeśli masz `val` zamiast `valid`, dalszy skrypt spróbuje to naprawić.

---

## 9. Przygotuj roboczą kopię datasetu

Nie trenuj bezpośrednio na `dataset_raw`. Zrobimy kopię roboczą `dataset_birds`.

```bash
cd ~/yolo_train
rm -rf dataset_birds
rsync -a dataset_raw/ dataset_birds/
```

Jeśli Roboflow utworzył `val/` zamiast `valid/`, zmień nazwę:

```bash
cd ~/yolo_train/dataset_birds
if [ -d val ] && [ ! -d valid ]; then mv val valid; fi
```

Upewnij się, że podstawowe foldery istnieją:

```bash
mkdir -p train/images train/labels valid/images valid/labels test/images test/labels
```

---

## 10. Wymuś jedną klasę: `bird = 0`

Użyj tej sekcji tylko wtedy, gdy chcesz wykrywać jedną klasę: `bird`.

To jest właściwe dla przypadku: „wykryj ptaki na zdjęciu”, niezależnie od gatunku.

Nie używaj tej sekcji, jeśli chcesz osobno wykrywać gatunki, np. `crow`, `sparrow`, `goose` itd.

```bash
cd ~/yolo_train/dataset_birds

find . -path "*/labels/*.txt" -type f -print0 \
  | xargs -0 -I{} sed -i -E 's/^[0-9]+ /0 /' {}
```

---

## 11. Wyczyść duplikaty i błędne etykiety

Ten skrypt:

- usuwa identyczne zduplikowane wiersze w labelach,
- zostawia tylko poprawny format YOLO: `class x_center y_center width height`,
- wymusza klasę `0`,
- ignoruje błędne rekordy.

```bash
cd ~/yolo_train/dataset_birds

python - <<'PY'
from pathlib import Path

base = Path(".")
changed_files = 0
before_total = 0
after_total = 0
invalid_rows = 0

for label_path in base.glob("**/labels/*.txt"):
    try:
        original_lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception as e:
        print(f"Nie mogę odczytać: {label_path} -> {e}")
        continue

    good = []
    for ln in original_lines:
        parts = ln.split()
        if len(parts) != 5:
            invalid_rows += 1
            continue
        try:
            values = [float(v) for v in parts[1:]]
        except ValueError:
            invalid_rows += 1
            continue
        if not all(0.0 <= v <= 1.0 for v in values):
            invalid_rows += 1
            continue
        good.append("0 " + " ".join(f"{v:.6f}" for v in values))

    cleaned = sorted(set(good))
    before_total += len(original_lines)
    after_total += len(cleaned)

    if cleaned != original_lines:
        changed_files += 1
        label_path.write_text("\n".join(cleaned) + ("\n" if cleaned else ""), encoding="utf-8")

print(f"Zmienione pliki labeli: {changed_files}")
print(f"Wiersze labeli: {before_total} -> {after_total}")
print(f"Odrzucone błędne wiersze: {invalid_rows}")
PY
```

---

## 12. Sprawdź liczbę zdjęć i etykiet

```bash
cd ~/yolo_train/dataset_birds

python - <<'PY'
from pathlib import Path

image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
base = Path(".")

for split in ["train", "valid", "test"]:
    image_dir = base / split / "images"
    label_dir = base / split / "labels"

    images = {p.stem for p in image_dir.glob("*") if p.suffix.lower() in image_exts} if image_dir.exists() else set()
    labels = {p.stem for p in label_dir.glob("*.txt")} if label_dir.exists() else set()

    no_label = images - labels
    orphan_labels = labels - images

    print(f"\n{split}")
    print(f"  images:          {len(images)}")
    print(f"  labels:          {len(labels)}")
    print(f"  images no label: {len(no_label)}")
    print(f"  orphan labels:   {len(orphan_labels)}")

    if orphan_labels:
        print("  UWAGA: są label files bez obrazów. Pierwsze przykłady:")
        for x in sorted(list(orphan_labels))[:10]:
            print("   -", x)
PY
```

Interpretacja:

- `orphan labels` powinno być `0`,
- `images no label` może być większe od `0`, jeśli masz zdjęcia bez ptaków jako tło/background,
- dla datasetu około 9000 zdjęć suma `train + valid + test` powinna być bliska 9000.

---

## 13. Utwórz lokalny plik `data.yaml`

Ten plik mówi YOLO, gdzie są dane i jakie są klasy.

```bash
cd ~/yolo_train/dataset_birds

cat > data.yaml <<'YAML'
path: .
train: train/images
val: valid/images
test: test/images
nc: 1
names: ['bird']
YAML

cat data.yaml
```

---

## 14. Smoke-test: mini-trening 1 epoki

To jest test techniczny przed właściwym treningiem.

Jeśli ten krok nie działa, nie zaczynaj dużego treningu.

```bash
cd ~/yolo_train
source .venv/bin/activate
cd ~/yolo_train/dataset_birds

yolo task=detect mode=train \
  model=yolo26n.pt \
  data=data.yaml \
  device=0 \
  imgsz=640 \
  batch=2 \
  epochs=1 \
  workers=2 \
  cache=False \
  plots=False \
  name=smoke_test_yolo26n_e1
```

Jeśli pojawi się błąd `CUDA out of memory`, spróbuj:

```bash
yolo task=detect mode=train \
  model=yolo26n.pt \
  data=data.yaml \
  device=0 \
  imgsz=640 \
  batch=1 \
  epochs=1 \
  workers=2 \
  cache=False \
  plots=False \
  name=smoke_test_yolo26n_e1_b1
```

---

## 15. Główny trening YOLO26X — wariant podstawowy

To jest główna komenda dla datasetu około 9000 zdjęć.

Startujemy z `yolo26x.pt`, `imgsz=640`, `epochs=300`.

```bash
cd ~/yolo_train
source .venv/bin/activate
cd ~/yolo_train/dataset_birds

yolo task=detect mode=train \
  model=yolo26x.pt \
  data=data.yaml \
  device=0 \
  imgsz=640 \
  batch=4 \
  nbs=32 \
  epochs=300 \
  patience=75 \
  optimizer=auto \
  cos_lr=True \
  mosaic=1.0 \
  close_mosaic=10 \
  workers=8 \
  cache=disk \
  save_period=10 \
  plots=True \
  seed=42 \
  name=birds_yolo26x_640_b4_nbs32_cm10_e300
```

Najważniejsze parametry:

- `model=yolo26x.pt` — duży model YOLO26, dobry jeśli GPU daje radę,
- `imgsz=640` — obrazy treningowe skalowane do 640 px,
- `batch=4` — fizyczny batch; zmniejsz przy braku VRAM,
- `nbs=32` — nominal batch size; stabilniejsze niż bardzo niskie wartości,
- `epochs=300` — maksymalnie 300 epok,
- `patience=75` — early stopping po 75 epokach bez poprawy,
- `optimizer=auto` — Ultralytics sam dobiera optimizer,
- `cos_lr=True` — łagodniejszy harmonogram learning rate,
- `mosaic=1.0` — mocna augmentacja mosaic,
- `close_mosaic=10` — wyłącza mosaic w ostatnich 10 epokach,
- `cache=disk` — cache na dysku, bez ryzyka zapchania RAM,
- `save_period=10` — zapis checkpointu co 10 epok,
- `seed=42` — łatwiej porównywać runy.

---

## 16. Jeśli zabraknie pamięci GPU

Jeśli zobaczysz błąd podobny do:

```text
CUDA out of memory
```

najpierw spróbuj zmienić tylko `batch=4` na:

```text
batch=2
```

Jeśli nadal brakuje VRAM, użyj mniejszego modelu.

### Wariant awaryjny: YOLO26L

```bash
cd ~/yolo_train
source .venv/bin/activate
cd ~/yolo_train/dataset_birds

yolo task=detect mode=train \
  model=yolo26l.pt \
  data=data.yaml \
  device=0 \
  imgsz=640 \
  batch=4 \
  nbs=32 \
  epochs=300 \
  patience=75 \
  optimizer=auto \
  cos_lr=True \
  mosaic=1.0 \
  close_mosaic=10 \
  workers=8 \
  cache=disk \
  save_period=10 \
  plots=True \
  seed=42 \
  name=birds_yolo26l_640_b4_nbs32_cm10_e300
```

### Wariant jeszcze lżejszy: YOLO26M

```bash
cd ~/yolo_train
source .venv/bin/activate
cd ~/yolo_train/dataset_birds

yolo task=detect mode=train \
  model=yolo26m.pt \
  data=data.yaml \
  device=0 \
  imgsz=640 \
  batch=4 \
  nbs=32 \
  epochs=300 \
  patience=75 \
  optimizer=auto \
  cos_lr=True \
  mosaic=1.0 \
  close_mosaic=10 \
  workers=8 \
  cache=disk \
  save_period=10 \
  plots=True \
  seed=42 \
  name=birds_yolo26m_640_b4_nbs32_cm10_e300
```

---

## 17. Monitorowanie GPU podczas treningu

Otwórz drugi terminal:

```bash
watch -n1 nvidia-smi
```

Opcjonalnie zainstaluj wygodniejszy podgląd:

```bash
sudo apt install -y nvtop
nvtop
```

---

## 18. Zabezpieczenie przed usypianiem systemu

W osobnym terminalu uruchom:

```bash
systemd-inhibit --what=idle:sleep --why="YOLO training" sleep infinity
```

Zostaw ten terminal otwarty na czas treningu.

Po treningu zatrzymaj go:

```text
Ctrl+C
```

---

## 19. TensorBoard — podgląd wykresów treningu

Otwórz nowy terminal:

```bash
cd ~/yolo_train
source .venv/bin/activate

tensorboard --logdir ~/yolo_train/dataset_birds/runs/detect --port 6006
```

W przeglądarce otwórz:

```text
http://localhost:6006
```

Jeśli port jest zajęty:

```bash
tensorboard --logdir ~/yolo_train/dataset_birds/runs/detect --port 6007
```

Wtedy otwórz:

```text
http://localhost:6007
```

---

## 20. Gdzie znajdziesz wynik treningu

Po treningu najważniejsze pliki będą tutaj:

```text
~/yolo_train/dataset_birds/runs/detect/birds_yolo26x_640_b4_nbs32_cm10_e300/weights/best.pt
~/yolo_train/dataset_birds/runs/detect/birds_yolo26x_640_b4_nbs32_cm10_e300/weights/last.pt
```

Najczęściej używasz:

```text
best.pt
```

`last.pt` służy głównie do kontynuacji treningu.

---

## 21. Jak bezpiecznie przerwać trening

W terminalu, w którym działa trening, naciśnij raz:

```text
Ctrl+C
```

Nie zamykaj okna brutalnie, jeśli nie musisz.

YOLO powinno zapisać aktualny checkpoint `last.pt`.

---

## 22. Jak wznowić ten sam trening po przerwaniu

Użyj `resume=True` i wskaż `last.pt`.

```bash
cd ~/yolo_train
source .venv/bin/activate
cd ~/yolo_train/dataset_birds

yolo task=detect mode=train resume=True \
  model=runs/detect/birds_yolo26x_640_b4_nbs32_cm10_e300/weights/last.pt
```

Ważne:

- nie dodawaj tutaj `name=`,
- nie zmieniaj parametrów treningu,
- resume używa zapisanej konfiguracji z poprzedniego runu.

---

## 23. Jak kontynuować po dojściu do 300/300 epok

Jeśli trening doszedł do końca `300/300`, nie używaj już `resume=True`.

Zacznij nowy run z poprzedniego `last.pt` albo `best.pt`.

Przykład kontynuacji z `last.pt`:

```bash
cd ~/yolo_train
source .venv/bin/activate
cd ~/yolo_train/dataset_birds

yolo task=detect mode=train \
  model=runs/detect/birds_yolo26x_640_b4_nbs32_cm10_e300/weights/last.pt \
  data=data.yaml \
  device=0 \
  imgsz=640 \
  batch=4 \
  nbs=32 \
  epochs=100 \
  patience=30 \
  optimizer=auto \
  cos_lr=True \
  mosaic=1.0 \
  close_mosaic=10 \
  workers=8 \
  cache=disk \
  save_period=10 \
  plots=True \
  seed=42 \
  name=birds_yolo26x_640_continue_e100
```

---

## 24. Fine-tuning / polishing run

Użyj tego dopiero wtedy, gdy główny trening wygląda dobrze i chcesz ostrożnie dopracować model.

Ten wariant:

- startuje z `best.pt`,
- ma niższy learning rate,
- wyłącza mosaic,
- robi krótszy trening.

```bash
cd ~/yolo_train
source .venv/bin/activate
cd ~/yolo_train/dataset_birds

yolo task=detect mode=train \
  model=runs/detect/birds_yolo26x_640_b4_nbs32_cm10_e300/weights/best.pt \
  data=data.yaml \
  device=0 \
  imgsz=640 \
  batch=4 \
  nbs=32 \
  epochs=60 \
  patience=20 \
  optimizer=auto \
  lr0=0.0005 \
  lrf=0.1 \
  cos_lr=True \
  mosaic=0 \
  close_mosaic=0 \
  workers=8 \
  cache=disk \
  plots=True \
  seed=42 \
  name=birds_yolo26x_640_polish_lr0005_mos0_e60
```

Wynik będzie tutaj:

```text
~/yolo_train/dataset_birds/runs/detect/birds_yolo26x_640_polish_lr0005_mos0_e60/weights/best.pt
```

---

## 25. Walidacja modelu

Dla głównego modelu:

```bash
cd ~/yolo_train
source .venv/bin/activate
cd ~/yolo_train/dataset_birds

yolo task=detect mode=val \
  model=runs/detect/birds_yolo26x_640_b4_nbs32_cm10_e300/weights/best.pt \
  data=data.yaml \
  device=0 \
  imgsz=640 \
  batch=4 \
  plots=True
```

Dla modelu po polishing run:

```bash
yolo task=detect mode=val \
  model=runs/detect/birds_yolo26x_640_polish_lr0005_mos0_e60/weights/best.pt \
  data=data.yaml \
  device=0 \
  imgsz=640 \
  batch=4 \
  plots=True
```

---

## 26. Predykcja testowa na obrazach walidacyjnych

Dla głównego modelu:

```bash
cd ~/yolo_train
source .venv/bin/activate
cd ~/yolo_train/dataset_birds

yolo task=detect mode=predict \
  model=runs/detect/birds_yolo26x_640_b4_nbs32_cm10_e300/weights/best.pt \
  source=valid/images \
  device=0 \
  imgsz=640 \
  conf=0.25 \
  iou=0.7 \
  save=True
```

Wyniki znajdziesz w nowym folderze wewnątrz:

```text
~/yolo_train/dataset_birds/runs/detect/
```

Jeśli model wykrywa za mało ptaków, spróbuj niższego progu:

```text
conf=0.15
```

Jeśli model wykrywa zbyt dużo fałszywych ptaków, spróbuj wyższego progu:

```text
conf=0.35
```

---

## 27. Eksport modelu

### Natywny `.pt`

Najważniejszy plik już istnieje:

```text
runs/detect/birds_yolo26x_640_b4_nbs32_cm10_e300/weights/best.pt
```

To jest plik, którego najczęściej użyjesz w aplikacjach opartych o Ultralytics.

### ONNX

```bash
cd ~/yolo_train
source .venv/bin/activate
cd ~/yolo_train/dataset_birds

yolo task=detect mode=export \
  model=runs/detect/birds_yolo26x_640_b4_nbs32_cm10_e300/weights/best.pt \
  format=onnx \
  opset=17 \
  dynamic=True
```

---

## 28. Najczęstsze problemy

### `yolo: command not found`

Aktywuj środowisko:

```bash
cd ~/yolo_train
source .venv/bin/activate
```

Jeśli nadal nie działa:

```bash
pip install -U ultralytics
```

---

### `CUDA out of memory`

Kolejność działań:

1. zmień `batch=4` na `batch=2`,
2. jeśli nadal źle, zmień `batch=2` na `batch=1`,
3. jeśli nadal źle, użyj `yolo26l.pt`,
4. jeśli nadal źle, użyj `yolo26m.pt`,
5. zostaw `imgsz=640`, dopóki możesz.

---

### PyTorch nie widzi GPU

Sprawdź:

```bash
nvidia-smi
```

Potem:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

Jeśli `False`, najczęściej problem dotyczy instalacji PyTorch, sterownika NVIDIA albo wersji CUDA.

---

### Roboflow pobrał inną liczbę zdjęć niż w panelu

To może się zdarzyć. Najprościej:

1. utwórz nową wersję datasetu w Roboflow,
2. pobierz eksport jeszcze raz,
3. ponownie wykonaj kroki od sekcji 8.

---

### Trening zakończył się za wcześnie

Jeśli zadziałał early stopping, to zwykle znaczy, że metryki walidacyjne przestały się poprawiać przez `patience=75` epok.

To nie musi być błąd.

Sprawdź:

```text
runs/detect/NAZWA_RUNU/results.png
runs/detect/NAZWA_RUNU/results.csv
```

---

## 29. Minimalna checklista przed dużym treningiem

Przed uruchomieniem treningu 300 epok sprawdź:

- [ ] `nvidia-smi` działa,
- [ ] `torch.cuda.is_available()` zwraca `True`,
- [ ] `yolo version` działa,
- [ ] dataset jest w `~/yolo_train/dataset_birds`,
- [ ] `data.yaml` ma `nc: 1` i `names: ['bird']`,
- [ ] `orphan labels = 0`,
- [ ] smoke-test 1 epoki działa,
- [ ] masz wolne miejsce na dysku,
- [ ] komputer nie przejdzie w tryb uśpienia.

---

## 30. Zalecany pierwszy scenariusz dla tego projektu

Dla datasetu około 9000 zdjęć ptaków:

1. Pobierz dataset z Roboflow.
2. Przygotuj `dataset_birds`.
3. Wymuś jedną klasę `bird = 0`, jeśli celem jest detekcja ptaków, a nie gatunków.
4. Wykonaj smoke-test 1 epoki na `yolo26n.pt`.
5. Uruchom główny trening `yolo26x.pt`, `imgsz=640`, `epochs=300`, `batch=4`.
6. Jeśli brakuje VRAM, przejdź na `batch=2`, potem ewentualnie `yolo26l.pt` albo `yolo26m.pt`.
7. Po treningu sprawdź `best.pt`, wykresy i predykcje na `valid/images`.
8. Dopiero potem rozważ polishing run.

