# jak-wytresowac-model-pt

Instrukcja dla Linuxa: od datasetu z Roboflow do własnego modelu YOLO `.pt`.

Ten repozytorium ma być prostą, praktyczną instrukcją dla osoby, która chce wytrenować własny model `.pt` na Linuxie i nie chce za każdym razem ręcznie przepisywać wielu komend.

Główny scenariusz:

- dataset z Roboflow,
- około 9000 zannotowanych zdjęć ptaków,
- zadanie: detekcja ptaków,
- jedna klasa: `bird`,
- model startowy: `yolo26x.pt`,
- rozmiar treningowy: `640 px`,
- pierwszy poważny trening: `300` epok,
- system: Linux / Ubuntu albo podobny,
- GPU: NVIDIA z CUDA.

> Ważne: `yolo26x.pt` oznacza start z gotowych wag pretrained i dalszy trening na własnym dataspecie. To jest praktyczne podejście. To nie jest „trening od zera” w sensie losowej inicjalizacji wag.

---

## 1. Najprostsza ścieżka: jeden skrypt `.sh`

W repo znajduje się skrypt:

```text
train_yolo26_birds_linux.sh
```

Ten skrypt automatyzuje większość pracy:

1. tworzy katalog roboczy,
2. tworzy środowisko Python `.venv`,
3. instaluje PyTorch i Ultralytics,
4. pobiera albo rozpakowuje dataset,
5. sprawdza `data.yaml`,
6. robi smoke-test 1 epoki,
7. uruchamia główny trening,
8. zapisuje log,
9. pozwala wznowić przerwany trening,
10. opcjonalnie pozwala odpalić polishing run.

To jest zalecany start.

---

## 2. Pobranie repozytorium

Na Linuxie:

```bash
cd ~
git clone https://github.com/JakubPelka/jak-wytresowac-model-pt.git
cd jak-wytresowac-model-pt
```

Nadaj skryptowi prawo uruchamiania:

```bash
chmod +x train_yolo26_birds_linux.sh
```

---

## 3. Opcja A: dataset pobierany bezpośrednio z Roboflow

W Roboflow:

1. otwórz projekt,
2. wybierz wersję datasetu,
3. kliknij `Download Dataset`,
4. wybierz format zgodny z YOLO / Ultralytics, np. `YOLOv8`,
5. wybierz opcję `curl`,
6. skopiuj sam URL z komendy `curl`.

Przykład komendy z Roboflow może wyglądać tak:

```bash
curl -L "https://app.roboflow.com/ds/XXXXX?key=YYYYY" > roboflow.zip
```

Do skryptu wklejasz tylko fragment wewnątrz cudzysłowu:

```text
https://app.roboflow.com/ds/XXXXX?key=YYYYY
```

Uruchomienie:

```bash
ROBOFLOW_DOWNLOAD_URL="TU_WKLEJ_LINK_Z_ROBOFLOW" ./train_yolo26_birds_linux.sh
```

Nie zapisuj tego linku w repozytorium. Link Roboflow może zawierać klucz API.

---

## 4. Opcja B: dataset pobrany wcześniej jako ZIP

Jeśli masz już ZIP z Roboflow, np. w `Downloads`:

```bash
DATASET_ZIP="$HOME/Downloads/roboflow.zip" ./train_yolo26_birds_linux.sh
```

Skrypt sam rozpakowuje ZIP, szuka `data.yaml` i przygotowuje dataset do treningu.

---

## 5. Opcja C: dataset jest już rozpakowany lokalnie

Jeśli dataset jest już rozpakowany i ma plik `data.yaml`, możesz wskazać folder:

```bash
DATASET_DIR="$HOME/yolo_train/moj_dataset" ./train_yolo26_birds_linux.sh
```

Folder powinien mieć strukturę podobną do:

```text
moj_dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

`test/` może istnieć, ale nie musi.

---

## 6. Co dokładnie robi domyślny trening

Domyślnie skrypt uruchamia główny trening z parametrami:

```bash
model=yolo26x.pt
imgsz=640
batch=4
nbs=32
epochs=300
patience=75
optimizer=auto
cos_lr=True
mosaic=1.0
close_mosaic=10
workers=8
cache=disk
save_period=10
seed=42
```

Nazwa głównego runu:

```text
birds_yolo26x_640_b4_nbs32_cm10_e300
```

Najważniejszy wynik po treningu:

```text
~/yolo_train/birds_yolo26/runs/detect/birds_yolo26x_640_b4_nbs32_cm10_e300/weights/best.pt
```

Checkpoint do kontynuacji:

```text
~/yolo_train/birds_yolo26/runs/detect/birds_yolo26x_640_b4_nbs32_cm10_e300/weights/last.pt
```

---

## 7. Gdzie skrypt tworzy pliki

Domyślna struktura robocza:

```text
~/yolo_train/
└── birds_yolo26/
    ├── .venv/
    ├── dataset/
    │   ├── data.yaml
    │   ├── train/
    │   ├── valid/
    │   └── test/
    ├── runs/
    │   └── detect/
    └── train_YYYYMMDD_HHMMSS.log
```

Repozytorium GitHub nie powinno zawierać datasetu, wyników treningu ani modeli `.pt`, chyba że świadomie publikujesz konkretny model jako release.

---

## 8. Jeśli trening się przerwie

Jeśli trening został przerwany, np. przez `Ctrl+C`, restart komputera albo błąd systemu, użyj:

```bash
RESUME=1 ./train_yolo26_birds_linux.sh
```

Skrypt domyślnie spróbuje wznowić:

```text
~/yolo_train/birds_yolo26/runs/detect/birds_yolo26x_640_b4_nbs32_cm10_e300/weights/last.pt
```

Jeśli chcesz wskazać inny checkpoint:

```bash
RESUME=1 RESUME_LAST="$HOME/sciezka/do/last.pt" ./train_yolo26_birds_linux.sh
```

W trybie `RESUME=1` nie zmieniaj parametrów treningu. Resume ma kontynuować ten sam run z zapisanym stanem.

---

## 9. Jeśli trening doszedł do 300/300 epok

Jeśli trening zakończył pełne `300/300` epok, nie używaj już `RESUME=1`.

Wtedy są dwie możliwości:

1. zostawić model jako gotowy,
2. uruchomić polishing run,
3. rozpocząć nowy trening-kontynuację ręcznie z `best.pt` albo `last.pt`.

W tym repo skrypt obsługuje polishing run, ale nie robi automatycznej kontynuacji po 300 epokach jako osobnego eksperymentu. To lepiej robić świadomie, po sprawdzeniu wyników.

---

## 10. Opcjonalny polishing run

Polishing run to krótki, ostrożny fine-tuning po głównym treningu.

Uruchom go dopiero wtedy, gdy główny trening wygląda dobrze:

```bash
RUN_POLISH=1 ./train_yolo26_birds_linux.sh
```

Ten tryb:

- startuje z `best.pt` głównego treningu,
- wyłącza mosaic,
- używa niższego learning rate,
- trenuje krócej.

Parametry polishing run:

```bash
epochs=60
patience=20
lr0=0.0005
lrf=0.1
mosaic=0
close_mosaic=0
```

Wynik:

```text
~/yolo_train/birds_yolo26/runs/detect/birds_yolo26x_640_b4_nbs32_cm10_e300_polish_lr0005_mos0_e60/weights/best.pt
```

---

## 11. Jeśli zabraknie pamięci GPU

Jeśli pojawi się błąd:

```text
CUDA out of memory
```

Najpierw zmniejsz batch:

```bash
BATCH=2 ./train_yolo26_birds_linux.sh
```

Jeśli nadal brakuje VRAM:

```bash
BATCH=1 ./train_yolo26_birds_linux.sh
```

Jeśli nadal jest problem, użyj mniejszego modelu:

```bash
MODEL=yolo26l.pt BATCH=4 ./train_yolo26_birds_linux.sh
```

Jeszcze lżejszy wariant:

```bash
MODEL=yolo26m.pt BATCH=4 ./train_yolo26_birds_linux.sh
```

Na początku nie zmieniaj `imgsz=640`, jeśli nie musisz. Najpierw zmniejsz `batch` albo model.

---

## 12. Uruchomienie tylko smoke-testu

Jeśli chcesz tylko sprawdzić, czy instalacja i dataset działają, bez pełnego treningu:

```bash
RUN_FULL_TRAIN=0 ./train_yolo26_birds_linux.sh
```

Skrypt zrobi wtedy tylko przygotowanie środowiska, kontrolę datasetu i smoke-test 1 epoki.

---

## 13. Pominięcie smoke-testu

Jeśli dataset był już sprawdzony i chcesz od razu odpalić pełny trening:

```bash
RUN_SMOKE_TEST=0 ./train_yolo26_birds_linux.sh
```

Dla pierwszego uruchomienia nie zalecam pomijania smoke-testu.

---

## 14. Własny folder roboczy

Domyślnie skrypt używa:

```text
~/yolo_train/birds_yolo26
```

Możesz wskazać inny folder:

```bash
WORKDIR="$HOME/yolo_train/test_birds_run_01" ./train_yolo26_birds_linux.sh
```

To przydaje się, gdy chcesz zrobić osobne eksperymenty bez nadpisywania poprzednich wyników.

---

## 15. CPU-only albo inna wersja PyTorch

Domyślnie skrypt instaluje PyTorch dla CUDA 12.6:

```text
https://download.pytorch.org/whl/cu126
```

Jeśli chcesz CPU-only:

```bash
TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu" ./train_yolo26_birds_linux.sh
```

CPU-only przy YOLO26X i 9000 zdjęć będzie bardzo wolne. To raczej tylko test techniczny.

Jeśli Twoja konfiguracja CUDA wymaga innej wersji PyTorch, ustaw `TORCH_INDEX_URL` zgodnie z aktualną komendą ze strony PyTorch.

---

## 16. Gdy PyTorch i Ultralytics są już zainstalowane

Jeśli nie chcesz, żeby skrypt ponownie instalował PyTorch:

```bash
INSTALL_TORCH=0 ./train_yolo26_birds_linux.sh
```

Skrypt nadal zainstaluje / zaktualizuje `ultralytics` i `pyyaml`.

---

## 17. Monitorowanie GPU

W drugim terminalu:

```bash
watch -n1 nvidia-smi
```

Wygodniejsza opcja:

```bash
sudo apt install -y nvtop
nvtop
```

---

## 18. Zabezpieczenie przed uśpieniem systemu

W osobnym terminalu:

```bash
systemd-inhibit --what=idle:sleep --why="YOLO training" sleep infinity
```

Zostaw ten terminal otwarty podczas treningu.

Po treningu zatrzymaj:

```text
Ctrl+C
```

---

## 19. TensorBoard

Po rozpoczęciu treningu możesz podejrzeć wykresy.

W nowym terminalu:

```bash
cd ~/yolo_train/birds_yolo26
source .venv/bin/activate
tensorboard --logdir runs/detect --port 6006
```

W przeglądarce:

```text
http://localhost:6006
```

Jeśli port jest zajęty:

```bash
tensorboard --logdir runs/detect --port 6007
```

---

## 20. Walidacja modelu po treningu

Po treningu:

```bash
cd ~/yolo_train/birds_yolo26
source .venv/bin/activate

yolo task=detect mode=val \
  model=runs/detect/birds_yolo26x_640_b4_nbs32_cm10_e300/weights/best.pt \
  data=dataset/data.yaml \
  device=0 \
  imgsz=640 \
  batch=4 \
  plots=True
```

Jeśli trenowałeś inną nazwą runu albo innym modelem, zmień ścieżkę do `best.pt`.

---

## 21. Testowa predykcja na obrazach walidacyjnych

```bash
cd ~/yolo_train/birds_yolo26
source .venv/bin/activate

yolo task=detect mode=predict \
  model=runs/detect/birds_yolo26x_640_b4_nbs32_cm10_e300/weights/best.pt \
  source=dataset/valid/images \
  device=0 \
  imgsz=640 \
  conf=0.25 \
  iou=0.7 \
  save=True
```

Wyniki znajdziesz w nowym folderze w:

```text
~/yolo_train/birds_yolo26/runs/detect/
```

Jeśli model wykrywa za mało ptaków, przetestuj niższy próg:

```bash
conf=0.15
```

Jeśli model ma dużo fałszywych trafień:

```bash
conf=0.35
```

---

## 22. Eksport modelu

Natywny model `.pt` jest gotowy tutaj:

```text
~/yolo_train/birds_yolo26/runs/detect/birds_yolo26x_640_b4_nbs32_cm10_e300/weights/best.pt
```

Eksport do ONNX:

```bash
cd ~/yolo_train/birds_yolo26
source .venv/bin/activate

yolo task=detect mode=export \
  model=runs/detect/birds_yolo26x_640_b4_nbs32_cm10_e300/weights/best.pt \
  format=onnx \
  opset=17 \
  dynamic=True
```

---

## 23. Czy lepiej jeden skrypt czy kilka skryptów?

Na start: jeden skrypt jest OK.

Lepszy podział na przyszłość może wyglądać tak:

```text
scripts/
├── 01_prepare_env.sh
├── 02_download_dataset.sh
├── 03_train_yolo26.sh
├── 04_resume_training.sh
└── 05_polish_model.sh
```

Ale na tym etapie jeden skrypt jest wygodniejszy, bo:

- łatwiej go odpalić,
- mniej decyzji na początku,
- mniejsze ryzyko pomylenia kolejności,
- tryby `RESUME=1`, `RUN_POLISH=1`, `DATASET_ZIP=...` i `MODEL=...` już pokrywają najważniejsze przypadki.

Jeśli checkpointy i eksperymenty zaczną się mnożyć, wtedy warto rozdzielić skrypt na 2–3 mniejsze.

---

## 24. Minimalna checklista przed pełnym treningiem

Przed uruchomieniem treningu 300 epok sprawdź:

- `nvidia-smi` działa,
- dataset z Roboflow został pobrany,
- `data.yaml` istnieje,
- smoke-test 1 epoki działa,
- masz wystarczająco dużo miejsca na dysku,
- komputer nie przejdzie w tryb uśpienia,
- link Roboflow z kluczem API nie został zapisany w repo,
- dataset, `runs/`, logi i modele `.pt` nie trafią przypadkiem do GitHuba.

---

## 25. Najkrótszy scenariusz dla tego projektu

Dla datasetu około 9000 zdjęć ptaków:

```bash
cd ~
git clone https://github.com/JakubPelka/jak-wytresowac-model-pt.git
cd jak-wytresowac-model-pt
chmod +x train_yolo26_birds_linux.sh
ROBOFLOW_DOWNLOAD_URL="TU_WKLEJ_LINK_Z_ROBOFLOW" ./train_yolo26_birds_linux.sh
```

Jeśli brakuje VRAM:

```bash
BATCH=2 ROBOFLOW_DOWNLOAD_URL="TU_WKLEJ_LINK_Z_ROBOFLOW" ./train_yolo26_birds_linux.sh
```

Jeśli trening się przerwie:

```bash
RESUME=1 ./train_yolo26_birds_linux.sh
```

Po wszystkim najważniejszy plik to:

```text
~/yolo_train/birds_yolo26/runs/detect/birds_yolo26x_640_b4_nbs32_cm10_e300/weights/best.pt
```
