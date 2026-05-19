# Windows PowerShell workflow

Krótka instrukcja uruchomienia treningu YOLO26 na Windows przez skrypt PowerShell.

Plik skryptu:

```text
train_yolo26_birds_windows.ps1
```

## 1. Wymagania

Na komputerze powinny być dostępne:

- Windows 10/11,
- Python 3,
- PowerShell,
- karta NVIDIA z aktualnymi sterownikami,
- dostęp do internetu,
- dataset z Roboflow albo ZIP pobrany z Roboflow.

Sprawdzenie GPU:

```powershell
nvidia-smi
```

Jeśli `nvidia-smi` nie działa, sprawdź sterowniki NVIDIA.

## 2. Uruchomienie PowerShell

Otwórz PowerShell w folderze repozytorium.

Jeśli Windows blokuje skrypt, pozwól na jego uruchomienie tylko w tej sesji:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 3. Uruchomienie z linkiem Roboflow

W Roboflow wybierz eksport datasetu w formacie YOLO / Ultralytics, skopiuj link download i uruchom:

```powershell
.\train_yolo26_birds_windows.ps1 -RoboflowDownloadUrl "TU_WKLEJ_LINK_Z_ROBOFLOW"
```

Nie zapisuj linku Roboflow w repozytorium. Może zawierać klucz API.

## 4. Uruchomienie z lokalnym ZIP-em

Jeśli dataset został już pobrany jako ZIP:

```powershell
.\train_yolo26_birds_windows.ps1 -DatasetZip "C:\Users\jakub\Downloads\roboflow.zip"
```

Zmień ścieżkę na właściwą.

## 5. Co robi skrypt

Skrypt automatycznie:

1. tworzy folder roboczy,
2. tworzy środowisko Python `.venv`,
3. instaluje PyTorch i Ultralytics,
4. pobiera albo rozpakowuje dataset,
5. sprawdza `data.yaml`,
6. wykonuje smoke-test 1 epoki,
7. uruchamia główny trening,
8. zapisuje log,
9. zapisuje model `best.pt`.

Domyślny folder roboczy:

```text
C:\Users\<user>\yolo_train\birds_yolo26
```

Najważniejszy wynik:

```text
C:\Users\<user>\yolo_train\birds_yolo26\runs\detect\birds_yolo26x_640_b4_nbs32_cm10_e300\weights\best.pt
```

## 6. Gdy trening się przerwie

Kontynuacja z checkpointu `last.pt`:

```powershell
.\train_yolo26_birds_windows.ps1 -Resume 1
```

## 7. Gdy zabraknie pamięci GPU

Najpierw zmniejsz batch:

```powershell
.\train_yolo26_birds_windows.ps1 -Batch 2
```

Jeśli nadal brakuje VRAM:

```powershell
.\train_yolo26_birds_windows.ps1 -Batch 1
```

Możesz też użyć mniejszego modelu:

```powershell
.\train_yolo26_birds_windows.ps1 -Model yolo26l.pt -Batch 4
```

## 8. Tylko smoke-test

Jeśli chcesz sprawdzić środowisko i dataset bez pełnego treningu:

```powershell
.\train_yolo26_birds_windows.ps1 -RunFullTrain 0
```

## 9. Polishing run

Po udanym głównym treningu można uruchomić delikatny fine-tuning:

```powershell
.\train_yolo26_birds_windows.ps1 -RunPolish 1
```

## 10. Ważne

Nie dodawaj do GitHuba:

- datasetu z Roboflow,
- modeli `.pt`,
- folderu `runs/`,
- logów treningu,
- linków Roboflow z kluczem API,
- folderu `.venv`.

Do repo dodaj tylko:

```text
train_yolo26_birds_windows.ps1
Readme_PS.md
```
