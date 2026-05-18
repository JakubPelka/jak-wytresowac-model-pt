# jak-wytresowac-model-pt
Instrukcja dla Linux - od zestawu annotowanych zdjęć do modelu .pt

# Training a YOLO26 `.pt` model on Linux with Roboflow

This guide describes a clean, repeatable workflow for training a custom Ultralytics YOLO26 detection model on Linux.

Default setup in this README:

- OS: Linux / Ubuntu-like system
- GPU: NVIDIA CUDA-capable GPU recommended
- Framework: Ultralytics YOLO26
- Dataset source: Roboflow YOLO export
- Training image size: `640`
- Training epochs: `300`
- Default model: `yolo26x.pt`
- Working folder: `~/yolo_train`
- Dataset folder: `~/yolo_train/dataset_merged`

> If your GPU has limited VRAM, start with `yolo26n.pt`, `yolo26s.pt`, or reduce `batch`.

---

## 0. Target folder structure

```text
~/yolo_train/
├── .venv/
├── dataset1/
├── dataset2/
├── dataset_merged/
│   ├── data.yaml
│   ├── yolo26x.pt              # optional, downloaded automatically if missing
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   └── runs/
│       └── detect/
└── README.md
```

---

## 1. Create project folder

```bash
mkdir -p ~/yolo_train
cd ~/yolo_train
```

---

## 2. Install system packages

```bash
sudo apt update
sudo apt install -y python3.12-venv curl unzip
```

If `python3.12-venv` is not available:

```bash
sudo apt install -y python3-venv
```

---

## 3. Create and activate Python virtual environment

```bash
cd ~/yolo_train
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
```

Check that the environment is active:

```bash
which python
python -V
```

You should see a path inside:

```text
/home/jakub-pelka/yolo_train/.venv/
```

---

## 4. Install PyTorch for NVIDIA GPU

For CUDA 12.4:

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
```

If this fails, try CUDA 12.1:

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
```

CPU-only fallback:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

Verify GPU access:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

Expected result for GPU training:

```text
CUDA available: True
GPU: NVIDIA ...
```

---

## 5. Install Ultralytics and tools

```bash
pip install ultralytics opencv-python-headless tqdm tensorboard pyyaml
```

Check YOLO command:

```bash
yolo version
```

Enable TensorBoard logging:

```bash
yolo settings tensorboard=True
yolo settings | grep -i tensorboard
```

---

## 6. Download Roboflow datasets

### Example: download one Roboflow dataset

Replace the URL with your own Roboflow YOLO export URL.

```bash
cd ~/yolo_train
mkdir -p dataset1
cd dataset1

curl -L "https://app.roboflow.com/ds/YOUR_DATASET_LINK?key=YOUR_KEY" -o roboflow.zip
unzip -q roboflow.zip
rm roboflow.zip
```

For a second dataset:

```bash
cd ~/yolo_train
mkdir -p dataset2
cd dataset2

curl -L "https://app.roboflow.com/ds/YOUR_SECOND_DATASET_LINK?key=YOUR_KEY" -o roboflow.zip
unzip -q roboflow.zip
rm roboflow.zip
```

Each dataset should have a structure similar to:

```text
dataset1/
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

---

## 7. Verify dataset YAML files

Inspect dataset YAML files:

```bash
cat ~/yolo_train/dataset1/data.yaml
cat ~/yolo_train/dataset2/data.yaml
```

For one-class bird detection, the important part should be equivalent to:

```yaml
nc: 1
names: ['bird']
```

Roboflow sometimes uses names such as `Bird`, `birds`, or names from old exports. The model only sees class IDs in label files, but `names:` controls what appears in logs and plots.

---

## 8. Merge two datasets into `dataset_merged`

If you use only one dataset, you can still use this process by placing it in `dataset1` and leaving `dataset2` empty.

```bash
cd ~/yolo_train

D1=/home/jakub-pelka/yolo_train/dataset1
D2=/home/jakub-pelka/yolo_train/dataset2
MERGED=/home/jakub-pelka/yolo_train/dataset_merged

mkdir -p "$MERGED"/{train,valid,test}/{images,labels}
```

Copy files with prefixes to avoid filename collisions:

```bash
copy_with_prefix () {
  SRC="$1"; DST="$2"; PREF="$3"
  [ -d "$SRC" ] || return 0
  find "$SRC" -maxdepth 1 -type f -print0 | while IFS= read -r -d '' f; do
    b=$(basename "$f")
    cp "$f" "$DST/${PREF}_${b}"
  done
}

for SPLIT in train valid test; do
  copy_with_prefix "$D1/$SPLIT/images" "$MERGED/$SPLIT/images" d1
  copy_with_prefix "$D1/$SPLIT/labels" "$MERGED/$SPLIT/labels" d1
done

for SPLIT in train valid test; do
  copy_with_prefix "$D2/$SPLIT/images" "$MERGED/$SPLIT/images" d2
  copy_with_prefix "$D2/$SPLIT/labels" "$MERGED/$SPLIT/labels" d2
done
```

---

## 9. Force all labels to one class: `bird = 0`

This is useful when datasets use different class names but all objects should become one class.

```bash
find "$MERGED" -path "*/labels/*.txt" -type f -print0 \
 | xargs -0 -I{} sed -i -E 's/^[0-9]+ /0 /' {}
```

---

## 10. Clean duplicate and invalid labels

Some Roboflow exports may contain duplicated label rows. This script removes exact duplicate rows and filters invalid YOLO labels.

```bash
python - <<'PY'
import os, glob

base = "/home/jakub-pelka/yolo_train/dataset_merged"

def clean_label_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except Exception:
        return 0, 0

    good = []
    for ln in lines:
        try:
            parts = ln.split()
            if len(parts) != 5:
                continue
            nums = list(map(float, parts[1:]))
            if not all(0.0 <= v <= 1.0 for v in nums):
                continue
            good.append("0 " + " ".join(f"{v:.6f}" for v in nums))
        except Exception:
            continue

    cleaned = sorted(set(good))

    if cleaned != lines:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(cleaned) + ("\n" if cleaned else ""))

    return len(lines), len(cleaned)

changed = 0
before_total = 0
after_total = 0

for split in ("train", "valid", "test"):
    label_dir = os.path.join(base, split, "labels")
    if not os.path.isdir(label_dir):
        continue
    for path in glob.glob(os.path.join(label_dir, "*.txt")):
        before, after = clean_label_file(path)
        if before != after:
            changed += 1
            before_total += before
            after_total += after

print(f"Changed label files: {changed}")
print(f"Rows: {before_total} -> {after_total}")
PY
```

Optional: remove empty label files:

```bash
find "$MERGED" -path "*/labels/*.txt" -size 0 -delete
```

---

## 11. Sanity-check image/label counts

```bash
export MERGED=/home/jakub-pelka/yolo_train/dataset_merged

python - <<'PY'
import os, glob

base = os.environ["MERGED"]

for split in ["train", "valid", "test"]:
    image_dir = f"{base}/{split}/images"
    label_dir = f"{base}/{split}/labels"

    if not os.path.isdir(image_dir):
        print(f"{split}: missing images folder")
        continue

    images = {
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(image_dir + "/*")
    }

    labels = {
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(label_dir + "/*.txt")
    } if os.path.isdir(label_dir) else set()

    print(
        f"{split}: images={len(images)} labels={len(labels)} "
        f"no-label images={len(images - labels)} orphan labels={len(labels - images)}"
    )
PY
```

`orphan labels` should normally be `0`. `no-label images` may exist if you intentionally have background images.

---

## 12. Create local `data.yaml` inside `dataset_merged`

```bash
cd ~/yolo_train/dataset_merged

cat > data.yaml <<'YAML'
path: .
train: train/images
val: valid/images
test: test/images
nc: 1
names: ['bird']
YAML
```

Verify:

```bash
cat data.yaml
```

---

## 13. Quick dataset validation smoke-test

This does not train your custom model. It only checks that the dataset can be read by Ultralytics.

```bash
cd ~/yolo_train/dataset_merged

yolo mode=val \
  model=yolo26n.pt \
  data=data.yaml \
  imgsz=640 batch=2
```

---

## 14. Start training YOLO26

Recommended first serious run:

```bash
cd ~/yolo_train/dataset_merged

yolo task=detect mode=train \
  model=yolo26x.pt \
  data=data.yaml \
  imgsz=640 \
  batch=4 \
  nbs=12 \
  epochs=300 \
  patience=50 \
  workers=8 \
  cache=disk \
  mosaic=1 \
  save_period=5 \
  name=birds_yolo26x_640_b4_nbs12_mos1_e300
```

### What the main parameters mean

- `model=yolo26x.pt` – largest YOLO26 detection model; use `yolo26n/s/m/l.pt` if VRAM is limited.
- `imgsz=640` – training image size.
- `batch=4` – physical batch size; reduce if you get CUDA OOM.
- `nbs=12` – nominal batch size; affects gradient accumulation/scaling.
- `epochs=300` – total training epochs for this run.
- `patience=50` – early stopping after 50 epochs without improvement.
- `cache=disk` – safer for large datasets than RAM cache.
- `mosaic=1` – uses mosaic augmentation during most of the training.
- `save_period=5` – saves checkpoints every 5 epochs.

---

## 15. Monitor GPU usage

Open a second terminal:

```bash
watch -n1 nvidia-smi
```

Optional better monitor:

```bash
sudo apt install -y nvtop
nvtop
```

---

## 16. Prevent Linux sleep during long training

Open a separate terminal and run:

```bash
systemd-inhibit --what=idle:sleep --why="YOLO training" sleep infinity
```

Keep this terminal open while training runs. Stop it with `Ctrl+C` after training.

---

## 17. TensorBoard

Open a new terminal:

```bash
cd ~/yolo_train
source .venv/bin/activate

tensorboard --logdir /home/jakub-pelka/yolo_train/dataset_merged/runs/detect --port 6006
```

Open in browser:

```text
http://localhost:6006
```

If port `6006` is busy:

```bash
tensorboard --logdir /home/jakub-pelka/yolo_train/dataset_merged/runs/detect --port 6007
```

---

## 18. How to pause and resume training

### Pause safely

In the training terminal press:

```text
Ctrl+C
```

Press once only. YOLO should save `weights/last.pt`.

### Resume the same run

Example:

```bash
cd ~/yolo_train/dataset_merged

yolo task=detect mode=train resume=True \
  model=runs/detect/birds_yolo26x_640_b4_nbs12_mos1_e300/weights/last.pt
```

Important:

- Do not pass `name=` or `project=` when resuming the same run.
- Resuming usually uses the original saved training arguments.

---

## 19. Continue after the original epoch limit

If a run reaches `300/300` and you want to continue, the cleanest method is to start a continuation run from the previous `last.pt` or `best.pt`.

Example continuation from `last.pt`:

```bash
cd ~/yolo_train/dataset_merged

yolo task=detect mode=train \
  model=runs/detect/birds_yolo26x_640_b4_nbs12_mos1_e300/weights/last.pt \
  data=data.yaml \
  imgsz=640 \
  batch=4 \
  nbs=32 \
  epochs=100 \
  patience=30 \
  workers=8 \
  cache=disk \
  mosaic=1 \
  save_period=5 \
  name=birds_yolo26x_640_continue_nbs32_e100
```

This creates a new run folder but continues learning from your trained weights.

---

## 20. Fine-tuning / polishing run

If the main training looks good and starts to plateau, you can run a short fine-tune with reduced augmentation and lower learning rate.

```bash
cd ~/yolo_train/dataset_merged

yolo task=detect mode=train \
  model=runs/detect/birds_yolo26x_640_b4_nbs12_mos1_e300/weights/best.pt \
  data=data.yaml \
  imgsz=640 \
  batch=4 \
  nbs=24 \
  epochs=80 \
  patience=25 \
  lr0=0.002 \
  lrf=0.01 \
  mosaic=0 \
  close_mosaic=0 \
  workers=8 \
  cache=disk \
  name=birds_yolo26x_640_finetune_lr002_mos0
```

Use this only after the main run has produced a strong `best.pt`.

---

## 21. Validate the trained model

```bash
cd ~/yolo_train/dataset_merged

yolo mode=val \
  model=runs/detect/birds_yolo26x_640_b4_nbs12_mos1_e300/weights/best.pt \
  data=data.yaml \
  imgsz=640 \
  batch=4 \
  plots=True
```

Optional validation with test-time augmentation:

```bash
yolo mode=val \
  model=runs/detect/birds_yolo26x_640_b4_nbs12_mos1_e300/weights/best.pt \
  data=data.yaml \
  imgsz=640 \
  augment=True
```

---

## 22. Run prediction on validation images

```bash
cd ~/yolo_train/dataset_merged

yolo mode=predict \
  model=runs/detect/birds_yolo26x_640_b4_nbs12_mos1_e300/weights/best.pt \
  source=valid/images \
  imgsz=640 \
  conf=0.25 \
  iou=0.7 \
  save=True
```

Try lower `conf` if you need higher recall:

```bash
conf=0.15
```

Try higher `conf` if you want fewer false positives:

```bash
conf=0.35
```

---

## 23. Export the trained model

### Export to ONNX

```bash
cd ~/yolo_train/dataset_merged

yolo mode=export \
  model=runs/detect/birds_yolo26x_640_b4_nbs12_mos1_e300/weights/best.pt \
  format=onnx \
  opset=17 \
  dynamic=True
```

### Export to TorchScript

```bash
yolo mode=export \
  model=runs/detect/birds_yolo26x_640_b4_nbs12_mos1_e300/weights/best.pt \
  format=torchscript
```

### Native `.pt`

The native PyTorch model is already available here:

```text
runs/detect/birds_yolo26x_640_b4_nbs12_mos1_e300/weights/best.pt
runs/detect/birds_yolo26x_640_b4_nbs12_mos1_e300/weights/last.pt
```

For most Ultralytics-based Python apps, use `best.pt`.

---

## 24. Useful troubleshooting

### `yolo: command not found`

Activate the venv:

```bash
cd ~/yolo_train
source .venv/bin/activate
```

If still missing:

```bash
pip install ultralytics
```

### CUDA out of memory

Try, in this order:

```bash
batch=2
```

or:

```bash
model=yolo26l.pt
```

or:

```bash
model=yolo26m.pt
```

You can also keep `imgsz=640` and reduce `batch`.

### TensorBoard shows no dashboards

Make sure TensorBoard was enabled before training:

```bash
yolo settings tensorboard=True
```

Then start TensorBoard from the correct folder:

```bash
tensorboard --logdir ~/yolo_train/dataset_merged/runs/detect --port 6006
```

### Duplicate labels warning

Clean labels using section 10.

### Wrong class name in validation output

Check:

```bash
cat ~/yolo_train/dataset_merged/data.yaml
```

It should contain:

```yaml
nc: 1
names: ['bird']
```

After training your custom model, validation should show your dataset class name.

---

## 25. Recommended baseline command

Use this as the main command for a fresh training run:

```bash
cd ~/yolo_train/dataset_merged

yolo task=detect mode=train \
  model=yolo26x.pt \
  data=data.yaml \
  imgsz=640 \
  batch=4 \
  nbs=12 \
  epochs=300 \
  patience=50 \
  workers=8 \
  cache=disk \
  mosaic=1 \
  save_period=5 \
  name=birds_yolo26x_640_b4_nbs12_mos1_e300
```

---

## 26. Quick checklist before leaving training overnight

- [ ] Training terminal is active.
- [ ] `watch -n1 nvidia-smi` shows stable GPU use.
- [ ] Sleep blocker is active:

```bash
systemd-inhibit --what=idle:sleep --why="YOLO training" sleep infinity
```

- [ ] TensorBoard is running:

```bash
tensorboard --logdir ~/yolo_train/dataset_merged/runs/detect --port 6006
```

- [ ] Checkpoints are saved every 5 epochs:

```bash
save_period=5
```

- [ ] You know where the model will be:

```text
~/yolo_train/dataset_merged/runs/detect/<run_name>/weights/best.pt
```
