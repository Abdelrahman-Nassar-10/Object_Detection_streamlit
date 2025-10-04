# 🔍 Object Detection Studio

A web app for image object detection with two flavors:

- **Local (YOLOv3 / YOLOv3-tiny via CVLib + OpenCV)** — best when running on your machine.
- **Cloud (TensorFlow Hub EfficientDet on Streamlit Cloud)** — no big weights to store; model is fetched at runtime.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![CVLib](https://img.shields.io/badge/CVLib-0.2.7%2B-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%20~%202.16-ff6f00)

---

## ✨ Features (both versions)

- **80+ COCO classes** (people, vehicles, animals, furniture, etc.)
- **Adjustable confidence** (0.01–0.95)
- **Image enhancements** (brightness, contrast, sharpness, saturation)
- **Analytics** (processing time, per-class counts, avg confidence)
- **Filtering** by detected class
- **Export** results as **JSON**, **CSV**, and annotated **PNG**

---

## 🧭 Which version should I use?

| Use case | Recommended version |
|---|---|
| Quick local experiments, full OpenCV/CVLib stack | **YOLO (local)** |
| Easy web demo with a shareable link (no big weights) | **TensorFlow Hub (Streamlit Cloud)** |
| Limited bandwidth / can’t host big `.weights` | **TensorFlow Hub (Cloud)** |
| Custom YOLO configs/weights you already have | **YOLO (local)** |

---

## 📦 Repo layout (suggested)

```
.
├─ app_yolo.py                # YOLO (local) app
├─ app_tf.py                  # TensorFlow Hub (cloud) app
├─ requirements.txt           # Pinned deps for chosen target (see below)
├─ runtime.txt                # Pin Python on Streamlit Cloud (e.g., python-3.11)
├─ models/
│  ├─ yolov3.cfg              # (optional) local YOLO config
│  ├─ yolov3.weights          # (optional) local YOLO weights (use Git LFS if large)
│  └─ yolov3-tiny.cfg/weights # (optional) tiny versions
└─ README.md
```

> You can keep just one app file if you prefer. The key is: **YOLO for local**, **TF-Hub for Cloud**.

---

## 🚀 Quick Start — YOLO (Local)

> Uses **cvlib** + **OpenCV**. If you use YOLOv3/YOLOv3-tiny weights locally, place them under `models/` or let `cvlib` download them to its cache on first run.

1) Create & activate a virtual env
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

2) Install dependencies (YOLO local set)
```bash
pip install streamlit==1.28.2 opencv-python==4.8.0.74 cvlib==0.2.7 pillow==9.5.0 numpy==1.24.3 pandas==2.0.3
```

3) Run
```bash
streamlit run app_yolo.py
```

4) Open `http://localhost:8501` and upload an image.

> **Tip:** If you keep `.weights` in git, use **Git LFS**:
```bash
git lfs install
git lfs track "*.weights"
git add .gitattributes models/yolov3*.weights
git commit -m "track YOLO weights via LFS"
```

---

## ☁️ Deploy — TensorFlow Hub (Streamlit Cloud)

> The TF version **downloads the model from TF-Hub on first run** and caches it. No local weights required.

### Files for Cloud
- **`app_tf.py`** (your TF-Hub EfficientDet app)
- **`requirements.txt`** (choose one set below and keep it in repo root)
- **`runtime.txt`** to pin Python (recommended)

### Recommended pins (Python **3.11** + TF **2.16.1**)
**`runtime.txt`**
```
python-3.11
```

**`requirements.txt`**
```
streamlit==1.28.2
opencv-python-headless==4.8.0.74

tensorflow-cpu==2.16.1
tensorflow-hub==0.16.1

numpy==1.26.4
Pillow==10.3.0
pandas==2.2.2
```

> Alternatively (Python **3.10** + TF **2.13.0**):
> ```
> python-3.10
> ```
> ```
> streamlit==1.28.2
> opencv-python-headless==4.8.0.74
> tensorflow-cpu==2.13.0
> tensorflow-hub==0.14.0
> numpy==1.24.3
> Pillow==9.5.0
> pandas==2.0.3
> ```

### Deploy steps
1) Push your repo to GitHub (root must contain `app_tf.py`, `requirements.txt`, and (optionally) `runtime.txt`).
2) On **share.streamlit.io**, select your repo + branch + **main module: `app_tf.py`**.
3) After first boot, click **“Rerun”** if the TF-Hub model just finished caching.

---

## 🖱️ Usage (both)

1. Upload an image (JPG/PNG).  
2. Adjust the **Confidence Threshold**.  
3. *(Optional)* Enable **Image Enhancement** (brightness/contrast/sharpness/saturation).  
4. Click **Analyze Image**.  
5. Review detections, filter classes, and **Export** JSON/CSV/annotated PNG.

---

## 🛠️ Technical Details

- **YOLO (local)**: `cvlib.detect_common_objects` (OpenCV DNN under the hood); can work with **YOLOv3** and **YOLOv3-tiny**.  
- **TensorFlow (cloud)**: **EfficientDet Lite** from **TensorFlow Hub** (returns either dict or tuple; the app handles both shapes).  
- **Image IO**: PIL (Pillow).  
- **Visualization**: OpenCV rectangles + class labels.

---

## 🧩 Troubleshooting

- **Streamlit Cloud “Error installing requirements”**
  - Mismatch between Python & TensorFlow. Use **`runtime.txt`** + the pinned **`requirements.txt`** above.
  - Use **`opencv-python-headless`** on Cloud (no display server).

- **Text header invisible**
  - Remove CSS gradient or force a visible color (the README’s TF app shows a simple white title).

- **Big files rejected by GitHub**
  - Configure **Git LFS** for `.weights` or avoid committing them (TF-Hub version doesn’t need weights).

---

## 🧪 Performance quick guide

| Model | Speed | Accuracy | Best for |
|---|---|---|---|
| YOLOv3 | Medium | High | Local use, higher accuracy |
| YOLOv3-tiny | Fast | Good | Local real-time/smaller model |
| EfficientDet-Lite (TF-Hub) | Fast | Good | Streamlit Cloud, no large weights |

---

## 🙏 Acknowledgments

- **CVLib**, **OpenCV** (YOLO pipeline)
- **TensorFlow** & **TensorFlow Hub** (EfficientDet models)
- **COCO Dataset** (class labels)
- **Streamlit** (UI framework)

---

⭐ If this project helped you, **star** the repo!

**Made by Abdelrahman Nassar**
