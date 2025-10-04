# streamlit run "c:/ML stuff/ObjectDetectiontask/app_v2.py"

import streamlit as st
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from PIL import Image, ImageEnhance
import numpy as np
import json
import pandas as pd
from typing import List, Tuple, Optional
import time
import io
from collections import Counter

# ---------- NEW: auto-download YOLOv3-tiny on first load ----------
import urllib.request, ssl
from pathlib import Path

@st.cache_resource
def ensure_yolov3_tiny_in_cache():
    """
    Fetch yolov3-tiny.cfg and yolov3-tiny.weights into cvlib's cache:
      ~/.cvlib/object_detection/yolo/
    Runs once per server session (cached), so subsequent uses are fast.
    """
    cache_dir = Path.home() / ".cvlib" / "object_detection" / "yolo"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cache_dir / "yolov3-tiny.cfg"
    wts_path = cache_dir / "yolov3-tiny.weights"

    CFG_URLS = [
        "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-tiny.cfg",
        "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",
    ]
    WTS_URLS = [
        "https://github.com/AlexeyAB/darknet/releases/download/yolov3/yolov3-tiny.weights",
        "https://pjreddie.com/media/files/yolov3-tiny.weights",
    ]

    ctx = ssl.create_default_context()
    headers = {"User-Agent": "Mozilla/5.0"}

    def fetch(urls, dest: Path, min_bytes: int, label: str):
        if dest.exists() and dest.stat().st_size >= min_bytes:
            return
        for u in urls:
            try:
                req = urllib.request.Request(u, headers=headers)
                with urllib.request.urlopen(req, context=ctx) as r, open(dest, "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)  # 1 MB chunks
                        if not chunk:
                            break
                        f.write(chunk)
                if dest.stat().st_size >= min_bytes:
                    return
            except Exception:
                time.sleep(1)
        raise RuntimeError(f"Failed to fetch {label}. Tried: {urls}")

    # cfg ≈ 9 KB
    fetch(CFG_URLS, cfg_path, min_bytes=5_000, label="yolov3-tiny.cfg")
    # weights ≈ 33 MB
    st.info("Preparing YOLOv3-tiny weights (~33 MB) on first run…")
    fetch(WTS_URLS, wts_path, min_bytes=10_000_000, label="yolov3-tiny.weights")
    st.success("YOLOv3-tiny model is ready ✅")

# run once on import (cached)
try:
    ensure_yolov3_tiny_in_cache()
except Exception as e:
    st.error(str(e))
    st.stop()
# ---------------------------------------------------------------

# Page configuration
st.set_page_config(
    page_title="Advanced Object Detection Studio",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .stats-container { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem; border-radius: 10px; color: white; margin: 1rem 0; }
    .detection-card { background: #f8f9fa; padding: 1rem; border-radius: 8px;
        border-left: 4px solid #667eea; margin: 0.5rem 0; }
    .sidebar-section { background: #f0f2f6; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
    .metric-card { background: white; padding: 1rem; border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; margin: 0.5rem 0; }
    .debug-info { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px;
        padding: 1rem; margin: 1rem 0; font-family: monospace; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

class ObjectDetectionApp:
    def __init__(self):
        # Fixed to YOLOv3-tiny (small + fast for cloud)
        self.model_name = "yolov3-tiny"

    def apply_image_enhancements(self, image: Image.Image, brightness: float,
                                 contrast: float, sharpness: float, saturation: float = 1.0) -> Image.Image:
        try:
            enhanced = image.copy()
            if brightness != 1.0:
                enhanced = ImageEnhance.Brightness(enhanced).enhance(brightness)
            if contrast != 1.0:
                enhanced = ImageEnhance.Contrast(enhanced).enhance(contrast)
            if sharpness != 1.0:
                enhanced = ImageEnhance.Sharpness(enhanced).enhance(sharpness)
            if saturation != 1.0:
                enhanced = ImageEnhance.Color(enhanced).enhance(saturation)
            return enhanced
        except Exception as e:
            st.error(f"Error enhancing image: {str(e)}")
            return image

    def detect_objects_simple(self, image_np: np.ndarray, confidence: float = 0.3) -> Tuple[List, List, List, dict]:
        start_time = time.time()
        try:
            # Use cvlib with YOLOv3-tiny (weights/cfg already cached by ensure_yolov3_tiny_in_cache)
            bbox, label, conf = cv.detect_common_objects(
                image_np, confidence=confidence, model=self.model_name
            )
            processing_time = time.time() - start_time
            metrics = {
                "processing_time": processing_time,
                "total_detections": len(label),
                "unique_classes": len(set(label)) if label else 0,
                "avg_confidence": float(np.mean(conf)) if conf else 0.0,
                "max_confidence": float(max(conf)) if conf else 0.0,
                "min_confidence": float(min(conf)) if conf else 0.0
            }
            return bbox, label, conf, metrics
        except Exception as e:
            st.error(f"🚨 Detection error: {str(e)}")
            import traceback
            st.error(f"Full error traceback: {traceback.format_exc()}")
            return [], [], [], {"error": str(e), "processing_time": time.time() - start_time}

    def filter_detections(self, bbox: List, labels: List, confidences: List,
                          selected_classes: List[str]) -> Tuple[List, List, List]:
        if not selected_classes or not labels:
            return bbox, labels, confidences
        filtered_bbox, filtered_label, filtered_conf = [], [], []
        for i, lbl in enumerate(labels):
            if lbl in selected_classes:
                filtered_bbox.append(bbox[i])
                filtered_label.append(labels[i])
                filtered_conf.append(confidences[i])
        return filtered_bbox, filtered_label, filtered_conf

    def create_detection_summary(self, labels: List[str], confidences: List[float]) -> dict:
        if not labels:
            return {
                "total_objects": 0, "unique_classes": 0, "average_confidence": 0,
                "class_counts": {}, "max_confidence": 0, "min_confidence": 0
            }
        label_counts = Counter(labels)
        avg_confidence = float(np.mean(confidences)) if confidences else 0
        return {
            "total_objects": len(labels),
            "unique_classes": len(label_counts),
            "average_confidence": avg_confidence,
            "class_counts": dict(label_counts),
            "max_confidence": float(max(confidences)) if confidences else 0,
            "min_confidence": float(min(confidences)) if confidences else 0
        }

    def export_results_json(self, labels: List[str], confidences: List[float], bboxes: List) -> str:
        results = []
        for i, label in enumerate(labels):
            results.append({
                "class": label,
                "confidence": float(confidences[i]) if i < len(confidences) else 0,
                "bbox": bboxes[i] if i < len(bboxes) else None
            })
        return json.dumps(results, indent=2)

    def export_results_csv(self, labels: List[str], confidences: List[float]) -> str:
        df = pd.DataFrame({"Object_Class": labels, "Confidence": confidences})
        return df.to_csv(index=False)

def main():
    app = ObjectDetectionApp()

    # Header
    st.markdown('<h1 class="main-header">🔍 Advanced Object Detection Studio</h1>', unsafe_allow_html=True)
    st.caption("Model: **YOLOv3-tiny** · First load will fetch the model, then it’s cached.")

    # Debug mode toggle
    debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=False, help="Show detailed debugging information")

    # Sidebar configuration
    st.sidebar.header("⚙️ Detection Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", min_value=0.01, max_value=0.95, value=0.5, step=0.05,
        help="Lower values = more detections (try 0.1 for maximum sensitivity)"
    )

    # Image enhancement settings
    st.sidebar.header("🎨 Image Enhancement")
    use_enhancement = st.sidebar.checkbox("Enable Image Enhancement", value=False)
    brightness = contrast = sharpness = saturation = 1.0
    if use_enhancement:
        brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
        contrast   = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
        sharpness  = st.sidebar.slider("Sharpness", 0.5, 2.0, 1.0, 0.1)
        saturation = st.sidebar.slider("Saturation", 0.5, 2.0, 1.0, 0.1)

    # Main content area
    st.header("📁 Upload & Process")
    uploaded_file = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png"], help="Upload a single image for testing"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        enhanced_image = app.apply_image_enhancements(
            image, brightness, contrast, sharpness, saturation
        ) if use_enhancement else image

        image_np = np.array(enhanced_image)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
            if debug_mode:
                st.markdown(f"""
                <div class="debug-info">
                <strong>Image Debug Info:</strong><br>
                Size: {image.width} x {image.height}<br>
                Mode: {image.mode}<br>
                Array shape: {np.array(image).shape}
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.subheader("Enhanced Image" if use_enhancement else "Processing Image")
            st.image(enhanced_image, use_column_width=True)
            if debug_mode:
                st.markdown(f"""
                <div class="debug-info">
                <strong>Processing Debug Info:</strong><br>
                Enhancement: {use_enhancement}<br>
                Array shape: {image_np.shape}<br>
                Data type: {image_np.dtype}<br>
                Min/Max values: {image_np.min()}/{image_np.max()}
                </div>
                """, unsafe_allow_html=True)

        # Detection button
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            st.markdown("---")
            st.subheader("🔄 Detection Process")

            with st.spinner('Detecting objects…'):
                bbox, labels, confidences, metrics = app.detect_objects_simple(
                    image_np, confidence_threshold
                )

            if labels:
                st.success(f"🎉 Success! Found {len(labels)} objects")

                st.subheader("🏷️ Detected Objects")
                for idx, (label, conf) in enumerate(zip(labels, confidences), start=1):
                    st.write(f"{idx}. **{label}** (confidence: {conf:.3f})")

                unique_classes = sorted(set(labels))
                selected_classes = st.multiselect(
                    "🎯 Filter by object classes:",
                    unique_classes,
                    default=unique_classes,
                    help="Select which object classes to display"
                )

                filtered_bbox, filtered_labels, filtered_conf = app.filter_detections(
                    bbox, labels, confidences, selected_classes
                )

                try:
                    output_image = draw_bbox(image_np, filtered_bbox, filtered_labels, filtered_conf)
                    st.subheader("🎯 Detection Results")
                    st.image(output_image, caption='Detected Objects', use_column_width=True)

                    stats = app.create_detection_summary(filtered_labels, filtered_conf)
                    col1, col2, col3, col4 = st.columns(4)
                    with col1: st.metric("Total Objects", stats['total_objects'])
                    with col2: st.metric("Unique Classes", stats['unique_classes'])
                    with col3: st.metric("Avg Confidence", f"{stats['average_confidence']:.2f}")
                    with col4: st.metric("Processing Time", f"{metrics['processing_time']:.2f}s")

                    st.subheader("💾 Export Results")
                    export_col1, export_col2, export_col3 = st.columns(3)
                    with export_col1:
                        json_data = app.export_results_json(filtered_labels, filtered_conf, filtered_bbox)
                        st.download_button("📄 Download JSON", data=json_data,
                                           file_name="detection_results.json", mime="application/json")
                    with export_col2:
                        csv_data = app.export_results_csv(filtered_labels, filtered_conf)
                        st.download_button("📊 Download CSV", data=csv_data,
                                           file_name="detection_results.csv", mime="text/csv")
                    with export_col3:
                        img_buffer = io.BytesIO()
                        Image.fromarray(output_image).save(img_buffer, format='PNG')
                        st.download_button("🖼️ Download Image", data=img_buffer.getvalue(),
                                           file_name="processed_image.png", mime="image/png")
                except Exception as e:
                    st.error(f"Error drawing bounding boxes: {str(e)}")
                    st.write("Raw detection results:")
                    st.write(f"Bounding boxes: {bbox}")
                    st.write(f"Labels: {labels}")
                    st.write(f"Confidences: {confidences}")
            else:
                st.warning("⚠️ No objects detected in the image.")
                st.markdown("""
                ### 🔧 Troubleshooting Tips:
                1. Lower the confidence threshold (try 0.1 or 0.2)
                2. Try a different image with clear, common objects
                3. Disable image enhancement if the image becomes too contrasted
                """)

    else:
        st.markdown("""
        ## 👋 Welcome to Advanced Object Detection Studio (auto-fetch YOLOv3-tiny)
        1. Upload an image
        2. Adjust the confidence threshold
        3. Click **Analyze Image**
        """)

if __name__ == "__main__":
    main()
