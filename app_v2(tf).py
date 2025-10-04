import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import json
import pandas as pd
from typing import List, Tuple
import time
import io
from collections import Counter
import tensorflow as tf
import tensorflow_hub as hub

# Page configuration
st.set_page_config(
    page_title="Advanced Object Detection Studio",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal CSS (simple white title)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: .2px;
        margin: 0.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# COCO class names (91 classes)
COCO_CLASSES = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack',
    'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table',
    'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

@st.cache_resource(show_spinner=False)
def load_detection_model():
    """Load TensorFlow Hub object detection model"""
    try:
        model_url = "https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1"
        model = hub.load(model_url)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

class ObjectDetectionApp:
    def __init__(self):
        self.model = None

    def detect_objects_tf(self, image_pil: Image.Image, confidence: float = 0.5) -> Tuple[List, List, List, dict]:
            """Detect objects using TF-Hub EfficientDet (works both with dict or tuple outputs)."""
            start_time = time.time()
            try:
                if self.model is None:
                    st.write("Loading detection model...")
                    self.model = load_detection_model()
                if self.model is None:
                    raise Exception("Model failed to load")
        
                # PIL -> float32 [0,1], shape [1,H,W,3]
                img = np.array(image_pil.convert("RGB"), dtype=np.float32) / 255.0
                img = tf.convert_to_tensor(img)[tf.newaxis, ...]  # [1,H,W,3]
        
                # Call either the serving_default fn or the module directly
                outputs = self.model(img)
        
                # Normalize outputs to a single dict of numpy arrays
                if isinstance(outputs, dict):
                    boxes   = outputs.get("detection_boxes",   outputs.get("output_0"))[0].numpy()
                    classes = outputs.get("detection_classes", outputs.get("output_2"))[0].numpy()
                    scores  = outputs.get("detection_scores",  outputs.get("output_1"))[0].numpy()
                else:
                    # Some hubs return a tuple: (boxes, scores, classes, num)
                    boxes  = outputs[0][0].numpy()
                    scores = outputs[1][0].numpy()
                    classes= outputs[2][0].numpy()
        
                H, W = image_pil.size[1], image_pil.size[0]  # (height, width)
                bbox, labels, confidences = [], [], []
        
                for i in range(len(scores)):
                    s = float(scores[i])
                    if s < confidence:
                        continue
        
                    ymin, xmin, ymax, xmax = [float(v) for v in boxes[i]]  # normalized
                    x1 = int(max(0, min(W - 1, xmin * W)))
                    y1 = int(max(0, min(H - 1, ymin * H)))
                    x2 = int(max(0, min(W - 1, xmax * W)))
                    y2 = int(max(0, min(H - 1, ymax * H)))
                    if x2 <= x1 or y2 <= y1:
                        continue
        
                    cid = int(classes[i])
                    # EfficientDet COCO ids are typically 1-based; fall back to 0-based if needed
                    if 0 <= cid < len(COCO_CLASSES):
                        name = COCO_CLASSES[cid]
                    elif 0 <= cid - 1 < len(COCO_CLASSES):
                        name = COCO_CLASSES[cid - 1]
                    else:
                        name = f"class_{cid}"
        
                    bbox.append([x1, y1, x2, y2])
                    labels.append(name)
                    confidences.append(s)
        
                processing_time = time.time() - start_time
                metrics = {
                    "processing_time": processing_time,
                    "total_detections": len(labels),
                    "unique_classes": len(set(labels)) if labels else 0,
                    "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
                    "max_confidence": float(max(confidences)) if confidences else 0.0,
                    "min_confidence": float(min(confidences)) if confidences else 0.0
                }
                return bbox, labels, confidences, metrics
        
            except Exception as e:
                import traceback
                return [], [], [], {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "processing_time": time.time() - start_time
                }


    def apply_image_enhancements(self, image: Image.Image,
                                 brightness: float,
                                 contrast: float,
                                 sharpness: float,
                                 saturation: float = 1.0) -> Image.Image:
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
        except Exception:
            return image

    def filter_detections(self, bbox: List, labels: List, confidences: List,
                          selected_classes: List[str]) -> Tuple[List, List, List]:
        if not selected_classes or not labels:
            return bbox, labels, confidences
        filt_b, filt_l, filt_c = [], [], []
        for i, lbl in enumerate(labels):
            if lbl in selected_classes:
                filt_b.append(bbox[i]); filt_l.append(lbl); filt_c.append(confidences[i])
        return filt_b, filt_l, filt_c

    def create_detection_summary(self, labels: List[str], confidences: List[float]) -> dict:
        if not labels:
            return {"total_objects": 0, "unique_classes": 0, "average_confidence": 0,
                    "class_counts": {}, "max_confidence": 0, "min_confidence": 0}
        counts = Counter(labels)
        return {
            "total_objects": len(labels),
            "unique_classes": len(counts),
            "average_confidence": float(np.mean(confidences)) if confidences else 0,
            "class_counts": dict(counts),
            "max_confidence": float(max(confidences)) if confidences else 0,
            "min_confidence": float(min(confidences)) if confidences else 0
        }

    def export_results_json(self, labels: List[str], confidences: List[float], bboxes: List) -> str:
        rows = []
        for i, label in enumerate(labels):
            bb = bboxes[i] if i < len(bboxes) else None
            if bb is not None:
                bb = [int(x) for x in bb]
            rows.append({"class": label,
                         "confidence": float(confidences[i]) if i < len(confidences) else 0,
                         "bbox": bb})
        return json.dumps(rows, indent=2)

    def export_results_csv(self, labels: List[str], confidences: List[float]) -> str:
        return pd.DataFrame({"Object_Class": labels, "Confidence": confidences}).to_csv(index=False)


def draw_boxes(image_np, boxes, labels, confidences):
    """Draw visible RGB boxes with adaptive thickness."""
    # Ensure uint8 RGB
    if image_np.dtype != np.uint8:
        image = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
    else:
        image = image_np.copy()
    if image.shape[-1] == 3:
        rgb = image
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]
    thickness = max(2, min(h, w) // 200)

    np.random.seed(42)
    palette = {lbl: tuple(int(c) for c in np.random.randint(0, 255, 3)) for lbl in set(labels)}

    for (x1, y1, x2, y2), lbl, conf in zip(boxes, labels, confidences):
        color = palette.get(lbl, (0, 255, 0))
        cv2.rectangle(rgb, (x1, y1), (x2, y2), color, thickness)
        text = f"{lbl}: {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, max(1, thickness - 1))
        cv2.rectangle(rgb, (x1, max(0, y1 - th - 6)), (x1 + tw + 2, y1), color, -1)
        cv2.putText(rgb, text, (x1 + 1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), max(1, thickness - 1))
    return rgb


def main():
    app = ObjectDetectionApp()

    st.markdown('<h1 class="main-header">Advanced Object Detection Studio</h1>', unsafe_allow_html=True)
    st.info("Using TensorFlow Hub EfficientDet (Lite2) — ~90 object classes")

    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

    st.sidebar.header("Detection Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.01, 0.95, 0.5, 0.05)

    st.sidebar.header("Image Enhancement")
    use_enhancement = st.sidebar.checkbox("Enable Enhancement", value=False)
    brightness = contrast = sharpness = saturation = 1.0
    if use_enhancement:
        brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
        contrast   = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
        sharpness  = st.sidebar.slider("Sharpness", 0.5, 2.0, 1.0, 0.1)
        saturation = st.sidebar.slider("Saturation", 0.5, 2.0, 1.0, 0.1)

    st.header("Upload & Process")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        enhanced_image = app.apply_image_enhancements(image, brightness, contrast, sharpness, saturation) if use_enhancement else image

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image)
        with col2:
            st.subheader("Enhanced Image" if use_enhancement else "Processing Image")
            st.image(enhanced_image)

        if st.button("Analyze Image", type="primary"):
            st.markdown("---")
            st.subheader("Detection Process")

            with st.spinner('Detecting objects...'):
                bbox, labels, confidences, metrics = app.detect_objects_tf(enhanced_image, confidence_threshold)

            if "error" in metrics:
                st.error("Detection failed")
                st.error(metrics.get("error"))
                if debug_mode:
                    st.code(metrics.get("traceback", ""))
                return

            if labels:
                st.success(f"Success! Found {len(labels)} objects")

                st.subheader("Detected Objects")
                for idx, (label, conf) in enumerate(zip(labels, confidences)):
                    st.write(f"{idx + 1}. **{label}** (confidence: {conf:.3f})")

                unique_classes = sorted(list(set(labels)))
                selected_classes = st.multiselect("Filter by class:", unique_classes, default=unique_classes)

                filtered_bbox, filtered_labels, filtered_conf = app.filter_detections(bbox, labels, confidences, selected_classes)

                image_np = np.array(enhanced_image)
                output_image = draw_boxes(image_np, filtered_bbox, filtered_labels, filtered_conf)

                st.subheader("Detection Results")
                # IMPORTANT: channels="RGB" to display colors correctly
                st.image(output_image, caption='Detected Objects', channels="RGB")

                stats = app.create_detection_summary(filtered_labels, filtered_conf)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Objects", stats['total_objects'])
                c2.metric("Unique Classes", stats['unique_classes'])
                c3.metric("Avg Confidence", f"{stats['average_confidence']:.2f}")
                c4.metric("Processing Time", f"{metrics['processing_time']:.2f}s")

                if debug_mode:
                    st.write("n_boxes:", len(filtered_bbox))
                    if filtered_bbox:
                        st.write("sample box:", filtered_bbox[0], "label:", filtered_labels[0], "conf:", filtered_conf[0])

                st.subheader("Export Results")
                ec1, ec2, ec3 = st.columns(3)
                with ec1:
                    json_data = app.export_results_json(filtered_labels, filtered_conf, filtered_bbox)
                    st.download_button("Download JSON", json_data, "results.json", "application/json")
                with ec2:
                    csv_data = app.export_results_csv(filtered_labels, filtered_conf)
                    st.download_button("Download CSV", csv_data, "results.csv", "text/csv")
                with ec3:
                    buf = io.BytesIO()
                    Image.fromarray(output_image).save(buf, format='PNG')
                    st.download_button("Download Image", buf.getvalue(), "result.png", "image/png")
            else:
                st.warning("No objects detected. Try lowering the confidence or a different image.")
    else:
        st.markdown("""
        ## Welcome!
        1. Upload an image  
        2. Adjust confidence threshold  
        3. Click **Analyze Image**  
        4. View & export results
        """)

if __name__ == "__main__":
    main()
