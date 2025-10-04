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
    page_title="Object Detection Studio",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color:#ffffff; 
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
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

# --- NEW: universal unpacker for TF-Hub outputs ------------------------------------
def _unpack_detections(outputs):  # <<< CHANGED (new helper)
    """
    Supports TF-Hub object detection modules that return either:
      - dict with keys like 'detection_boxes', 'detection_scores', 'detection_classes'
      - tuple in the order (boxes, scores, classes, num_detections)
    Returns numpy arrays: boxes, scores, classes for batch 0.
    """
    # EagerTensor or dict?
    if isinstance(outputs, dict):
        boxes = outputs["detection_boxes"][0].numpy()
        scores = outputs["detection_scores"][0].numpy()
        # some hubs use 'detection_class_entities' (strings) instead of ids
        classes_tensor = outputs.get("detection_classes", outputs.get("detection_class_entities"))[0]
        classes = classes_tensor.numpy() if hasattr(classes_tensor, "numpy") else classes_tensor
    else:
        # Treat like tuple: (boxes, scores, classes, num_detections)
        boxes = outputs[0][0].numpy()
        scores = outputs[1][0].numpy()
        classes_tensor = outputs[2][0]
        classes = classes_tensor.numpy() if hasattr(classes_tensor, "numpy") else classes_tensor
    return boxes, scores, classes
# -----------------------------------------------------------------------------------

class ObjectDetectionApp:
    def __init__(self):
        self.model = None
        
    def detect_objects_tf(self, image_pil: Image.Image, confidence: float = 0.5) -> Tuple[List, List, List, dict]:
        """Detect objects using TensorFlow Hub model"""
        start_time = time.time()
        
        try:
            if self.model is None:
                st.write("Loading detection model...")
                self.model = load_detection_model()
            if self.model is None:
                raise Exception("Model failed to load")
            
            st.write("Running object detection...")
            
            # Convert PIL to numpy
            img_array = np.array(image_pil)
            if img_array.dtype != np.uint8:
                img_array = (img_array * 255).astype(np.uint8)
            
            # Add batch dimension
            img_tensor = tf.convert_to_tensor(img_array)
            img_tensor = tf.expand_dims(img_tensor, 0)
            
            # Run detection
            outputs = self.model(img_tensor)  # <<< CHANGED (renamed results -> outputs)

            # Robustly extract (boxes, scores, classes) no matter the return type
            detection_boxes, detection_scores, detection_classes = _unpack_detections(outputs)  # <<< CHANGED

            # Convert to image coordinates
            height, width = img_array.shape[:2]
            bbox, labels, confidences = [], [], []
            
            for i in range(len(detection_scores)):
                if float(detection_scores[i]) >= confidence:
                    ymin, xmin, ymax, xmax = detection_boxes[i]
                    x1 = int(xmin * width)
                    y1 = int(ymin * height)
                    x2 = int(xmax * width)
                    y2 = int(ymax * height)
                    bbox.append([x1, y1, x2, y2])

                    # classes can be ints (ids) OR strings (entity names)
                    cls = detection_classes[i]
                    if isinstance(cls, (bytes, bytearray)):
                        lbl = cls.decode("utf-8")
                    elif isinstance(cls, (np.str_, str)):
                        lbl = str(cls)
                    else:
                        cls_id = int(cls)
                        lbl = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
                    labels.append(lbl)
                    confidences.append(float(detection_scores[i]))
            
            processing_time = time.time() - start_time
            st.write(f"Detection completed in {processing_time:.2f}s")
            st.write(f"Found {len(labels)} objects")
            
            metrics = {
                "processing_time": processing_time,
                "total_detections": len(labels),
                "unique_classes": len(set(labels)) if labels else 0,
                "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
                "max_confidence": max(confidences) if confidences else 0.0,
                "min_confidence": min(confidences) if confidences else 0.0
            }
            return bbox, labels, confidences, metrics
            
        except Exception as e:
            import traceback
            st.error(f"Detection error: {str(e)}")
            return [], [], [], {"error": str(e), "traceback": traceback.format_exc(), "processing_time": time.time() - start_time}
    
    def apply_image_enhancements(self, image: Image.Image, brightness: float, 
                                contrast: float, sharpness: float, saturation: float = 1.0) -> Image.Image:
        """Apply image enhancements"""
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
    
    def filter_detections(self, bbox: List, labels: List, confidences: List,
                         selected_classes: List[str]) -> Tuple[List, List, List]:
        """Filter detections"""
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
        """Create statistics"""
        if not labels:
            return {
                "total_objects": 0,
                "unique_classes": 0,
                "average_confidence": 0,
                "class_counts": {},
                "max_confidence": 0,
                "min_confidence": 0
            }
        label_counts = Counter(labels)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        return {
            "total_objects": len(labels),
            "unique_classes": len(label_counts),
            "average_confidence": avg_confidence,
            "class_counts": dict(label_counts),
            "max_confidence": max(confidences) if confidences else 0.0,
            "min_confidence": min(confidences) if confidences else 0.0
        }
    
    def export_results_json(self, labels: List[str], confidences: List[float], bboxes: List) -> str:
        """Export as JSON"""
        results = []
        for i, label in enumerate(labels):
            bbox = bboxes[i] if i < len(bboxes) else None
            if bbox is not None:
                bbox = [int(x) for x in bbox]
            results.append({
                "class": label,
                "confidence": float(confidences[i]) if i < len(confidences) else 0.0,
                "bbox": bbox
            })
        return json.dumps(results, indent=2)
    
    def export_results_csv(self, labels: List[str], confidences: List[float]) -> str:
        """Export as CSV"""
        df = pd.DataFrame({"Object_Class": labels, "Confidence": confidences})
        return df.to_csv(index=False)

def draw_boxes(image_np, boxes, labels, confidences):
    """Draw bounding boxes"""
    image = image_np.copy()
    np.random.seed(42)
    unique_labels = list(set(labels))
    colors = {label: tuple(map(int, np.random.randint(0, 255, 3))) for label in unique_labels}
    for box, label, conf in zip(boxes, labels, confidences):
        x1, y1, x2, y2 = map(int, box)
        color = colors.get(label, (0, 255, 0))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label_text = f"{label}: {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return image

def main():
    app = ObjectDetectionApp()
    st.markdown('<h1 class="main-header">Advanced Object Detection Studio</h1>', unsafe_allow_html=True)
    st.info("Using TensorFlow Hub EfficientDet - Detects 90+ object classes")
    
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
    st.sidebar.header("Detection Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.01, 0.95, 0.5, 0.05)
    
    st.sidebar.header("Image Enhancement")
    use_enhancement = st.sidebar.checkbox("Enable Enhancement", value=False)
    brightness = contrast = sharpness = saturation = 1.0
    if use_enhancement:
        brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
        contrast   = st.sidebar.slider("Contrast",   0.5, 2.0, 1.0, 0.1)
        sharpness  = st.sidebar.slider("Sharpness",  0.5, 2.0, 1.0, 0.1)
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
            elif labels:
                st.success(f"Success! Found {len(labels)} objects")
                st.subheader("Detected Objects")
                for idx, (label, conf) in enumerate(zip(labels, confidences)):
                    st.write(f"{idx + 1}. **{label}** (confidence: {conf:.3f})")
                
                unique_classes = sorted(list(set(labels)))
                selected_classes = st.multiselect("Filter by class:", unique_classes, default=unique_classes)
                filtered_bbox, filtered_labels, filtered_conf = app.filter_detections(bbox, labels, confidences, selected_classes)
                
                try:
                    image_np = np.array(enhanced_image)
                    output_image = draw_boxes(image_np, filtered_bbox, filtered_labels, filtered_conf)
                    st.subheader("Detection Results")
                    st.image(output_image, caption='Detected Objects')
                    
                    stats = app.create_detection_summary(filtered_labels, filtered_conf)
                    c1, c2, c3, c4 = st.columns(4)
                    with c1: st.metric("Total Objects", stats['total_objects'])
                    with c2: st.metric("Unique Classes", stats['unique_classes'])
                    with c3: st.metric("Avg Confidence", f"{stats['average_confidence']:.2f}")
                    with c4: st.metric("Processing Time", f"{metrics['processing_time']:.2f}s")
                    
                    st.subheader("Export Results")
                    e1, e2, e3 = st.columns(3)
                    with e1:
                        json_data = app.export_results_json(filtered_labels, filtered_conf, filtered_bbox)
                        st.download_button("Download JSON", json_data, "results.json", "application/json")
                    with e2:
                        csv_data = app.export_results_csv(filtered_labels, filtered_conf)
                        st.download_button("Download CSV", csv_data, "results.csv", "text/csv")
                    with e3:
                        img_buffer = io.BytesIO()
                        Image.fromarray(output_image).save(img_buffer, format='PNG')
                        st.download_button("Download Image", img_buffer.getvalue(), "result.png", "image/png")
                except Exception as e:
                    st.error(f"Error drawing boxes: {str(e)}")
            else:
                st.warning("No objects detected")
                st.markdown("""
                ### Troubleshooting:
                1. Lower the confidence threshold (try 0.1-0.3)
                2. Try a different image with clear objects
                3. Ensure good lighting and clear visibility
                """)
    else:
        st.markdown("""
        ## Welcome!
        ### Quick Start:
        1. Upload an image
        2. Adjust confidence threshold  
        3. Click "Analyze Image"
        4. View and export results
        """)

if __name__ == "__main__":
    main()
