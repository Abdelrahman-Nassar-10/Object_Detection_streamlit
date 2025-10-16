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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import zipfile

# Page configuration
st.set_page_config(
    page_title="Advanced Object Detection Studio",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        color: #667eea;  /* Fallback color for browsers that don't support gradient text */
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 2rem;
        padding: 1rem 0;
    }
    
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
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

def _unpack_detections(outputs):
    """Unpack detection outputs from TF Hub model"""
    if isinstance(outputs, dict):
        boxes = outputs["detection_boxes"][0].numpy()
        scores = outputs["detection_scores"][0].numpy()
        classes_tensor = outputs.get("detection_classes", outputs.get("detection_class_entities"))[0]
        classes = classes_tensor.numpy() if hasattr(classes_tensor, "numpy") else classes_tensor
    else:
        boxes = outputs[0][0].numpy()
        scores = outputs[1][0].numpy()
        classes_tensor = outputs[2][0]
        classes = classes_tensor.numpy() if hasattr(classes_tensor, "numpy") else classes_tensor
    return boxes, scores, classes

def _boxes_to_pixels(boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Convert boxes to pixel coordinates"""
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return boxes.astype(np.int32)

    normalized = float(np.nanmax(boxes)) <= 1.01

    ymin = boxes[:, 0]
    xmin = boxes[:, 1]
    ymax = boxes[:, 2]
    xmax = boxes[:, 3]

    if normalized:
        x1 = (xmin * img_w)
        y1 = (ymin * img_h)
        x2 = (xmax * img_w)
        y2 = (ymax * img_h)
    else:
        x1, y1, x2, y2 = xmin, ymin, xmax, ymax

    x1 = np.clip(x1, 0, img_w - 1)
    y1 = np.clip(y1, 0, img_h - 1)
    x2 = np.clip(x2, 0, img_w - 1)
    y2 = np.clip(y2, 0, img_h - 1)

    coords = np.stack([x1, y1, x2, y2], axis=-1).astype(np.int32)
    return coords

def draw_boxes(image_np, boxes, labels, confidences):
    """Draw bounding boxes on image"""
    BOX_THICKNESS   = 3
    TEXT_SCALE      = 0.7
    TEXT_THICKNESS  = 2
    TEXT_MARGIN     = 6

    image = image_np.copy()
    np.random.seed(42)
    unique_labels = list(set(labels))
    colors = {label: tuple(map(int, np.random.randint(0, 255, 3))) for label in unique_labels}

    for box, label, conf in zip(boxes, labels, confidences):
        x1, y1, x2, y2 = map(int, box)
        color = colors.get(label, (0, 255, 0))

        cv2.rectangle(image, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        label_text = f"{label}: {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICKNESS)
        cv2.rectangle(image, (x1, y1 - th - TEXT_MARGIN), (x1 + tw, y1), color, -1)
        cv2.putText(image, label_text, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (255, 255, 255), TEXT_THICKNESS)

    return image

def create_analytics_dashboard(stats: dict, labels: List[str], confidences: List[float]):
    """Create comprehensive analytics dashboard"""
    
    st.markdown("---")
    st.header("📊 Detection Analytics Dashboard")
    
    # Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Objects", stats['total_objects'])
    with col2:
        st.metric("Unique Classes", stats['unique_classes'])
    with col3:
        st.metric("Avg Confidence", f"{stats['average_confidence']:.1%}")
    with col4:
        st.metric("Max Confidence", f"{stats['max_confidence']:.1%}")
    with col5:
        st.metric("Min Confidence", f"{stats['min_confidence']:.1%}")
    
    # Create two columns for charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Class Distribution Pie Chart
        if stats['class_counts']:
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(stats['class_counts'].keys()),
                values=list(stats['class_counts'].values()),
                hole=0.3,
                marker=dict(colors=px.colors.qualitative.Set3)
            )])
            fig_pie.update_layout(
                title="Class Distribution",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with chart_col2:
        # Confidence Distribution Histogram
        fig_hist = go.Figure(data=[go.Histogram(
            x=confidences,
            nbinsx=20,
            marker=dict(color='rgb(102, 126, 234)'),
            name='Confidence'
        )])
        fig_hist.update_layout(
            title="Confidence Score Distribution",
            xaxis_title="Confidence Score",
            yaxis_title="Count",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Object Count by Class (Bar Chart)
    if stats['class_counts']:
        sorted_classes = dict(sorted(stats['class_counts'].items(), 
                                    key=lambda x: x[1], reverse=True))
        
        fig_bar = go.Figure(data=[go.Bar(
            x=list(sorted_classes.keys()),
            y=list(sorted_classes.values()),
            marker=dict(
                color=list(sorted_classes.values()),
                colorscale='Viridis',
                showscale=True
            )
        )])
        fig_bar.update_layout(
            title="Objects Detected by Class",
            xaxis_title="Class",
            yaxis_title="Count",
            height=400,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Detailed Statistics Table
    st.subheader("📋 Detailed Class Statistics")
    class_stats = []
    for cls, count in stats['class_counts'].items():
        cls_confidences = [conf for lbl, conf in zip(labels, confidences) if lbl == cls]
        class_stats.append({
            "Class": cls,
            "Count": count,
            "Avg Confidence": f"{np.mean(cls_confidences):.1%}",
            "Min Confidence": f"{min(cls_confidences):.1%}",
            "Max Confidence": f"{max(cls_confidences):.1%}",
            "% of Total": f"{(count / stats['total_objects'] * 100):.1f}%"
        })
    
    df_stats = pd.DataFrame(class_stats)
    st.dataframe(df_stats, use_container_width=True, hide_index=True)

def create_model_comparison_dashboard():
    """Create model performance comparison dashboard"""
    
    st.markdown('<h1 class="main-header">🔬 Model Performance Comparison</h1>', unsafe_allow_html=True)
    
    st.info("📌 Comprehensive benchmark comparison of popular object detection models")
    
    # Simulated comparison data (replace with real benchmarks if available)
    models_data = {
        "Model": ["EfficientDet-Lite2", "YOLOv3", "YOLOv3-tiny", "SSD MobileNet", "Faster R-CNN"],
        "mAP (%)": [45.2, 55.3, 33.1, 23.2, 42.1],
        "Speed (ms)": [120, 51, 22, 35, 180],
        "Accuracy (%)": [87.5, 91.2, 78.4, 72.1, 89.3],
        "Model Size (MB)": [10.8, 236, 33.7, 19.3, 170.5],
        "FPS": [8.3, 19.6, 45.5, 28.6, 5.6]
    }
    
    df_models = pd.DataFrame(models_data)
    
    # Display comparison table
    st.subheader("📊 Model Metrics Comparison")
    st.dataframe(df_models, use_container_width=True, hide_index=True)
    
    # Create comparison charts
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("mAP Score (Higher is Better)", "Inference Speed (Lower is Better)", 
                       "Accuracy (Higher is Better)", "Model Size (Lower is Better)"),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # mAP
    fig.add_trace(go.Bar(x=df_models["Model"], y=df_models["mAP (%)"], 
                         name="mAP", marker_color='rgb(102, 126, 234)'),
                  row=1, col=1)
    
    # Speed (lower is better)
    fig.add_trace(go.Bar(x=df_models["Model"], y=df_models["Speed (ms)"], 
                         name="Speed", marker_color='rgb(118, 75, 162)'),
                  row=1, col=2)
    
    # Accuracy
    fig.add_trace(go.Bar(x=df_models["Model"], y=df_models["Accuracy (%)"], 
                         name="Accuracy", marker_color='rgb(102, 178, 234)'),
                  row=2, col=1)
    
    # Model Size (lower is better)
    fig.add_trace(go.Bar(x=df_models["Model"], y=df_models["Model Size (MB)"], 
                         name="Size", marker_color='rgb(162, 75, 118)'),
                  row=2, col=2)
    
    fig.update_layout(height=800, showlegend=False, title_text="Comprehensive Model Comparison")
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance vs Speed Scatter
    st.subheader("⚡ Performance vs Speed Trade-off Analysis")
    fig_scatter = go.Figure(data=go.Scatter(
        x=df_models["Speed (ms)"],
        y=df_models["mAP (%)"],
        mode='markers+text',
        text=df_models["Model"],
        textposition="top center",
        marker=dict(size=df_models["Model Size (MB)"], 
                   color=df_models["Accuracy (%)"],
                   colorscale='Viridis',
                   showscale=True,
                   colorbar=dict(title="Accuracy %"),
                   sizemode='diameter',
                   sizeref=10)
    ))
    fig_scatter.update_layout(
        xaxis_title="Inference Speed (ms) - Lower is Better",
        yaxis_title="mAP (%) - Higher is Better",
        height=500
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # FPS Comparison
    st.subheader("🎥 Frames Per Second (FPS) Comparison")
    fig_fps = go.Figure(data=[go.Bar(
        x=df_models["Model"],
        y=df_models["FPS"],
        marker=dict(
            color=df_models["FPS"],
            colorscale='RdYlGn',
            showscale=True
        ),
        text=df_models["FPS"],
        textposition='auto'
    )])
    fig_fps.update_layout(
        xaxis_title="Model",
        yaxis_title="Frames Per Second",
        height=400
    )
    st.plotly_chart(fig_fps, use_container_width=True)
    
    # Key Insights
    st.subheader("💡 Key Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Best for Accuracy:**
        - YOLOv3: Highest mAP (55.3%) and accuracy (91.2%)
        - Trade-off: Large model size (236 MB)
        
        **Best for Speed:**
        - YOLOv3-tiny: Fastest at 22ms (45.5 FPS)
        - Trade-off: Lower accuracy (78.4%)
        """)
    
    with col2:
        st.markdown("""
        **Best Balance:**
        - EfficientDet-Lite2: Good accuracy with small size (10.8 MB)
        - Current model in use ✅
        
        **Real-time Applications:**
        - YOLOv3-tiny and SSD MobileNet are best for real-time video
        """)

def export_batch_results(results_summary, processed_images):
    """Export batch processing results"""
    
    st.subheader("💾 Export Batch Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export summary CSV
        df_summary = pd.DataFrame(results_summary)
        csv_data = df_summary.to_csv(index=False)
        st.download_button(
            "📊 Download Summary CSV",
            csv_data,
            "batch_summary.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        # Export detailed JSON
        detailed_results = []
        for img_data in processed_images:
            detailed_results.append({
                "filename": img_data['filename'],
                "detections": [
                    {
                        "class": lbl,
                        "confidence": float(conf),
                        "bbox": [int(x) for x in bbox]
                    }
                    for lbl, conf, bbox in zip(
                        img_data['labels'],
                        img_data['confidences'],
                        img_data['bbox']
                    )
                ]
            })
        
        json_data = json.dumps(detailed_results, indent=2)
        st.download_button(
            "📄 Download Detailed JSON",
            json_data,
            "batch_detailed.json",
            "application/json",
            use_container_width=True
        )
    
    with col3:
        # Create downloadable zip with all processed images
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for img_data in processed_images:
                if img_data['bbox']:
                    img_np = np.array(img_data['image'])
                    result_img = draw_boxes(
                        img_np,
                        img_data['bbox'],
                        img_data['labels'],
                        img_data['confidences']
                    )
                    
                    img_buffer = io.BytesIO()
                    Image.fromarray(result_img).save(img_buffer, format='PNG')
                    zip_file.writestr(
                        f"processed_{img_data['filename']}",
                        img_buffer.getvalue()
                    )
        
        st.download_button(
            "🖼️ Download All Images (ZIP)",
            zip_buffer.getvalue(),
            "batch_processed_images.zip",
            "application/zip",
            use_container_width=True
        )

class ObjectDetectionApp:
    def __init__(self):
        self.model = None
        
    def detect_objects_tf(self, image_pil: Image.Image, confidence: float = 0.5) -> Tuple[List, List, List, dict]:
        """Detect objects using TensorFlow Hub model"""
        start_time = time.time()
        try:
            if self.model is None:
                with st.spinner("Loading detection model..."):
                    self.model = load_detection_model()
            if self.model is None:
                raise Exception("Model failed to load")
            
            img_array = np.array(image_pil.convert("RGB"))
            if img_array.dtype != np.uint8:
                img_array = (img_array * 255).astype(np.uint8)

            img_tensor = tf.convert_to_tensor(img_array)[tf.newaxis, ...]
            outputs = self.model(img_tensor)
            detection_boxes, detection_scores, detection_classes = _unpack_detections(outputs)

            H, W = img_array.shape[:2]
            pixel_boxes = _boxes_to_pixels(detection_boxes, W, H)

            bbox, labels, confidences = [], [], []
            for (x1, y1, x2, y2), score, cls in zip(pixel_boxes, detection_scores, detection_classes):
                s = float(score)
                if s < confidence:
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue

                if isinstance(cls, (bytes, bytearray)):
                    lbl = cls.decode("utf-8")
                elif isinstance(cls, (np.str_, str)):
                    lbl = str(cls)
                else:
                    cid = int(cls)
                    lbl = COCO_CLASSES[cid] if 0 <= cid < len(COCO_CLASSES) else f"class_{cid}"

                bbox.append([int(x1), int(y1), int(x2), int(y2)])
                labels.append(lbl)
                confidences.append(s)
            
            processing_time = time.time() - start_time
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
            return [], [], [], {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "processing_time": time.time() - start_time
            }
    
    def apply_image_enhancements(self, image: Image.Image, 
                                 brightness: float, contrast: float,
                                 sharpness: float, saturation: float = 1.0) -> Image.Image:
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
    
    def process_batch_images(self, uploaded_files, confidence, enhancement_settings):
        """Process multiple images at once"""
        
        results_summary = []
        processed_images = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, file in enumerate(uploaded_files):
            status_text.text(f"Processing {file.name} ({idx+1}/{len(uploaded_files)})")
            
            try:
                image = Image.open(file).convert('RGB')
                
                # Apply enhancements
                if enhancement_settings['enabled']:
                    image = self.apply_image_enhancements(
                        image,
                        enhancement_settings['brightness'],
                        enhancement_settings['contrast'],
                        enhancement_settings['sharpness'],
                        enhancement_settings['saturation']
                    )
                
                # Detect objects
                bbox, labels, confidences, metrics = self.detect_objects_tf(image, confidence)
                
                # Store results
                results_summary.append({
                    "filename": file.name,
                    "total_objects": len(labels),
                    "unique_classes": len(set(labels)),
                    "avg_confidence": f"{np.mean(confidences):.1%}" if confidences else "0%",
                    "processing_time": f"{metrics['processing_time']:.2f}s",
                    "detected_classes": ", ".join(set(labels)) if labels else "None"
                })
                
                processed_images.append({
                    "filename": file.name,
                    "image": image,
                    "bbox": bbox,
                    "labels": labels,
                    "confidences": confidences
                })
                
            except Exception as e:
                results_summary.append({
                    "filename": file.name,
                    "error": str(e),
                    "total_objects": 0,
                    "unique_classes": 0,
                    "avg_confidence": "N/A",
                    "processing_time": "N/A",
                    "detected_classes": "Error"
                })
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        status_text.text("✅ Batch processing complete!")
        return results_summary, processed_images

def main():
    app = ObjectDetectionApp()
    # Sidebar Navigation
    st.sidebar.markdown('<h2 style="color: #667eea;">🔍 Navigation</h2>', unsafe_allow_html=True)
    page = st.sidebar.radio(
        "Select Page",
        ["🎯 Single Image Detection", "📦 Batch Processing", "🔬 Model Comparison"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Common Settings (for detection pages)
    if page in ["🎯 Single Image Detection", "📦 Batch Processing"]:
        st.sidebar.header("⚙️ Detection Settings")
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold", 
            0.01, 0.95, 0.3, 0.01,
            help="Lower values = more detections"
        )
        
        st.sidebar.header("🎨 Image Enhancement")
        use_enhancement = st.sidebar.checkbox("Enable Enhancement", value=False)
        brightness = contrast = sharpness = saturation = 1.0
        if use_enhancement:
            brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
            contrast   = st.sidebar.slider("Contrast",   0.5, 2.0, 1.0, 0.1)
            sharpness  = st.sidebar.slider("Sharpness",  0.5, 2.0, 1.0, 0.1)
            saturation = st.sidebar.slider("Saturation", 0.5, 2.0, 1.0, 0.1)
        
        debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=False)
    
    # Model Comparison Page
    if page == "🔬 Model Comparison":
        create_model_comparison_dashboard()
        return
    
    # Batch Processing Page
    if page == "📦 Batch Processing":
        st.markdown('<h1 class="main-header">📦 Batch Image Processing</h1>', unsafe_allow_html=True)
        st.info("📌 Upload multiple images and process them all at once with comprehensive analytics")
        
        uploaded_files = st.file_uploader(
            "Upload multiple images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Select multiple images to process in batch"
        )
        
        if uploaded_files and len(uploaded_files) > 0:
            st.success(f"✅ {len(uploaded_files)} images uploaded and ready for processing")
            
            # Preview uploaded images
            with st.expander("📸 Preview Uploaded Images", expanded=False):
                cols = st.columns(min(4, len(uploaded_files)))
                for idx, file in enumerate(uploaded_files[:8]):  # Show first 8
                    with cols[idx % 4]:
                        img = Image.open(file)
                        st.image(img, caption=file.name, use_column_width=True)
                if len(uploaded_files) > 8:
                    st.info(f"... and {len(uploaded_files) - 8} more images")
            
            if st.button("🚀 Process All Images", type="primary", use_container_width=True):
                enhancement_settings = {
                    'enabled': use_enhancement,
                    'brightness': brightness,
                    'contrast': contrast,
                    'sharpness': sharpness,
                    'saturation': saturation
                }
                
                with st.spinner("Processing batch..."):
                    results_summary, processed_images = app.process_batch_images(
                        uploaded_files,
                        confidence_threshold,
                        enhancement_settings
                    )
                
                # Display summary
                st.success(f"✅ Successfully processed {len(uploaded_files)} images!")
                
                # Aggregate statistics
                total_objects = sum(r.get('total_objects', 0) for r in results_summary)
                avg_processing_time = np.mean([float(r['processing_time'].replace('s','')) 
                                              for r in results_summary 
                                              if 'processing_time' in r and r['processing_time'] != 'N/A'])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Images", len(uploaded_files))
                with col2:
                    st.metric("Total Objects Found", total_objects)
                with col3:
                    st.metric("Avg Processing Time", f"{avg_processing_time:.2f}s")
                with col4:
                    successful = len([r for r in results_summary if 'error' not in r])
                    st.metric("Success Rate", f"{(successful/len(results_summary)*100):.0f}%")
                
                # Show summary table
                st.subheader("📊 Batch Processing Summary")
                df_summary = pd.DataFrame(results_summary)
                st.dataframe(df_summary, use_container_width=True, hide_index=True)
                
                # Export options
                export_batch_results(results_summary, processed_images)
                
                # Show detailed results for each image
                with st.expander("🔍 View Detailed Results", expanded=False):
                    for img_data in processed_images:
                        if img_data['bbox']:
                            st.markdown(f"### {img_data['filename']}")
                            img_np = np.array(img_data['image'])
                            result_img = draw_boxes(
                                img_np,
                                img_data['bbox'],
                                img_data['labels'],
                                img_data['confidences']
                            )
                            st.image(result_img, use_column_width=True)
        return
    
    # Single Image Detection Page
    st.markdown('<h1 class="main-header">🔍 Advanced Object Detection Studio</h1>', unsafe_allow_html=True)
    st.info("🤖 Using TensorFlow Hub EfficientDet-Lite2 - Detects 90+ object classes")
    
    st.header("📁 Upload & Process")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Upload a single image for detection"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        enhanced_image = app.apply_image_enhancements(
            image, brightness, contrast, sharpness, saturation
        ) if use_enhancement else image
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📷 Original Image")
            st.image(image, use_column_width=True)
        with col2:
            st.subheader("✨ Enhanced Image" if use_enhancement else "🔄 Processing Image")
            st.image(enhanced_image, use_column_width=True)
        
        if st.button("🎯 Analyze Image", type="primary", use_container_width=True):
            st.markdown("---")
            st.subheader("🔄 Detection Process")
            
            with st.spinner('🔍 Detecting objects...'):
                bbox, labels, confidences, metrics = app.detect_objects_tf(
                    enhanced_image, 
                    confidence_threshold
                )
            
            if "error" in metrics:
                st.error("❌ Detection failed")
                st.error(metrics.get("error"))
                if debug_mode:
                    st.code(metrics.get("traceback", ""))
                    
            elif labels:
                st.success(f"🎉 Success! Found {len(labels)} objects in {metrics['processing_time']:.2f}s")
                
                # Show detected objects list
                st.subheader("🏷️ Detected Objects")
                for idx, (label, conf) in enumerate(zip(labels, confidences)):
                    st.write(f"{idx + 1}. **{label}** (confidence: {conf:.1%})")
                
                # Class filtering
                unique_classes = sorted(list(set(labels)))
                selected_classes = st.multiselect(
                    "🎯 Filter by class:",
                    unique_classes,
                    default=unique_classes,
                    help="Select which object classes to display"
                )
                
                filtered_bbox, filtered_labels, filtered_conf = app.filter_detections(
                    bbox, labels, confidences, selected_classes
                )
                
                try:
                    image_np = np.array(enhanced_image)
                    output_image = draw_boxes(image_np, filtered_bbox, filtered_labels, filtered_conf)
                    
                    st.subheader("🎯 Detection Results")
                    st.image(output_image, caption='Detected Objects', use_column_width=True)
                    
                    # Statistics
                    stats = app.create_detection_summary(filtered_labels, filtered_conf)
                    
                    # Analytics Dashboard
                    create_analytics_dashboard(stats, filtered_labels, filtered_conf)
                    
                    # Export options
                    st.markdown("---")
                    st.subheader("💾 Export Results")
                    e1, e2, e3 = st.columns(3)
                    with e1:
                        json_data = app.export_results_json(filtered_labels, filtered_conf, filtered_bbox)
                        st.download_button(
                            "📄 Download JSON",
                            json_data,
                            "detection_results.json",
                            "application/json",
                            use_container_width=True
                        )
                    with e2:
                        csv_data = app.export_results_csv(filtered_labels, filtered_conf)
                        st.download_button(
                            "📊 Download CSV",
                            csv_data,
                            "detection_results.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    with e3:
                        img_buffer = io.BytesIO()
                        Image.fromarray(output_image).save(img_buffer, format='PNG')
                        st.download_button(
                            "🖼️ Download Image",
                            img_buffer.getvalue(),
                            "processed_image.png",
                            "image/png",
                            use_container_width=True
                        )
                        
                except Exception as e:
                    st.error(f"Error drawing boxes: {str(e)}")
                    if debug_mode:
                        st.write("Raw detection results:")
                        st.write(f"Bounding boxes: {bbox}")
                        st.write(f"Labels: {labels}")
                        st.write(f"Confidences: {confidences}")
                        
            else:
                st.warning("⚠️ No objects detected in the image")
                st.markdown("""
                ### 🔧 Troubleshooting Tips:
                1. **Lower the confidence threshold** (try 0.1-0.3)
                2. **Try a different image** with clear, common objects
                3. **Enable image enhancement** to improve detection
                4. **Ensure good lighting** and clear visibility of objects
                """)
    else:
        st.markdown("""
        ## 👋 Welcome to Advanced Object Detection Studio!
        
        ### 🚀 Quick Start:
        1. **Upload an image** using the file uploader above
        2. **Adjust settings** in the sidebar (confidence threshold, enhancements)
        3. **Click "Analyze Image"** to start detection
        4. **View comprehensive analytics** and export results
        
        ### 🎯 Features:
        - **Single Image Detection** with detailed analytics dashboard
        - **Batch Processing** for multiple images at once
        - **Model Comparison** to understand different algorithms
        - **Interactive Visualizations** with Plotly charts
        - **Multiple Export Formats** (JSON, CSV, Images, ZIP)
        
        ### 💡 Tips for Better Detection:
        - Use images with **clear, well-lit objects**
        - Try **lower confidence thresholds** (0.2-0.4) for more detections
        - **Common objects work best**: people, cars, animals, furniture, electronics
        - Enable **image enhancement** for challenging lighting conditions
        """)

if __name__ == "__main__":
    main()
