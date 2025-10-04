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
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .debug-info {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: monospace;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

class ObjectDetectionApp:
    
    def __init__(self):
        # Try importing cvlib
        try:
            import cvlib as cv
            from cvlib.object_detection import draw_bbox
            self.cvlib_available = True
            self.cv = cv
            self.draw_bbox = draw_bbox
        except ImportError:
            self.cvlib_available = False
            st.warning("cvlib not installed. Using OpenCV fallback only.")
    
    def detect_with_cvlib(self, image_np: np.ndarray, confidence: float = 0.5) -> Tuple[List, List, List, dict]:
        """Use cvlib for object detection"""
        start_time = time.time()
        
        try:
            st.write("Using cvlib YOLO detection...")
            st.write(f"Image shape: {image_np.shape}")
            st.write(f"Confidence threshold: {confidence}")
            
            # Use cvlib
            bbox, labels, conf = self.cv.detect_common_objects(image_np, confidence=confidence, model='yolov3-tiny')
            
            processing_time = time.time() - start_time
            
            st.write(f"Detection completed in {processing_time:.2f}s")
            st.write(f"Found {len(labels)} objects")
            
            metrics = {
                "processing_time": processing_time,
                "total_detections": len(labels),
                "unique_classes": len(set(labels)) if labels else 0,
                "avg_confidence": np.mean(conf) if conf else 0,
                "max_confidence": max(conf) if conf else 0,
                "min_confidence": min(conf) if conf else 0
            }
            
            return bbox, labels, conf, metrics
            
        except Exception as e:
            import traceback
            st.error(f"cvlib detection error: {str(e)}")
            st.write("Falling back to OpenCV face detection...")
            return self.detect_with_opencv_cascade(image_np, confidence)
    
    def detect_with_opencv_cascade(self, image_np: np.ndarray, confidence: float = 0.5) -> Tuple[List, List, List, dict]:
        """Fallback: OpenCV cascade classifier"""
        start_time = time.time()
        
        try:
            st.write("Using OpenCV cascade detection (faces only)...")
            
            # Load cascades
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            bbox = []
            labels = []
            confidences = []
            
            for (x, y, w, h) in faces:
                bbox.append([x, y, x+w, y+h])
                labels.append("person")
                confidences.append(0.9)
            
            processing_time = time.time() - start_time
            
            st.write(f"Face detection completed in {processing_time:.2f}s")
            st.write(f"Found {len(labels)} faces")
            
            metrics = {
                "processing_time": processing_time,
                "total_detections": len(labels),
                "unique_classes": 1 if labels else 0,
                "avg_confidence": 0.9 if labels else 0,
                "max_confidence": 0.9 if labels else 0,
                "min_confidence": 0.9 if labels else 0
            }
            
            return bbox, labels, confidences, metrics
            
        except Exception as e:
            import traceback
            return [], [], [], {"error": str(e), "traceback": traceback.format_exc(), "processing_time": time.time() - start_time}
    
    def apply_image_enhancements(self, image: Image.Image, brightness: float, 
                                contrast: float, sharpness: float, saturation: float = 1.0) -> Image.Image:
        """Apply image enhancements"""
        try:
            enhanced = image.copy()
            
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(enhanced)
                enhanced = enhancer.enhance(brightness)
            
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(contrast)
                
            if sharpness != 1.0:
                enhancer = ImageEnhance.Sharpness(enhanced)
                enhanced = enhancer.enhance(sharpness)
                
            if saturation != 1.0:
                enhancer = ImageEnhance.Color(enhanced)
                enhanced = enhancer.enhance(saturation)
                
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
        avg_confidence = np.mean(confidences) if confidences else 0
        
        return {
            "total_objects": len(labels),
            "unique_classes": len(label_counts),
            "average_confidence": avg_confidence,
            "class_counts": dict(label_counts),
            "max_confidence": max(confidences) if confidences else 0,
            "min_confidence": min(confidences) if confidences else 0
        }
    def export_results_json(self, labels: List[str], confidences: List[float], bboxes: List) -> str:
        """Export as JSON"""
        results = []
        for i, label in enumerate(labels):
            # Convert bbox coordinates to regular Python ints
            bbox = bboxes[i] if i < len(bboxes) else None
            if bbox is not None:
                bbox = [int(x) for x in bbox]  # Convert numpy int32 to Python int
            
            results.append({
                "class": label,
                "confidence": float(confidences[i]) if i < len(confidences) else 0,
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
    
    # Generate colors
    np.random.seed(42)
    unique_labels = list(set(labels))
    colors = {label: tuple(map(int, np.random.randint(0, 255, 3))) for label in unique_labels}
    
    for box, label, conf in zip(boxes, labels, confidences):
        x1, y1, x2, y2 = map(int, box)
        color = colors.get(label, (0, 255, 0))
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        label_text = f"{label}: {conf:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return image

def main():
    app = ObjectDetectionApp()
    
    st.markdown('<h1 class="main-header">Advanced Object Detection Studio</h1>', unsafe_allow_html=True)
    
    # Show system status
    if app.cvlib_available:
        st.success("✅ Full object detection available (cvlib + YOLO)")
    else:
        st.info("ℹ️ Running in basic mode (OpenCV face detection only)")
    
    # Sidebar
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
    
    st.sidebar.header("Detection Settings")
    
    detection_method = st.sidebar.radio(
        "Detection Method",
        ["Auto (cvlib if available)" if app.cvlib_available else "OpenCV Face Detection", "OpenCV Face Detection Only"],
        help="Auto will try cvlib first, then fall back to OpenCV if needed"
    )
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.01,
        max_value=0.95,
        value=0.5,
        step=0.05
    )
    
    st.sidebar.header("Image Enhancement")
    use_enhancement = st.sidebar.checkbox("Enable Enhancement", value=False)
    
    brightness = contrast = sharpness = saturation = 1.0
    
    if use_enhancement:
        brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
        contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
        sharpness = st.sidebar.slider("Sharpness", 0.5, 2.0, 1.0, 0.1)
        saturation = st.sidebar.slider("Saturation", 0.5, 2.0, 1.0, 0.1)
    
    # Main content
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
            st.image(image, use_container_width=True)        
        with col2:
            st.subheader("Enhanced Image" if use_enhancement else "Processing Image")
            st.image(enhanced_image, use_column_width=True)
        
        if st.button("Analyze Image", type="primary", use_container_width=True):
            st.markdown("---")
            st.subheader("Detection Process")
            
            with st.spinner('Detecting objects...'):
                image_np = np.array(enhanced_image)
                
                if "Auto" in detection_method and app.cvlib_available:
                    bbox, labels, confidences, metrics = app.detect_with_cvlib(image_np, confidence_threshold)
                else:
                    bbox, labels, confidences, metrics = app.detect_with_opencv_cascade(image_np, confidence_threshold)
            
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
                selected_classes = st.multiselect(
                    "Filter by class:",
                    unique_classes,
                    default=unique_classes
                )
                
                filtered_bbox, filtered_labels, filtered_conf = app.filter_detections(
                    bbox, labels, confidences, selected_classes
                )
                
                try:
                    # Use cvlib's draw_bbox if available, otherwise our custom function
                    if app.cvlib_available and hasattr(app, 'draw_bbox'):
                        output_image = app.draw_bbox(image_np, filtered_bbox, filtered_labels, filtered_conf)
                    else:
                        output_image = draw_boxes(image_np, filtered_bbox, filtered_labels, filtered_conf)
                    
                    st.subheader("Detection Results")
                    st.image(output_image, caption='Detected Objects', use_column_width=True)
                    
                    stats = app.create_detection_summary(filtered_labels, filtered_conf)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Objects", stats['total_objects'])
                    with col2:
                        st.metric("Unique Classes", stats['unique_classes'])
                    with col3:
                        st.metric("Avg Confidence", f"{stats['average_confidence']:.2f}")
                    with col4:
                        st.metric("Processing Time", f"{metrics['processing_time']:.2f}s")
                    
                    st.subheader("Export Results")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        json_data = app.export_results_json(filtered_labels, filtered_conf, filtered_bbox)
                        st.download_button("Download JSON", json_data, "results.json", "application/json")
                    
                    with col2:
                        csv_data = app.export_results_csv(filtered_labels, filtered_conf)
                        st.download_button("Download CSV", csv_data, "results.csv", "text/csv")
                    
                    with col3:
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
                3. If using face detection only, make sure faces are visible and well-lit
                """)
    
    else:
        st.markdown("""
        ## Welcome!
        
        ### Quick Start:
        1. Upload an image
        2. Adjust confidence threshold  
        3. Click "Analyze Image"
        4. View and export results
        
        ### Detection Capabilities:
        """ + ("**Full YOLO Detection:** 80+ object classes" if app.cvlib_available else "**Basic Face Detection:** Detects faces only") + """
        
        ### Supported Objects:
        """ + ("person, car, bicycle, motorcycle, airplane, bus, train, truck, boat, cat, dog, horse, elephant, and 67+ more!" if app.cvlib_available else "Faces/people only in this mode") + """
        """)

if __name__ == "__main__":
    main()