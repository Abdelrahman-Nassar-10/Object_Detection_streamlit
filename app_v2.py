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
    
    .stats-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .detection-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .sidebar-section {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
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
        self.detection_models = {
            "YOLOv3 (Default)": "yolov3",
            "YOLOv3-tiny (Fast)": "yolov3-tiny"
        }
        
    def apply_image_enhancements(self, image: Image.Image, brightness: float, 
                                contrast: float, sharpness: float, saturation: float = 1.0) -> Image.Image:
        """Apply image enhancements before detection"""
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
    
    def detect_objects_simple(self, image_np: np.ndarray, confidence: float = 0.3) -> Tuple[List, List, List, dict]:
        """Simplified object detection that matches your original working code"""
        start_time = time.time()
        
        try:
            st.write("🔍 Starting detection...")
            st.write(f"Image shape: {image_np.shape}")
            st.write(f"Confidence threshold: {confidence}")
            
            # Use the exact same method as your original working code
            bbox, label, conf = cv.detect_common_objects(image_np, confidence=confidence)
            
            processing_time = time.time() - start_time
            
            st.write(f"✅ Detection completed in {processing_time:.2f}s")
            st.write(f"Found {len(label)} objects")
            
            # Calculate metrics
            metrics = {
                "processing_time": processing_time,
                "total_detections": len(label),
                "unique_classes": len(set(label)) if label else 0,
                "avg_confidence": np.mean(conf) if conf else 0,
                "max_confidence": max(conf) if conf else 0,
                "min_confidence": min(conf) if conf else 0
            }
            
            return bbox, label, conf, metrics
            
        except Exception as e:
            st.error(f"🚨 Detection error: {str(e)}")
            import traceback
            st.error(f"Full error traceback: {traceback.format_exc()}")
            return [], [], [], {"error": str(e), "processing_time": time.time() - start_time}
    
    def filter_detections(self, bbox: List, labels: List, confidences: List,
                         selected_classes: List[str]) -> Tuple[List, List, List]:
        """Filter detections based on selected classes"""
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
        """Create detection statistics summary"""
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
    
    def export_results_json(self, labels: List[str], confidences: List[float], 
                           bboxes: List) -> str:
        """Export detection results as JSON"""
        results = []
        for i, label in enumerate(labels):
            results.append({
                "class": label,
                "confidence": float(confidences[i]) if i < len(confidences) else 0,
                "bbox": bboxes[i] if i < len(bboxes) else None
            })
        
        return json.dumps(results, indent=2)
    
    def export_results_csv(self, labels: List[str], confidences: List[float]) -> str:
        """Export detection results as CSV"""
        df = pd.DataFrame({
            "Object_Class": labels,
            "Confidence": confidences
        })
        return df.to_csv(index=False)

def main():
    app = ObjectDetectionApp()
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Advanced Object Detection Studio</h1>', 
                unsafe_allow_html=True)
    
    # Debug mode toggle
    debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=False, help="Show detailed debugging information")
    
    # Sidebar configuration
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.header("⚙️ Detection Settings")
    
    # Simplified model selection (only working models)
    selected_model_name = st.sidebar.selectbox(
        "Choose Detection Model",
        ["YOLOv3 (Default)", "YOLOv3-tiny (Fast)"],
        help="Using only tested working models"
    )
    
    # Confidence threshold with wider range
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.01,
        max_value=0.95,
        value=0.5,
        step=0.05,
        help="Lower values = more detections (try 0.1 for maximum sensitivity)"
    )
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Image enhancement settings (simplified)
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.header("🎨 Image Enhancement")
    
    # Add option to skip enhancement
    use_enhancement = st.sidebar.checkbox("Enable Image Enhancement", value=False)
    
    brightness = 1.0
    contrast = 1.0 
    sharpness = 1.0
    saturation = 1.0
    
    if use_enhancement:
        brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
        contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
        sharpness = st.sidebar.slider("Sharpness", 0.5, 2.0, 1.0, 0.1)
        saturation = st.sidebar.slider("Saturation", 0.5, 2.0, 1.0, 0.1)
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area
    st.header("📁 Upload & Process")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Upload a single image for testing"
    )
    
    if uploaded_file is not None:
        # Load and display the image
        image = Image.open(uploaded_file)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply enhancements only if enabled
        if use_enhancement:
            enhanced_image = app.apply_image_enhancements(
                image, brightness, contrast, sharpness, saturation
            )
        else:
            enhanced_image = image
        
        # Convert to numpy array
        image_np = np.array(enhanced_image)
        
        # Display images
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
                Format: {getattr(image, 'format', 'Unknown')}<br>
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
            
            with st.spinner('Detecting objects...'):
                # Perform detection with debugging
                bbox, labels, confidences, metrics = app.detect_objects_simple(
                    image_np, confidence_threshold
                )
            
            if labels:
                st.success(f"🎉 Success! Found {len(labels)} objects")
                
                # Show detected objects list
                st.subheader("🏷️ Detected Objects")
                for idx, (label, conf) in enumerate(zip(labels, confidences)):
                    st.write(f"{idx + 1}. **{label}** (confidence: {conf:.3f})")
                
                # Class filtering
                unique_classes = sorted(list(set(labels)))
                selected_classes = st.multiselect(
                    "🎯 Filter by object classes:",
                    unique_classes,
                    default=unique_classes,
                    help="Select which object classes to display"
                )
                
                # Apply filtering
                filtered_bbox, filtered_labels, filtered_conf = app.filter_detections(
                    bbox, labels, confidences, selected_classes
                )
                
                # Draw bounding boxes
                try:
                    output_image = draw_bbox(image_np, filtered_bbox, filtered_labels, filtered_conf)
                    
                    st.subheader("🎯 Detection Results")
                    st.image(output_image, caption='Detected Objects', use_column_width=True)
                    
                    # Statistics
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
                    
                    # Export options
                    st.subheader("💾 Export Results")
                    export_col1, export_col2, export_col3 = st.columns(3)
                    
                    with export_col1:
                        json_data = app.export_results_json(filtered_labels, filtered_conf, filtered_bbox)
                        st.download_button(
                            label="📄 Download JSON",
                            data=json_data,
                            file_name="detection_results.json",
                            mime="application/json"
                        )
                    
                    with export_col2:
                        csv_data = app.export_results_csv(filtered_labels, filtered_conf)
                        st.download_button(
                            label="📊 Download CSV",
                            data=csv_data,
                            file_name="detection_results.csv",
                            mime="text/csv"
                        )
                    
                    with export_col3:
                        # Convert processed image to downloadable format
                        img_buffer = io.BytesIO()
                        Image.fromarray(output_image).save(img_buffer, format='PNG')
                        st.download_button(
                            label="🖼️ Download Image",
                            data=img_buffer.getvalue(),
                            file_name="processed_image.png",
                            mime="image/png"
                        )
                
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
                1. **Lower the confidence threshold** (try 0.1 or 0.05)
                2. **Try a different image** with clear, common objects (people, cars, animals)
                3. **Enable Debug Mode** to see more information
                4. **Check image quality** - very dark, blurry, or low-resolution images may not work well
                5. **Try without image enhancement** (uncheck the enhancement option)
                """)
                
                if debug_mode:
                    st.markdown(f"""
                    <div class="debug-info">
                    <strong>Detection Debug Info:</strong><br>
                    Confidence threshold: {confidence_threshold}<br>
                    Image array shape: {image_np.shape}<br>
                    Processing time: {metrics.get('processing_time', 'N/A')}<br>
                    Error: {metrics.get('error', 'None')}
                    </div>
                    """, unsafe_allow_html=True)
    
    else:
        # Instructions
        st.markdown("""
        ## 👋 Welcome to Advanced Object Detection Studio!
        
        ### 🚀 Quick Start:
        1. **Upload an image** using the file uploader above
        2. **Adjust the confidence threshold** (try 0.1 for maximum sensitivity) 
        3. **Click "Analyze Image"** to start detection
        4. **View results** and export data
        
        ### 💡 Tips for Better Detection:
        - Use images with **clear, well-lit objects**
        - Try **lower confidence thresholds** (0.1-0.3) for more detections
        - **Common objects work best**: people, cars, animals, furniture, etc.
        - Enable **Debug Mode** in the sidebar to see detailed information
        
        ### 🎯 Supported Objects (80+ classes):
        **People & Animals**: person, cat, dog, horse, cow, elephant, bear, zebra, giraffe  
        **Vehicles**: car, motorcycle, airplane, bus, train, truck, boat, bicycle  
        **Electronics**: laptop, mouse, remote, keyboard, cell phone, microwave, tv  
        **Furniture**: chair, couch, dining table, bed  
        **Food**: banana, apple, sandwich, orange, pizza, donut, cake  
        **And many more...**
        """)

if __name__ == "__main__":
    main()
    