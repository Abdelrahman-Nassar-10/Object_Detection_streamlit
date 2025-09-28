# 🔍 Advanced Object Detection Studio

A web application for real-time object detection using YOLO models, built with Streamlit and OpenCV. This application provides an intuitive interface for detecting 80+ different object classes in images with advanced features like image enhancement, confidence control, and detailed analytics.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![CVLib](https://img.shields.io/badge/CVLib-0.2.7%2B-orange)

## ✨ Features

### 🎯 Core Detection Capabilities
- **Multiple AI Models**: YOLOv3 and YOLOv3-tiny for different speed/accuracy trade-offs
- **80+ Object Classes**: Detect people, vehicles, animals, electronics, furniture, and more
- **Adjustable Confidence**: Real-time confidence threshold control (0.01 - 0.95)
- **High Accuracy**: Uses pre-trained COCO dataset models

### 🎨 Image Enhancement
- **Real-time Adjustments**: Brightness, contrast, sharpness, and saturation controls
- **Live Preview**: See enhancements before processing
- **Optional Processing**: Toggle enhancements on/off for optimal results
- **Automatic RGB Conversion**: Handles various image formats seamlessly

### 📊 Advanced Analytics
- **Detection Statistics**: Comprehensive analysis including processing time and confidence scores
- **Class Filtering**: Show/hide specific object categories
- **Performance Metrics**: Real-time processing speed monitoring
- **Debug Mode**: Detailed technical information for troubleshooting

### 💾 Export & Sharing
- **Multiple Formats**: Export results as JSON, CSV, or processed images with bounding boxes
- **Downloadable Results**: Save detection data for further analysis
- **Professional Output**: High-quality annotated images

### 🛠️ User Experience
- **Professional UI**: Modern gradient design with responsive layout
- **Progress Indicators**: Real-time feedback during processing
- **Error Handling**: Graceful error management with helpful troubleshooting tips
- **Mobile Friendly**: Works on desktop and mobile browsers

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Internet connection (for first-time model downloads)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/abdelrahmannassar/advanced-object-detection-studio.git
   cd advanced-object-detection-studio
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   The app will automatically open at `http://localhost:8501`

## 📖 Usage Guide

### Basic Object Detection
1. **Upload an image** using the file uploader
2. **Select detection model** (YOLOv3 for accuracy, YOLOv3-tiny for speed)
3. **Adjust confidence threshold** (lower values = more detections)
4. **Click "Analyze Image"** to perform detection
5. **View results** with bounding boxes and confidence scores

### Image Enhancement
Use the sidebar controls to improve detection accuracy:
- **Brightness**: Adjust for dark or overexposed images
- **Contrast**: Enhance object boundaries
- **Sharpness**: Reduce blur for better detection
- **Saturation**: Adjust color intensity

### Advanced Features
- **Class Filtering**: After detection, select which object types to display
- **Debug Mode**: Enable detailed technical information
- **Export Options**: Download results in JSON, CSV, or image formats

## 🎯 Supported Object Classes

The application can detect 80+ object classes from the COCO dataset:

**👥 People & Animals**
`person` `cat` `dog` `horse` `sheep` `cow` `elephant` `bear` `zebra` `giraffe`

**🚗 Vehicles**
`car` `motorcycle` `airplane` `bus` `train` `truck` `boat` `bicycle`

**💻 Electronics**
`laptop` `mouse` `remote` `keyboard` `cell phone` `microwave` `tv` `oven` `toaster`

**🪑 Furniture**
`chair` `couch` `potted plant` `bed` `dining table` `toilet`

**🍎 Food & Kitchen**
`bottle` `wine glass` `cup` `fork` `knife` `spoon` `bowl` `banana` `apple` `sandwich` `orange` `pizza` `donut` `cake`

**⚽ Sports & Recreation**
`frisbee` `skis` `snowboard` `sports ball` `kite` `baseball bat` `skateboard` `surfboard` `tennis racket`

**🎒 Personal Items**
`backpack` `umbrella` `handbag` `tie` `suitcase` `book` `clock` `vase` `scissors` `teddy bear` `hair drier` `toothbrush`

*And many more...*

## 🛠️ Technical Details

### Architecture
- **Frontend**: Streamlit web application framework
- **Backend**: CVLib with TensorFlow/OpenCV
- **AI Models**: Pre-trained YOLO models (YOLOv3, YOLOv3-tiny)
- **Image Processing**: PIL (Pillow) for enhancement and format handling

### Performance
| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| YOLOv3 | Medium | High | General purpose, balanced performance |
| YOLOv3-tiny | Fast | Good | Real-time applications, quick results |

### System Requirements
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: ~2GB for models and dependencies
- **CPU**: Any modern processor (GPU acceleration optional)
- **Browser**: Chrome, Firefox, Safari, or Edge

## 🔧 Configuration

### Model Selection
- **YOLOv3 (Default)**: Best balance of speed and accuracy
- **YOLOv3-tiny (Fast)**: Optimized for speed, good for real-time use

### Confidence Threshold
- **0.01-0.3**: Maximum sensitivity, may include false positives
- **0.3-0.6**: Balanced detection (recommended)
- **0.6-0.95**: High confidence only, fewer but more accurate detections

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Development Setup
```bash
git clone https://github.com/abdelrahmannassar/advanced-object-detection-studio.git
cd advanced-object-detection-studio
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
```

## 📝 About This Project

This project demonstrates advanced computer vision capabilities using modern web technologies. Built as a learning exercise and portfolio piece showcasing skills in Python, AI/ML, and web development.

## 🙏 Acknowledgments

- **CVLib** for providing easy-to-use object detection models
- **Streamlit** for the amazing web application framework
- **OpenCV** and **TensorFlow** communities for computer vision tools
- **COCO Dataset** for training data and object classes
- **YOLO** creators for the detection architecture


## 🔮 Future Enhancements

- [ ] Video detection support
- [ ] Webcam real-time detection
- [ ] Custom model training interface
- [ ] Batch processing for multiple images
- [ ] API endpoint for programmatic access
- [ ] Docker containerization
- [ ] Cloud deployment options

## 🏆 Project Stats

- **Languages**: Python, HTML, CSS
- **Framework**: Streamlit
- **AI/ML**: Computer Vision, Object Detection
- **Dependencies**: 7 core packages
- **Supported Formats**: JPG, JPEG, PNG
- **Object Classes**: 80+ (COCO dataset)

---

⭐ **If you find this project helpful, please give it a star!**

**Made with by Abdelrahman Nassar**


