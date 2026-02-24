# Facial Recognition Meeting Assistant

A production-ready facial recognition system that captures faces via webcam, stores identities in MongoDB, and recognizes people using FaceNet embeddings with cosine similarity.

## 🎯 Features

- **Real-time Face Capture**: OpenCV webcam integration
- **Face Detection**: MTCNN with confidence validation
- **Face Embedding**: FaceNet (512-dimensional embeddings)
- **Face Recognition**: Cosine similarity matching
- **Database Storage**: MongoDB with duplicate prevention
- **Smart Registration**: Automatic new person detection
- **Privacy-First**: Temporary image cleanup
- **Comprehensive Logging**: Debug and production modes

## 📁 Project Structure

```
d:\opencv\
├── main.py              # Main application entry point
├── camera.py            # Webcam capture module
├── detector.py          # MTCNN face detection
├── embedder.py          # FaceNet embedding generation
├── database.py          # MongoDB storage handler
├── recognizer.py        # Cosine similarity recognition
├── config.py            # System configuration
├── test_system.py       # Testing utilities
├── requirements.txt     # Python dependencies
├── temp/                # Temporary image storage (auto-created)
└── README.md            # This file
```

## 🚀 Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- MongoDB 4.0 or higher
- Webcam (for live capture mode)

### 2. Install MongoDB

**Windows:**
1. Download from: https://www.mongodb.com/try/download/community
2. Run installer and follow setup wizard
3. Start MongoDB service:
   ```powershell
   net start MongoDB
   ```

**macOS:**
```bash
brew install mongodb-community
brew services start mongodb-community
```

**Linux:**
```bash
sudo apt-get install mongodb
sudo systemctl start mongodb
```

Verify MongoDB is running:
```powershell
mongo --eval "db.version()"
```

### 3. Install Python Dependencies

```powershell
# Navigate to project directory
cd d:\opencv

# Install required packages
pip install -r requirements.txt
```

**Note**: First run will download pretrained FaceNet model (~100MB)

### 4. Configuration

Edit [config.py](config.py) to customize:

```python
# Recognition threshold (0.85-0.90 recommended)
RECOGNITION_THRESHOLD = 0.85

# Duplicate detection threshold
DUPLICATE_THRESHOLD = 0.98

# Camera settings
CAMERA_INDEX = 0  # Change if using external webcam

# MongoDB connection
MONGODB_URI = "mongodb://localhost:27017/"
```

## 📖 Usage

### Interactive Mode (Live Camera)

Capture faces in real-time and automatically recognize or register:

```powershell
python main.py
```

**Controls:**
- Press `c` to capture current frame
- Press `q` to quit

**Workflow:**
1. System shows live camera feed
2. Position face in frame and press `c`
3. System detects face and generates embedding
4. If recognized: displays name and confidence
5. If new person: prompts for name and registers

### Batch Mode (Process Image File)

Process a single image without camera:

```powershell
python main.py path\to\image.jpg
```

### List Registered People

View all stored identities:

```powershell
python main.py --list
```

## 🧪 Testing

### Run All Tests

```powershell
python test_system.py all path\to\test_image.jpg
```

### Individual Tests

**Test 1: Embedding Consistency**
```powershell
python test_system.py 1 image.jpg
```
Verifies same image produces identical embeddings.

**Test 2: Same Person Recognition**
```powershell
python test_system.py 2 person1_photo1.jpg person1_photo2.jpg
```
Tests if different photos of same person have similarity ≥ 0.85.

**Test 3: Different Person Detection**
```powershell
python test_system.py 3 person1.jpg person2.jpg
```
Tests if different people have similarity < 0.85.

**Test 4: Duplicate Detection**
```powershell
python test_system.py 4
```
Verifies system prevents storing duplicate embeddings.

**Test 5: Recognition Pipeline**
```powershell
python test_system.py 5 image.jpg
```
Tests complete registration and recognition flow.

## 🔧 Module Testing

### Test Camera Module
```powershell
python camera.py
```
Opens webcam and captures test frame.

### Test Face Detector
```powershell
python detector.py image.jpg
```
Detects and visualizes face with bounding box.

### Test Face Embedder
```powershell
# Single image
python embedder.py image.jpg

# Compare two images
python embedder.py image1.jpg image2.jpg
```

### Test Database Connection
```powershell
python database.py
```

### Test Recognizer
```powershell
python recognizer.py
```

## 📊 Expected Behavior

### First-Time Person
```
1. Webcam captures face
2. System: "NEW PERSON DETECTED"
3. User enters name: "John Doe"
4. System stores embedding
5. Output: "✓ NEW person registered: John Doe"
```

### Returning Person
```
1. Webcam captures face
2. System computes similarity with all stored embeddings
3. Best match: "John Doe" with 0.93 confidence
4. Output: "✓ Person RECOGNIZED: John Doe (93% confidence)"
```

### Different Person
```
1. Webcam captures face
2. System computes similarity
3. Best match: 0.72 (below 0.85 threshold)
4. System: "NEW PERSON DETECTED"
5. User registers new identity
```

## 📝 Debug Logs Example

```
2026-02-24 10:30:15 - camera - INFO - Camera opened successfully at index 0
2026-02-24 10:30:18 - camera - INFO - Frame captured and saved to: d:\opencv\temp\captured_face.jpg
2026-02-24 10:30:18 - detector - INFO - Processing image: d:\opencv\temp\captured_face.jpg
2026-02-24 10:30:19 - detector - DEBUG - Detected 1 face(s)
2026-02-24 10:30:19 - detector - INFO - ✓ Face detected (confidence: 0.998)
2026-02-24 10:30:19 - embedder - INFO - Generating embedding for face - Shape: (224, 224, 3)
2026-02-24 10:30:19 - embedder - DEBUG - Converted BGR to RGB - Shape: (224, 224, 3)
2026-02-24 10:30:19 - embedder - DEBUG - Resized to 160x160
2026-02-24 10:30:20 - embedder - INFO - ============================================================
2026-02-24 10:30:20 - embedder - INFO - EMBEDDING STATISTICS:
2026-02-24 10:30:20 - embedder - INFO -   Shape:        (512,)
2026-02-24 10:30:20 - embedder - INFO -   Mean:         0.000234
2026-02-24 10:30:20 - embedder - INFO -   Std Dev:      0.044127
2026-02-24 10:30:20 - embedder - INFO -   L2 Norm:      1.000000
2026-02-24 10:30:20 - embedder - INFO -   First 5 vals: [ 0.0234 -0.0156  0.0445  0.0021 -0.0312]
2026-02-24 10:30:20 - embedder - INFO - ============================================================
2026-02-24 10:30:20 - recognizer - INFO - Comparing against 3 stored embeddings
2026-02-24 10:30:20 - recognizer - DEBUG - Similarity with 'Alice': 0.6234
2026-02-24 10:30:20 - recognizer - DEBUG - Similarity with 'Bob': 0.7123
2026-02-24 10:30:20 - recognizer - DEBUG - Similarity with 'Charlie': 0.9234
2026-02-24 10:30:20 - recognizer - INFO - Best match: 'Charlie' with similarity: 0.9234
2026-02-24 10:30:20 - recognizer - INFO - ✓ Person RECOGNIZED: 'Charlie' (confidence: 0.9234)
```

## 🎛️ Configuration Options

### Recognition Thresholds

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 0.80 | Loose matching | Varied lighting/angles |
| 0.85 | **Recommended** | Balanced accuracy |
| 0.90 | Strict matching | High security needs |
| 0.95 | Very strict | Minimize false positives |

### Debug Settings

```python
# config.py
DEBUG_MODE = True               # Enable debug features
SHOW_DEBUG_WINDOW = True        # Show detection visualizations
LOG_LEVEL = "DEBUG"             # Detailed logging
DELETE_TEMP_IMAGES = True       # Auto cleanup
```

## 🔒 Production Safeguards

✅ **Exception Handling**: All modules have try-catch blocks  
✅ **Database Connection**: Automatic retry and validation  
✅ **Duplicate Prevention**: Similarity-based deduplication  
✅ **Input Validation**: Name length and character checks  
✅ **Privacy**: Temporary image auto-deletion  
✅ **Logging**: Comprehensive debug and error tracking  
✅ **Configurable**: All thresholds externalized  

## 🐛 Troubleshooting

### Camera Not Opening
```
Error: Failed to open camera at index 0
Solution: Check if another app is using webcam or change CAMERA_INDEX in config.py
```

### MongoDB Connection Failed
```
Error: Failed to connect to MongoDB
Solution: 
  1. Verify MongoDB is running: mongo --eval "db.version()"
  2. Check MONGODB_URI in config.py
  3. Restart MongoDB service
```

### Face Not Detected
```
Warning: No faces detected in image
Solution:
  1. Ensure face is well-lit and clearly visible
  2. Move closer to camera
  3. Lower MIN_DETECTION_CONFIDENCE in config.py
```

### Low Recognition Accuracy
```
Issue: Same person not being recognized
Solution:
  1. Lower RECOGNITION_THRESHOLD (try 0.80)
  2. Ensure consistent lighting
  3. Capture multiple angles during registration
```

### Duplicate Embeddings
```
Warning: Embedding too similar to existing entry
Solution: This is expected - system prevents duplicates
```

## 📚 Technical Details

### Face Detection (MTCNN)
- **Architecture**: Multi-task Cascaded CNN
- **Thresholds**: P-Net (0.6), R-Net (0.7), O-Net (0.7)
- **Min Face Size**: 40 pixels
- **Confidence**: ≥ 0.90 required

### Face Embedding (FaceNet)
- **Model**: InceptionResnetV1 (pretrained on VGGFace2)
- **Input**: 160×160 RGB images
- **Output**: 512-dimensional L2-normalized vectors
- **Preprocessing**: BGR→RGB, resize, normalize to [-1, 1]

### Cosine Similarity
```python
similarity = dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
```
For L2-normalized vectors: `similarity = dot(embedding1, embedding2)`

Range: [-1, 1] where:
- 1.0 = identical
- 0.0 = orthogonal
- -1.0 = opposite

### Database Schema
```javascript
{
  name: "John Doe",
  embedding: [0.0234, -0.0156, ...],  // 512 floats
  date: ISODate("2026-02-24T10:30:20Z"),
  embedding_size: 512
}
```

## 🧑‍💻 Development

### Add New Features

Extend [recognizer.py](recognizer.py) for:
- Multiple face tracking
- Face clustering
- Age/gender estimation

### Custom Embedders

Replace FaceNet in [embedder.py](embedder.py) with:
- ArcFace
- DeepFace
- VGGFace

### Alternative Databases

Modify [database.py](database.py) to use:
- PostgreSQL with pgvector
- Redis
- SQLite with vector extension

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- **FaceNet**: Schroff et al. (2015)
- **MTCNN**: Zhang et al. (2016)
- **facenet-pytorch**: Tim Esler
- **OpenCV**: Open Source Computer Vision Library

---

**Author**: Built with GitHub Copilot  
**Date**: February 2026  
**Version**: 1.0.0
#   f a c e D e t e c t  
 