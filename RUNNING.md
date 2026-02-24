# Running the Facial Recognition System - Complete Guide

This guide will help you set up and run the facial recognition system from scratch.

## 📋 Prerequisites

- **Python 3.8 or higher** (Python 3.12 recommended)
- **Webcam** (built-in or external)
- **MongoDB Atlas account** (free tier works perfectly)
- **Git** (to clone the repository)

## 🚀 Step-by-Step Setup

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd opencv
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```powershell
pip install -r requirements.txt
```

**This will install:**
- `opencv-python` - Camera and image processing
- `numpy` - Numerical operations
- `pymongo` - MongoDB database driver
- `facenet-pytorch` - Face recognition model
- `mtcnn` - Face detection
- `Pillow` - Image manipulation
- `torch` & `torchvision` - Deep learning framework

**Note:** First run will download the pretrained FaceNet model (~100MB). This is normal.

### Step 4: Set Up MongoDB Atlas (Database)

#### 4.1 Create MongoDB Atlas Account

1. Go to [https://www.mongodb.com/cloud/atlas/register](https://www.mongodb.com/cloud/atlas/register)
2. Sign up for a **FREE** account
3. Choose **"Build a Database"**
4. Select **M0 FREE** tier
5. Choose your preferred cloud provider and region
6. Click **"Create Cluster"** (takes 3-5 minutes)

#### 4.2 Create Database User

1. In the left sidebar, click **"Database Access"**
2. Click **"Add New Database User"**
3. Choose **"Password"** authentication
4. Set username (e.g., `meetingapp`)
5. Set a strong password (e.g., `SecurePass123!`)
6. Set privileges to **"Read and write to any database"**
7. Click **"Add User"**

**⚠️ Important:** Save your username and password - you'll need them!

#### 4.3 Allow Network Access

1. In the left sidebar, click **"Network Access"**
2. Click **"Add IP Address"**
3. Click **"Allow Access from Anywhere"** (or add your specific IP)
4. Click **"Confirm"**

#### 4.4 Get Connection String

1. Go back to **"Database"** (left sidebar)
2. Click **"Connect"** on your cluster
3. Choose **"Connect your application"**
4. Select **Driver: Python**, **Version: 3.12 or later**
5. Copy the connection string - it looks like:
   ```
   mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```

### Step 5: Configure the Application

Open `config.py` and update the MongoDB connection:

**Find this section:**
```python
# ==================== Database Settings ====================
# MongoDB connection - Choose one:

# Option 1: Local MongoDB
MONGODB_URI = "mongodb://localhost:27017/"
```

**Replace with your Atlas connection string:**
```python
# ==================== Database Settings ====================
# MongoDB connection - Choose one:

# Option 1: Local MongoDB (COMMENTED OUT)
# MONGODB_URI = "mongodb://localhost:27017/"

# Option 2: MongoDB Atlas (YOUR CONNECTION STRING)
MONGODB_URI = "mongodb+srv://meetingapp:SecurePass123!@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority"
```

**⚠️ Important Replacements:**
- Replace `<username>` with your MongoDB username (e.g., `meetingapp`)
- Replace `<password>` with your MongoDB password (e.g., `SecurePass123!`)
- Replace `cluster0.xxxxx` with your actual cluster address

**Example:**
```python
MONGODB_URI = "mongodb+srv://meetingapp:MyPass123@cluster0.abc123.mongodb.net/?retryWrites=true&w=majority"
```

### Step 6: Test Database Connection

```powershell
python database.py
```

**Expected output:**
```
Connected to database successfully!
Total embeddings in database: 0
✓ Embedding stored successfully
✓ Embedding retrieved successfully
Database test complete!
```

If you see errors, double-check your connection string and network access settings.

## 🎥 Running the Application

### Interactive Mode (Recommended for First Use)

This mode uses your webcam to capture faces in real-time.

```powershell
python main.py
```

**What happens:**
1. Camera window opens showing live feed
2. Position your face in the frame
3. Press **`c`** to capture
4. System detects and analyzes your face
5. If first time: You'll be asked to enter your name
6. If returning: System recognizes you automatically

**Example Session:**
```
Press 'c' to capture, 'q' to quit

[You press 'c']

Step 1/3: Detecting face...
✓ Face detected (confidence: 0.998)

Step 2/3: Generating face embedding...
✓ Embedding generated successfully

Step 3/3: Recognizing person...
NEW PERSON DETECTED
Enter person's name: John Doe
Confirm name 'John Doe'? (y/n): y
✓ NEW person registered: John Doe

RESULT:
✓ NEW person registered: John Doe
  Status: First time interaction
```

**Second time you run it:**
```
[You press 'c']

Step 1/3: Detecting face...
✓ Face detected (confidence: 0.995)

Step 2/3: Generating face embedding...
✓ Embedding generated successfully

Step 3/3: Recognizing person...
✓ Person RECOGNIZED: John Doe (confidence: 0.923)

RESULT:
✓ Person RECOGNIZED: John Doe
  Confidence: 92.3%
  Status: Returning person
```

### Batch Mode (Process Saved Image)

If you have a saved image instead of using webcam:

```powershell
python main.py path\to\image.jpg
```

### List All Registered People

```powershell
python main.py --list
```

**Output:**
```
REGISTERED PEOPLE: 3
1. Alice Johnson
2. Bob Smith
3. John Doe
```

## 🧪 Testing the System

### Run All Tests

```powershell
python test_system.py all path\to\test_image.jpg
```

### Individual Component Tests

**Test 1: Embedding Consistency**
```powershell
python test_system.py 1 image.jpg
```
Verifies the same image produces identical embeddings.

**Test 2: Same Person Recognition**
```powershell
python test_system.py 2 person1_photo1.jpg person1_photo2.jpg
```
Tests if different photos of the same person are recognized.

**Test 3: Different Person Detection**
```powershell
python test_system.py 3 person1.jpg person2.jpg
```
Tests if different people are correctly distinguished.

**Test 4: Duplicate Prevention**
```powershell
python test_system.py 4
```
Verifies the system prevents storing duplicate embeddings.

**Test 5: Full Recognition Pipeline**
```powershell
python test_system.py 5 image.jpg
```
Tests the complete registration and recognition flow.

## ⚙️ Configuration Options

### Adjust Recognition Sensitivity

In `config.py`, modify the recognition threshold:

```python
# Cosine similarity thresholds
RECOGNITION_THRESHOLD = 0.85  # Default: 0.85

# Stricter (fewer false positives, may miss some matches)
RECOGNITION_THRESHOLD = 0.90

# More lenient (more matches, but may have false positives)
RECOGNITION_THRESHOLD = 0.80
```

### Change Camera

If using an external webcam:

```python
CAMERA_INDEX = 0  # Default (built-in camera)
CAMERA_INDEX = 1  # Try this for external camera
```

### Enable Debug Mode

For troubleshooting:

```python
DEBUG_MODE = True
SHOW_DEBUG_WINDOW = True
LOG_LEVEL = "DEBUG"
```

## 🐛 Troubleshooting

### Issue: "Failed to connect to database"

**Solution:**
1. Check your internet connection
2. Verify MongoDB Atlas connection string in `config.py`
3. Ensure network access is enabled in Atlas (allow all IPs: `0.0.0.0/0`)
4. Check username/password are correct (no `<` `>` brackets)

**Test connection:**
```powershell
python database.py
```

### Issue: "Failed to open camera at index 0"

**Solutions:**
1. Close other apps using the webcam (Zoom, Skype, etc.)
2. Try different camera index in `config.py`:
   ```python
   CAMERA_INDEX = 1  # or 2, 3...
   ```
3. Check webcam permissions in Windows Settings

### Issue: "No faces detected in image"

**Solutions:**
1. Ensure good lighting
2. Face should be clearly visible and front-facing
3. Move closer to camera (50-100cm distance)
4. Lower detection threshold in `config.py`:
   ```python
   MIN_DETECTION_CONFIDENCE = 0.85  # was 0.90
   ```

### Issue: "ModuleNotFoundError"

**Solution:**
```powershell
# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Person not being recognized (low similarity)

**Solutions:**
1. Lower recognition threshold in `config.py`:
   ```python
   RECOGNITION_THRESHOLD = 0.80  # was 0.85
   ```
2. Ensure consistent lighting between registration and recognition
3. Register multiple photos of the same person from different angles

## 📊 Understanding the Output

### Similarity Scores

| Range | Meaning | Action |
|-------|---------|--------|
| 0.95 - 1.00 | Same person, high confidence | ✓ Recognized |
| 0.85 - 0.95 | Same person, good confidence | ✓ Recognized |
| 0.70 - 0.85 | Uncertain | ✗ Not recognized (register new) |
| 0.00 - 0.70 | Different person | ✗ Not recognized |

### Embedding Statistics (in logs)

```
EMBEDDING STATISTICS:
  Shape:        (512,)           # Correct size
  Mean:         0.000234         # Should be close to 0
  Std Dev:      0.044127         # Typically 0.03-0.05
  L2 Norm:      1.000000         # Must be exactly 1.0
  First 5 vals: [0.02, -0.01, ...] # Sample values
```

## 🔒 Security & Privacy

### Best Practices for Collaborators

1. **Never commit `config.py` with real credentials**
   - Create `config.py.example` without credentials
   - Add `config.py` to `.gitignore`

2. **Use environment variables (optional but recommended):**

Create a `.env` file:
```
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/
```

Update `config.py`:
```python
import os
from dotenv import load_dotenv

load_dotenv()
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
```

Install python-dotenv:
```powershell
pip install python-dotenv
```

3. **Temporary images are auto-deleted** after processing (configurable in `config.py`)

## 📁 Project Structure

```
opencv/
├── main.py              # Main application entry point
├── camera.py            # Webcam capture
├── detector.py          # Face detection (MTCNN)
├── embedder.py          # Face embeddings (FaceNet)
├── database.py          # MongoDB operations
├── recognizer.py        # Face matching (cosine similarity)
├── config.py            # Configuration (UPDATE THIS!)
├── test_system.py       # Testing utilities
├── requirements.txt     # Python dependencies
├── temp/                # Temporary images (auto-created)
├── README.md            # Full documentation
├── QUICKSTART.md        # Quick reference
└── RUNNING.md           # This file
```

## 🤝 For Collaborators

### Quick Setup Commands

```powershell
# Clone and setup
git clone <repo-url>
cd opencv
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt

# Configure MongoDB Atlas in config.py
# (See Step 5 above)

# Test connection
python database.py

# Run the app
python main.py
```

### Sharing Database

All team members can use the **same MongoDB Atlas cluster**:
1. Share the connection string securely (Slack, email, etc.)
2. Everyone uses the same `MONGODB_URI` in their `config.py`
3. All registered faces will be shared across the team

**Note:** Make sure all team members have the connection string but **DO NOT** commit it to Git!

## 📞 Support

If you encounter issues:

1. Check this guide first
2. Review the [README.md](README.md) for technical details
3. Check the [QUICKSTART.md](QUICKSTART.md) for quick reference
4. Run tests to isolate the problem:
   ```powershell
   python test_system.py all test_image.jpg
   ```

## ✅ Checklist for New Setup

- [ ] Python 3.8+ installed
- [ ] Repository cloned
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] MongoDB Atlas account created
- [ ] Database cluster created
- [ ] Database user created (username/password saved)
- [ ] Network access enabled (0.0.0.0/0)
- [ ] Connection string copied
- [ ] `config.py` updated with connection string
- [ ] Database connection tested (`python database.py`)
- [ ] Webcam permissions granted
- [ ] Application runs successfully (`python main.py`)

---

**Happy Face Recognizing! 🎉**

If everything works, you should be able to capture faces, register new people, and recognize returning people with high accuracy.
