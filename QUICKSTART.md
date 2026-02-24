# Quick Start Guide - Facial Recognition System

## ⚡ 5-Minute Setup

### Step 1: Install MongoDB

**Windows (PowerShell as Administrator):**
```powershell
# Download MongoDB installer
# Visit: https://www.mongodb.com/try/download/community
# Run the installer, choose "Complete" installation

# Start MongoDB service
net start MongoDB

# Verify it's running
mongo --eval "db.version()"
```

**Alternative - MongoDB as a Service:**
If you don't want to install locally, use MongoDB Atlas (free cloud):
1. Visit https://www.mongodb.com/cloud/atlas
2. Create free account
3. Create cluster
4. Get connection string
5. Update `MONGODB_URI` in config.py

### Step 2: Install Python Packages

```powershell
cd d:\opencv
pip install -r requirements.txt
```

**Wait for FaceNet model download** (~100MB, first run only)

### Step 3: Run the System

```powershell
# Start the system
python main.py
```

## 🎥 First Run Example

```
1. Camera window opens
2. Position your face in frame
3. Press 'c' to capture
4. System detects face
5. Prompts: "Enter person's name: "
6. Type your name and press Enter
7. Confirm with 'y'
8. Done! You're registered
```

## 🔄 Second Run Example

```
1. Run: python main.py
2. Capture your face again (press 'c')
3. System recognizes you automatically
4. Output: "✓ Person RECOGNIZED: [Your Name] (92% confidence)"
```

## 📝 Common Commands

```powershell
# Interactive mode (webcam)
python main.py

# Process a saved image
python main.py photo.jpg

# List all registered people
python main.py --list

# Run tests
python test_system.py all photo.jpg
```

## ⚙️ Adjust Settings

Edit `config.py`:

```python
# Make recognition more/less strict
RECOGNITION_THRESHOLD = 0.85  # Lower = more lenient, Higher = stricter

# Change camera
CAMERA_INDEX = 0  # Try 1, 2, 3 if default doesn't work
```

## 🆘 Quick Fixes

**Camera won't open?**
```python
# In config.py, try:
CAMERA_INDEX = 1  # or 2, 3...
```

**MongoDB not connecting?**
```powershell
# Check if running:
net start MongoDB

# Or use cloud MongoDB and update config.py:
MONGODB_URI = "mongodb+srv://user:pass@cluster.mongodb.net/"
```

**Face not detected?**
```python
# In config.py, lower the threshold:
MIN_DETECTION_CONFIDENCE = 0.85  # was 0.90
```

## 📸 Best Practices

✅ Good lighting (face clearly visible)  
✅ Look directly at camera  
✅ Remove glasses if possible  
✅ Neutral expression for first registration  
✅ Distance: 50-100cm from camera  

❌ Avoid: dim lighting, extreme angles, motion blur  

## 🎯 Expected Results

| Scenario | Similarity | Outcome |
|----------|-----------|---------|
| Same person, same day | 0.95-0.99 | ✓ Recognized |
| Same person, different day | 0.85-0.95 | ✓ Recognized |
| Different person | 0.40-0.70 | ✗ Not recognized |
| Identical twin | 0.80-0.90 | ⚠ May recognize |

## 🐍 Python Version

Tested on: Python 3.8, 3.9, 3.10, 3.11

## 💡 Tips

- Register with multiple photos for better accuracy
- Use good lighting for best results
- System works offline after setup
- All data stored locally in MongoDB

---

**Need help?** Check the full [README.md](README.md) for detailed documentation.
