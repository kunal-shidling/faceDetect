# Quick Reference - Facial Recognition System

## 🚀 Quick Start Commands

```powershell
# 1️⃣ CONTINUOUS MONITORING (Best for meetings!)
python continuous.py

# 2️⃣ Single face capture
python main.py

# 3️⃣ Process saved image
python main.py photo.jpg

# 4️⃣ List all registered people
python main.py --list
```

## 📊 Mode Comparison

| Feature | Continuous Mode | Interactive Mode |
|---------|----------------|------------------|
| **Command** | `python continuous.py` | `python main.py` |
| **Camera** | Always on | Press 'c' to capture |
| **Faces** | Multiple at once | One at a time |
| **Display** | Green/Red boxes | Text output |
| **Best For** | Meetings, monitoring | Registration |
| **Speed** | Real-time | On-demand |

## 🎨 Visual Indicators (Continuous Mode)

- 🟢 **Green Box + Name** → Person recognized
- 🔴 **Red Box "Unknown"** → Person not in database
- 📊 **Top-left corner** → FPS, face count, total registered

## 🎯 Common Tasks

### Register Your First Person
```powershell
# Method 1: Interactive (easier)
python main.py
# Press 'c', enter name when prompted

# Method 2: From saved photo
python main.py your_photo.jpg
# Enter name when prompted
```

### Monitor a Meeting Room
```powershell
python continuous.py
# Just let it run - it will recognize everyone!
```

### Check Who's Registered
```powershell
python main.py --list
```

## ⚙️ Quick Config Changes

Edit `config.py`:

```python
# Make recognition more lenient
RECOGNITION_THRESHOLD = 0.80  # Default: 0.85

# Use external webcam
CAMERA_INDEX = 1  # Default: 0

# MongoDB Atlas connection
MONGODB_URI = "mongodb+srv://user:pass@cluster.mongodb.net/..."
```

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Camera not opening | Try `CAMERA_INDEX = 1` in config.py |
| Database error | Check MongoDB Atlas connection string |
| Face not detected | Better lighting, face camera directly |
| Low accuracy | Lower `RECOGNITION_THRESHOLD` to 0.80 |

## 📦 What Each File Does

```
continuous.py     ← Real-time multi-face monitoring
main.py          ← Single face capture & registration
detector.py      ← Face detection (MTCNN)
embedder.py      ← Face embeddings (FaceNet)
recognizer.py    ← Face matching (cosine similarity)
database.py      ← MongoDB storage
config.py        ← All settings (EDIT THIS!)
```

## 🎓 Example Workflow

**Day 1: Register Team**
```powershell
# Register each person
python main.py
# Person 1 presses 'c', enters name
# Person 2 presses 'c', enters name
# Person 3 presses 'c', enters name
```

**Day 2: Monitor Meeting**
```powershell
# Start continuous monitoring
python continuous.py
# System automatically shows green boxes with names
# for everyone in the meeting!
```

## 💡 Pro Tips

1. **Register with good lighting** - Makes recognition more accurate
2. **Multiple angles** - Register same person from different angles
3. **Continuous mode** - Best for 2+ people scenarios
4. **Interactive mode** - Best for one-on-one registration
5. **Batch mode** - Process photos you already have

## 🔗 Full Documentation

- [RUNNING.md](RUNNING.md) - Complete setup guide for collaborators
- [README.md](README.md) - Technical documentation
- [QUICKSTART.md](QUICKSTART.md) - 5-minute setup

---

**Questions?** Check the full [RUNNING.md](RUNNING.md) guide!
