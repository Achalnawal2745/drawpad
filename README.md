# 🎨 Hand Gesture Drawing Pad

A real-time hand gesture-controlled drawing application using **OpenCV** and **MediaPipe**. Draw, erase, and control the entire interface using just your webcam and hand gestures!


## ✨ Features

- 🖐️ **Advanced Hand Gesture Recognition** - 5 distinct gesture modes
- 🎨 **6 Color Options** - Pink, Blue, Green, Orange, Purple, White
- 🖱️ **Cursor Mode** - Control UI with hand gestures (2-second dwell click)
- 🧹 **Eraser Tool** - Toggle between drawing and erasing
- ↩️ **Undo Functionality** - Undo up to 20 strokes
- 📏 **Adjustable Brush Size** - Gesture-based thickness control
- ⌨️ **Keyboard Shortcuts** - Fast control with hotkeys
- 💾 **Modern Sidebar UI** - Clean, glassmorphic interface
- 🎯 **High Accuracy** - Improved finger detection algorithm



## 📋 Prerequisites

- **Python 3.10 or 3.11** (MediaPipe doesn't support 3.13 yet)
- **Webcam** (built-in or external)
- **Good lighting** for optimal hand tracking

## 🚀 Installation

### Option 1: Using Anaconda (Recommended)

```bash
conda create -n drawpad python=3.10 -y

conda activate drawpad

cd drawpad

pip install -r requirements.txt
```

### Option 2: Using venv

```bash

python -m venv venv

venv\Scripts\activate

source venv/bin/activate

pip install -r requirements.txt
```

## 🎮 Usage

### Running the Application

```bash
python main.py
```

### 🖐️ Hand Gestures

| Gesture | Fingers | Action |
|---------|---------|--------|
| ✌️ **Cursor Mode** | Index + Middle | Control UI, hover to click buttons |
| 👆 **Draw** | Index only | Draw on canvas |
| 🤘 **Standby** | Index + Pinky | Pause drawing |
| ✊ **Clear** | Closed fist | Clear entire canvas |
| 🤏 **Adjust Size** | Thumb + Index (+ Pinky to confirm) | Change brush thickness |

### ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `C` | Toggle Cursor Mode |
| `E` | Toggle Eraser |
| `U` | Undo last stroke |
| `X` | Clear canvas |
| `1` | Pink color |
| `2` | Blue color |
| `3` | Green color |
| `4` | Orange color |
| `5` | Purple color |
| `6` | White color |
| `Q` | Quit application |

## 🖱️ Cursor Mode Guide

**How to Click Buttons with Hand Gestures:**

1. ✌️ Raise **Index + Middle** fingers (keep others down)
2. A **green cursor** appears on screen
3. Move your hand to position the cursor over a button
4. **Hold still** for 2 seconds
5. Watch the **green ring** fill up around the cursor
6. Button **auto-clicks** when ring completes!

**Tip:** Use keyboard shortcuts for faster control!

## 📁 Project Structure

```
drawpad/
├── main.py              # Main application
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── screenshots/        # Demo images (optional)
```

## 🛠️ Dependencies

```
opencv-python>=4.11.0
mediapipe>=0.10.21
numpy>=1.26.4
```

## 💡 Tips for Best Performance

- ✅ Use **good lighting** (natural or bright artificial light)
- ✅ Keep your **hand flat** and visible to the camera
- ✅ Make **clear finger positions** (strict detection)
- ✅ Position yourself **1-2 feet** from the camera
- ✅ Use a **plain background** for better tracking
- ✅ Avoid **fast movements** for smoother drawing

## 🐛 Troubleshooting

### MediaPipe Installation Error
```bash
# Make sure you're using Python 3.10 or 3.11
python --version

# If using 3.13, create new environment with 3.10
conda create -n drawpad python=3.10 -y
```

### Camera Not Opening
```bash
# Try changing camera index in main.py (line 168)
cap = cv2.VideoCapture(0)  # Try 0, 1, or 2
```

### Hand Not Detected
- Improve lighting
- Move closer to camera
- Ensure hand is fully visible
- Check camera permissions


