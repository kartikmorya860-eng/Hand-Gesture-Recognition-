# Hand Gesture Recognition

A webcam-based hand gesture demo powered by OpenCV and NumPy.

## 🚀 What it does

This project captures live video from your webcam, detects a hand in the frame, and estimates the number of convexity defects to display a medicine recommendation label.

### Gesture-to-medicine mapping

- 0 defects: `Medicine: Paracetamol`
- 1 defect: `Medicine: Ibuprofen`
- 2 defects: `Medicine: Aspirin`
- 3+ defects: `Medicine: Consult doctor`

## ✨ Updated behavior

- Uses the full flipped video frame for more reliable detection
- Draws a green guidance box so you know where to place your hand
- Displays a clear status message when detection is not ready
- Improves skin mask preprocessing and contour filtering for better accuracy

## Files

- `main.py` - main Python script for webcam capture, hand detection, and medicine mapping.
- `test_cam.py` - camera diagnostic helper to verify webcam access and capture.

## Requirements

- Python 3.8+ or similar
- OpenCV
- NumPy

## Installation

```bash
pip install opencv-python numpy
```

## Usage

```bash
python main.py
```

Then:

1. Allow the script to open your webcam.
2. Place your hand inside the green box.
3. Watch the medicine label appear at the top of the window.
4. Press `q` to quit.

## Tips for best results

- Use moderate indoor lighting.
- Keep your hand centered inside the green box.
- Avoid strong shadows and very bright backlighting.
- Move closer if the app shows `Move hand closer to camera`.

## Troubleshooting

- If the camera does not open, close other apps using the webcam and try again.
- If detection is unstable, adjust lighting or reposition your hand.

## Disclaimer

This is a demo application for gesture detection and UI mapping only. It is not a medical diagnosis tool.
