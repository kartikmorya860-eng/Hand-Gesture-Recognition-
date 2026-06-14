# Hand Gesture Recognition

A simple hand gesture recognition project using OpenCV and NumPy.

## Overview

This project captures video from the webcam, detects a hand in a fixed region of interest, and counts convexity defects to recognize simple gestures.

The current implementation maps gestures to medicine labels:

- 0 defects: `Medicine: Paracetamol`
- 1 defect: `Medicine: Ibuprofen`
- 2 defects: `Medicine: Aspirin`

## Files

- `code.py` - main Python script for webcam capture, hand detection, and gesture mapping.
- `Hand Gusture report.pdf` - project report.

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Installation

```bash
pip install opencv-python numpy
```

## Usage

```bash
python code.py
```

Then place your hand inside the green box and use the gesture count to see the mapped label.

Press `q` to quit.

## Notes

- The ROI is a fixed box on the screen at coordinates `100:400`.
- Skin color detection uses an HSV threshold, so lighting conditions will affect performance.
- This is a simple demo and may require tuning for robust gesture recognition.
