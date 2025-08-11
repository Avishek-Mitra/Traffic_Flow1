# Traffic Analysis using YOLOv8 and DeepSORT

This project detects and tracks vehicles in a traffic video, counts vehicles per lane, and generates both a processed output video and a CSV log.

---

## Features
- **Vehicle Detection** using YOLOv8.
- **Vehicle Tracking** with DeepSORT.
- **Lane Detection** with live per-lane counts.
- **CSV Export** with vehicle ID, lane, frame number, and timestamp.
- **ESC Key Support** to stop processing early.

---

## Project Structure

Traffic Analysis/
1. main.py # Main Python script
2. Traffic.mp4 # Input video (place your own here)
3. output.mp4 # Processed output video (generated after run)
4. vehicle_output.csv # CSV log of detected vehicles (generated after run)
5. README.md # Project documentation


---

## Installation

1. **Clone or download** this project folder.
2. **Open a terminal** in the project directory.
3. **Install dependencies**:
   ```bash
   pip install ultralytics opencv-python deep-sort-realtime pandas numpy


## Usage

Place your traffic video in the project folder and rename it:
Traffic.mp4

## Run the script:
python main.py

Wait for processing to complete.

Processed video → output.mp4
Vehicle log → vehicle_output.csv

## Notes
1. Press ESC to stop the process early.
2. ake sure your video file name matches exactly (Traffic.mp4 with capital T if used in code).

## For faster testing, set:
FAST_MODE = True
in main.py to process only 300 frames.


## Output Example
Terminal Summary:
Vehicle Count per Lane:
Lane 1: 15 vehicles
Lane 2: 9 vehicles
Lane 3: 12 vehicles

 Output video saved as 'output.mp4'
CSV saved as 'vehicle_output.csv'


## CSV File:

Vehicle ID	Lane	Frame   	Time (s)
1	         2  	34	      1.13
2	         1	   56	      1.87
3	         3	   78	      2.60


## Requirements
Python 3.13+

YOLOv8 model file will be auto-downloaded on first run.

# Notes
1. For better accuracy, use yolov8s.pt or yolov8m.pt instead of yolov8n.pt.
2. Ensure your video resolution is good enough for detection.
3. Works best on roads with visible lane boundaries.


Author: Ronit Mitra
Language: Python 3.13.3
