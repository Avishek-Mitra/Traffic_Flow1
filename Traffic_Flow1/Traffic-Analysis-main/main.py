import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import Counter

# SETTINGS 

VIDEO_FILE = "Traffic.mp4"   # Input video file name (must be in same folder)
OUTPUT_FILE = "output.mp4"   # Processed video output file
CSV_FILE = "vehicle_output.csv"  # Vehicle log CSV
FAST_MODE = True  # True = only process first 300 frames for testing

# CHECK VIDEO FILE 

if not os.path.exists(VIDEO_FILE):
    print(f" Error: {VIDEO_FILE} not found!")
    exit()
else:
    print(f" Found video: {VIDEO_FILE}")

# LOAD YOLO MODEL
print("Loading YOLO model...")
model = YOLO("yolov8n.pt")  # Small model for faster performance

#  TRACKER INITIALIZATION 
tracker = DeepSort(max_age=30)

# OPEN VIDEO
cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    print(f" Error: Could not open {VIDEO_FILE}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video info - FPS: {fps}, Width: {width}, Height: {height}")

# OUTPUT VIDEO WRITER 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (width, height))

#  DEFINE LANES 
lane1_x = int(width * 1 / 3)
lane2_x = int(width * 2 / 3)

def draw_lanes(frame):
    """Draw lane divider lines."""
    cv2.line(frame, (lane1_x, 0), (lane1_x, height), (0, 255, 0), 2)
    cv2.line(frame, (lane2_x, 0), (lane2_x, height), (0, 255, 0), 2)
    return frame

# LOGS & COUNTS 
vehicle_log = {}
frame_count = 0
MAX_FRAMES = 300 if FAST_MODE else float('inf')
lane_counts = {1: 0, 2: 0, 3: 0}

# PROCESS VIDEO
while True:
    ret, frame = cap.read()
    if not ret or frame_count >= MAX_FRAMES:
        print("Video processing finished.")
        break

    frame_count += 1
    print(f"Processing frame {frame_count}...")

    frame = draw_lanes(frame)
    results = model(frame, verbose=False)[0]

    # Collect detections for tracker
    detections_for_tracker = []
    for box in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = box
        if int(cls) in [2, 3, 5, 7]:  # Car, Motorcycle, Bus, Truck
            w_box, h_box = x2 - x1, y2 - y1
            detections_for_tracker.append(([x1, y1, w_box, h_box], conf, "vehicle"))

    # Track vehicles
    tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x_center = int((ltrb[0] + ltrb[2]) / 2)

        # Determine lane
        if x_center < lane1_x:
            lane = 1
        elif x_center < lane2_x:
            lane = 2
        else:
            lane = 3

        # Log vehicle only once
        if track_id not in vehicle_log:
            vehicle_log[track_id] = {
                'lane': lane,
                'frame': frame_count,
                'time': round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 2)
            }
            lane_counts[lane] += 1

        # Draw bounding box
        cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])),
                      (int(ltrb[2]), int(ltrb[3])), (255, 0, 0), 2)
        cv2.putText(frame, f"ID:{track_id} Lane:{lane}",
                    (int(ltrb[0]), int(ltrb[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Show live lane counts
    cv2.putText(frame, f"Lane 1: {lane_counts[1]} vehicles", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f"Lane 2: {lane_counts[2]} vehicles", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Lane 3: {lane_counts[3]} vehicles", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    out.write(frame)
    cv2.imshow("Traffic Flow Analysis", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to stop early
        break

# SAVE & CLEANUP 
cap.release()
out.release()
cv2.destroyAllWindows()

# Save vehicle log to CSV
df = pd.DataFrame([
    {'Vehicle ID': vid, 'Lane': data['lane'], 'Frame': data['frame'], 'Time': data['time']}
    for vid, data in vehicle_log.items()
])
df.to_csv(CSV_FILE, index=False)

# Print summary
summary = Counter([data['lane'] for data in vehicle_log.values()])
print("\nVehicle Count per Lane:")
for lane in sorted(summary):
    print(f"Lane {lane}: {summary[lane]} vehicles")

print(f"\n Output video saved as '{OUTPUT_FILE}'")
print(f"CSV saved as '{CSV_FILE}'")