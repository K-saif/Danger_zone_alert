# ZoneGuard

A real-time computer vision solution for detecting and tracking persons entering designated danger zones in video feeds. Combines YOLO11 person detection with spatial tracking and camera-perspective speed/distance estimation to provide real-time alerts and detailed analytics.

**Key Question Answered:** *"Who entered the restricted zone, for how long, how far did they move, and at what speed?"*

## Features

✅ **Real-time Person Detection & Tracking** - YOLO11-based detection with persistent ID tracking across frames  
✅ **Interactive Danger Zone Definition** - User-drawn quadrilateral polygons for custom restricted area boundaries  
✅ **Camera-Centric Distance Estimation** - Estimates distance from camera using bounding box perspective scaling (meters)  
✅ **Speed Estimation** - Tracks motion velocity with sliding window smoothing (m/s and km/h)  
✅ **Distance Traveled Tracking** - Accumulates total distance moved while person is in zone (meters)  
✅ **Zone Entry/Exit Tracking** - Detects when persons enter and leave the danger zone with frame-accurate timing  
✅ **Time Duration Logging** - Records exact entry/exit times with millisecond precision  
✅ **Real-time Visualization** - Bounding boxes with ID, distance, and speed overlaid on video  
✅ **Console Alerts** - Instant notifications with entry/exit events and statistics  
✅ **Comprehensive Statistics Report** - Violation history with duration, distance, and speed analytics  
✅ **Modular Architecture** - Clean separation of concerns for maintenance and extensibility  

## Project Structure

```
Danger_zone_alert/
├── main.py                      # Entry point; video loop & output management
├── config.py                    # Configuration parameters
├── zone_alert_manager.py        # Core logic; detection, tracking, speed/distance, alerts
├── quadrilateral_tracker.py     # Zone polygon drawing & spatial checking
├── utils.py                     # Logging, video writer, utilities
├── data/                        # Video files
│   └── 0_te21_cropped.mp4
├── output/                      # Generated output videos
└── README.md
```

## Installation

### Requirements

- Python 3.8+
- OpenCV (`cv2`)
- YOLOv11 (`ultralytics`)
- NumPy

### Setup

1. Clone or download the project
2. Install dependencies:

```bash
pip install opencv-python ultralytics numpy
```

3. Place your video file in the `data/` directory

## Usage

### Running the System

```bash
python main.py
```

### Step-by-Step Guide

1. **Start the System**
   - The system will load YOLOv11 and display the first video frame
   - You'll see the quadrilateral drawing interface

2. **Draw the Danger Zone**
   - **Left-click** to add 4 points defining the danger zone perimeter
   - **Right-click** to remove the last point if needed
   - **Press SPACE** to confirm when all 4 points are added
   - **Press ESC** to cancel

3. **Preview**
   - The first frame will be displayed with your danger zone highlighted
   - Yellow border shows the zone boundary
   - Light cyan overlay shows the monitored area

4. **Video Processing**
   - The system processes the entire video
   - Green circles show persons outside the danger zone
   - Red circles show persons inside the danger zone
   - Red "DANGER ZONE!" alert appears on-screen when violations occur

5. **Results**
   - After video processing completes, detailed violation statistics are displayed
   - Shows entry time, exit time, and duration for each violation
   - Press **Q** to stop video processing at any time

## Configuration & Calibration

Edit `config.py` to customize the system:

```python
# Video Configuration
VIDEO_PATH = "/path/to/your/video.mp4"
OUTPUT_PATH = None  # Set to save processed video

# YOLO Configuration
YOLO_MODEL = "yolo11n.pt"
YOLO_CLASS = [0]  # 0 = person class
YOLO_CONFIDENCE = 0.4

# Display Colors
SAFE_POINT_COLOR = (0, 255, 0)      # Green
DANGER_POINT_COLOR = (0, 0, 255)    # Red
MASK_COLOR = (200, 200, 100)        # Light cyan
BORDER_COLOR = (0, 255, 255)        # Yellow
```

### Distance & Speed Calibration

In `zone_alert_manager.py`, adjust the calibration constants:

```python
FPS = 30                    # Match your video frame rate (adjust if needed)
FRAME_SKIP = 5              # Calculate speed every N frames
REAL_HEIGHT = 1.76          # Average person height (meters)
PIXEL_HEIGHT_REF = 384      # Bbox height at reference distance (pixels)
```

**Calibration Procedure:**

1. Select a frame where a known-height person is at a measurable distance from the camera
2. Measure their bounding box height in pixels using an image viewer
3. Update `PIXEL_HEIGHT_REF` to that value
4. Verify distance estimates match expectations
5. Test with a walk-through: person walks toward and away from camera, verify speed is reasonable

**Example:** If a 1.76m person has a 200-pixel bbox in a reference frame:
```python
REAL_HEIGHT = 1.76
PIXEL_HEIGHT_REF = 200
```

## How It Works

### Processing Pipeline

1. **Frame Capture** - Read frame from video file
2. **YOLO Tracking** - Detect persons and assign persistent track IDs
3. **Zone Classification** - Check if each detection is inside danger zone (polygon)
4. **Spatial Analysis** (in-zone only):
   - Estimate distance from camera via bounding box height
   - Calculate speed from distance history (sliding window)
   - Accumulate total distance traveled
5. **Alert Management** - Track entry/exit events and log violations
6. **Visualization** - Annotate frame with zones, boxes, IDs, distance, and speed
7. **Output** - Write annotated video and print statistics

### Distance Estimation (Perspective Scaling)

Distance is estimated using the **perspective projection principle**: a person's apparent size (bounding box height in pixels) is inversely proportional to their distance from the camera.

**Formula:**
```
Distance (meters) = K / bbox_height (pixels)
where K = REAL_HEIGHT × PIXEL_HEIGHT_REF
```

**Parameters:**
- `REAL_HEIGHT` = 1.76 m (average human height, configurable)
- `PIXEL_HEIGHT_REF` = 384 px (height of person's bbox at reference distance)
- `K` = 675.84 (pre-computed calibration constant)

**Key Points:**
- Assumes persons are roughly upright
- Requires camera calibration for accuracy
- Typically ±10-20% error at medium distances (3-10m)
- Degrades at extreme distances or image edges

### Speed Estimation (Sliding Window)

Speed is computed from distance history using a **rolling window of 8 frames**:

```
For each frame:
  1. Record current frame and distance
  2. Keep only last 8 frames of history
  3. Compute speed between consecutive frames
  4. Return smoothed average speed
```

**Parameters:**
- `FRAME_SKIP` = 5 (update calculation every 5 frames for stability)
- `WINDOW` = 8 (history length for smoothing)
- `FPS` = 30 (must match your video frame rate)

**Output:** Speed in m/s and km/h; absolute value (direction implicit)

### Zone Detection

1. **Bottom Center Point** - Uses the center point of the bottom edge of each person's bounding box
   - Represents the foot position on ground plane
   - Most accurate for detecting zone entry/exit

2. **Point-in-Polygon Algorithm** - Uses OpenCV's `pointPolygonTest()` to determine if bottom center is inside zone

3. **Entry/Exit Tracking**
   - Entry: Bottom center point enters the quadrilateral
   - Exit: Bottom center point leaves the quadrilateral
   - Times recorded with millisecond precision

## Output Information

### Real-time Console Alerts

```
[10:30:45] 🚨 ALERT! Person (ID: 1) entered danger zone!
[10:30:52] ⚠ Person (ID: 1) left danger zone (Duration: 7.33s, Distance: 5.47m)
```

### Video Frame Overlay

```
[RED BOX]   ID 1 | 3.21m | 1.45m/s
[GREEN BOX] ID 2 | 8.50m
[RED ZONE]  Polygon boundary of danger zone
```

### Violation Details Report

```
================================================================================
DANGER ZONE VIOLATION DETAILS
================================================================================

Violation #1
----------------
Person ID:         1
Entry Time:        2026-02-09 10:30:45.123
Exit Time:         2026-02-09 10:30:52.456
Duration:          7.33 seconds (7 sec 330 ms)
Distance Traveled: 5.47 meters
Max Speed:         1.45 m/s (5.22 km/h)

================================================================================
Total Violations: 1
================================================================================
```

**Information Provided:**
- `Person ID` - Unique tracking identifier
- `Entry Time` - Exact timestamp when person entered (YYYY-MM-DD HH:MM:SS.ms)
- `Exit Time` - Exact timestamp when person exited (YYYY-MM-DD HH:MM:SS.ms)
- `Duration` - Time spent in danger zone
- `Distance Traveled` - Total distance moved while in zone
- `Max Speed` - Highest speed detected during zone occupancy

## Keyboard Controls

| Key | Action |
|-----|--------|
| Left-Click | Add quadrilateral point (during drawing) |
| Right-Click | Remove last point (during drawing) |
| SPACE | Confirm quadrilateral (after 4 points added) |
| ESC | Cancel quadrilateral drawing |
| Q | Stop video processing |

## Assumptions & Limitations

### Key Assumptions

1. **Upright Persons** - System assumes persons are roughly vertical; significant tilting invalidates distance estimates
2. **Fixed Camera** - Camera is stationary and calibrated once; movement requires re-calibration
3. **Consistent Height** - Calibration assumes typical human height (~1.76m); large deviations cause errors
4. **Perspective Geometry** - Standard perspective projection; no lens distortion compensation
5. **Known Frame Rate** - Must match actual video FPS for accurate speed
6. **Detection Quality** - YOLO detection is prerequisite; poor lighting or occlusion degrades all metrics

### Accuracy Limitations

| Metric | Typical Accuracy | Notes |
|--------|------------------|-------|
| **Distance** | ±10-20% | Better at 3-10m; worse at extremes |
| **Speed** | ±15-25% | Smoothed by 8-frame window |
| **Duration** | ±1-2 frames | Frame-accurate (depends on FPS) |
| **Zone Entry/Exit** | ~1 frame latency | Depends on YOLO detection lag |

### Known Limitations

- **No 3D Reconstruction** - Estimates are relative to single camera view only
- **Camera-Space Only** - No ground truth validation against GPS or external sensors
- **Occlusion Handling** - Partially occluded persons have underestimated bbox height
- **Perspective Distortion** - Objects at image edges have lower accuracy
- **Scene-Specific Calibration** - Requires manual calibration per camera setup
- **Single Camera** - No multi-view triangulation for improved accuracy
- **Motion Blur** - Fast motion may cause detection misses or bbox jitter


---

## Future Enhancements

- [ ] Multiple zone support in single video
- [ ] Configurable alert thresholds (speed, distance, duration limits)
- [ ] Zone heat maps and occupancy analytics
- [ ] CSV/JSON export of violation history
- [ ] Persistent track ID logging across sessions
- [ ] Adaptive YOLO model selection based on accuracy needs
- [ ] Lens distortion correction
- [ ] Temporal smoothing (Kalman filter)
- [ ] Crowd density estimation
- [ ] Cloud integration for remote monitoring
