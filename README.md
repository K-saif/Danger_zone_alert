# Danger Zone Alert System

A real-time video monitoring system that detects and alerts when people enter a defined danger zone. Built with YOLOv11 for person detection and OpenCV for zone management.

## Features

✅ **Interactive Quadrilateral Drawing** - Easily define danger zones by drawing a 4-point polygon on the first video frame  
✅ **Real-time Person Detection** - Uses YOLOv11 for accurate person detection  
✅ **Zone Entry/Exit Tracking** - Tracks when persons enter and leave the danger zone  
✅ **Time Duration Tracking** - Records exact entry and exit times with millisecond precision  
✅ **Visual Alerts** - Red highlights for persons in danger zone, green for safe areas  
✅ **Console Alerts** - Instant notifications with timestamps when violations occur  
✅ **Detailed Violation Reports** - Complete history of all zone violations with timestamps  
✅ **Modular Architecture** - Clean code separation for easy maintenance and extensibility  

## Project Structure

```
Danger_zone_alert/
├── main.py                      # Main entry point
├── config.py                    # Configuration parameters
├── quadrilateral_tracker.py     # Quadrilateral drawing and zone geometry
├── zone_alert_manager.py        # Zone detection and alert management
├── utils.py                     # Helper utilities and logging
├── data/                        # Video files directory
│   └── 0_te21_cropped.mp4
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

## Configuration

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

## How It Works

### Zone Detection

1. **Bottom Center Point** - The system uses the center point of the bottom edge of each person's bounding box as the detection point
   - This mimics where a person's feet contact the ground
   - Most accurate for detecting zone entry/exit

2. **Point-in-Polygon Algorithm** - Uses OpenCV's `pointPolygonTest()` to determine if the bottom center point is inside the danger zone

3. **Entry/Exit Tracking**
   - Entry: Person's bottom center enters the quadrilateral
   - Exit: Person's bottom center leaves the quadrilateral
   - Times are recorded with millisecond precision

### Alert System

**Console Alerts:**
```
[17:29:12] 🚨 ALERT! Person (ID: 1) entered danger zone!
[17:29:40] ⚠ Person (ID: 1) left danger zone (Duration: 27.74s)
```

**Visual Alerts:**
- Red box with "DANGER ZONE!" text appears when any person is in zone
- Bottom center point changes from green (safe) to red (danger)

## Output Information

### Violation Details Report

```
================================================================================
DANGER ZONE VIOLATION DETAILS
================================================================================

Violation #1
--------------------------------------------------------------------------------
Person ID: 1
Entry Time:    2026-02-08 17:34:16.686
Exit Time:     2026-02-08 17:34:41.778
Duration:      25 sec 92 ms (25.09 seconds)

================================================================================
Total Violations: 1
================================================================================
```

**Information Provided:**
- `Person ID` - Unique tracking ID assigned by YOLO
- `Entry Time` - Exact timestamp when person entered (YYYY-MM-DD HH:MM:SS.ms)
- `Exit Time` - Exact timestamp when person exited (YYYY-MM-DD HH:MM:SS.ms)
- `Duration` - Time spent in danger zone (seconds and milliseconds)

## Module Documentation

### `quadrilateral_tracker.py`
Handles interactive quadrilateral drawing and zone masking.

**Key Methods:**
- `get_first_frame()` - Extract first video frame
- `draw_quadrilateral()` - Interactive 4-point polygon drawing
- `is_bbox_in_zone(bbox)` - Check if bounding box is in zone
- `apply_quadrilateral_mask(image, alpha)` - Draw zone visualization

### `zone_alert_manager.py`
Manages zone violations, entry/exit tracking, and alerts.

**Key Classes:**
- `PersonInZone` - Tracks a single person's zone entry/exit
- `ZoneAlertManager` - Manages all detection and alerting

**Key Methods:**
- `update(detection_results, frame)` - Process frame detections
- `finalize_zone_exits()` - Mark remaining persons as exited at video end
- `print_statistics()` - Display violation details

### `utils.py`
Helper utilities for logging and video processing.

**Key Classes:**
- `VideoWriter` - Write processed video to file
- `Logger` - Unified logging system with timestamps

### `config.py`
Central configuration file for all system parameters.

## Keyboard Controls

| Key | Action |
|-----|--------|
| Left-Click | Add quadrilateral point (during drawing) |
| Right-Click | Remove last point (during drawing) |
| SPACE | Confirm quadrilateral (after 4 points added) |
| ESC | Cancel quadrilateral drawing |
| Q | Stop video processing |

## Troubleshooting

### YOLO Model Download
If the YOLO model fails to download:
```bash
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

### Video Not Found
- Check that `VIDEO_PATH` in `config.py` is correct
- Use absolute file paths
- Ensure video file format is supported by OpenCV

### No Detections
- Adjust `YOLO_CONFIDENCE` in `config.py` (lower = more detections)
- Ensure video resolution is adequate
- Check lighting conditions in video

### Performance Issues
- Use smaller YOLO models (yolo11n is nano, lightweight)
- Reduce video resolution
- Skip output video writing (set `OUTPUT_PATH = None`)

## Performance Metrics

**Tested Configuration:**
- Video Resolution: 910x1080
- Video FPS: 24.14
- YOLO Model: yolo11n (nano)
- Processing Speed: Real-time on modern CPUs

## Future Enhancements

- [ ] Multiple zone support
- [ ] Polygon zone editing during playback
- [ ] Email/SMS alerts
- [ ] Recording violations to database
- [ ] Web dashboard for monitoring
- [ ] Motion prediction for proactive alerts

## License

This project is provided as-is for safety and monitoring purposes.

## Support

For issues or questions, check the configuration in `config.py` and ensure all dependencies are properly installed.

---

**Version:** 1.0  
**Last Updated:** February 8, 2026