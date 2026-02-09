# Speed Tracking Integration Guide

## Overview
Speed tracking has been successfully integrated into `zone_alert_manager.py`. The system now tracks the speed of persons moving through the danger zone using distance estimation from bounding box heights.

## Key Components Added

### Configuration Constants
- `FPS = 30`: Video frame rate (adjust based on your video)
- `FRAME_SKIP = 5`: Calculate speed every N frames for smoothing
- `REAL_HEIGHT = 1.76`: Average human height in meters
- `PIXEL_HEIGHT_REF = 384`: Reference pixel height at camera position
- `K = REAL_HEIGHT * PIXEL_HEIGHT_REF`: Calibration constant

### New Functions

#### `estimate_distance_from_bbox(bbox)`
Estimates distance from camera based on bounding box height using the principle that:
```
distance = K / pixel_height
```
**Returns**: Distance in meters (or None if invalid)

#### `estimate_speed(track_id, frame_idx, distance, track_history)`
Calculates speed using distance history over time:
- Keeps a rolling window of 8 frames for smooth calculation
- Computes frame-to-frame speed changes
- Returns average speed in m/s

### Enhanced PersonInZone Class
Added speed tracking fields:
- `last_speed`: Last calculated speed in m/s
- `last_distance`: Last estimated distance in meters

### Enhanced ZoneAlertManager Class

**New fields:**
- `track_history`: Dictionary storing distance history per person
- `frame_idx`: Current frame counter

**Modified methods:**
- `update()`: Now calculates speed when persons are in zone and stores it
- `print_statistics()`: Displays max speed (m/s and km/h) and last distance for each violation
- `reset()`: Clears speed tracking data

## Usage in Your main.py

The speed tracking is automatic once integrated. The manager will:

1. **Estimate distance** from each person's bounding box height
2. **Track speed** during zone occupancy
3. **Store max speed** in violation history
4. **Display statistics** with speed information

Example output:
```
Person ID: 1
Entry Time:    2026-02-09 10:30:45.123
Exit Time:     2026-02-09 10:30:52.456
Duration:      7.33 seconds
Max Speed:     +1.45 m/s (+5.22 km/h)
Last Distance: 3.21 m
```

## Calibration

To calibrate for your camera setup:

1. Record a video with a person of known height (e.g., 1.76m)
2. Measure the pixel height in a frame near the camera
3. Update `REAL_HEIGHT` and `PIXEL_HEIGHT_REF` constants
4. Ensure `FPS` matches your video frame rate

## Notes

- Speed is calculated every `FRAME_SKIP` frames to reduce noise
- The speed calculation uses a sliding window of 8 frames
- Speed values are positive/negative indicating direction (approaching/leaving)
- If person height varies significantly, calibration may be needed
