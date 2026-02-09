"""
Configuration Module
Central configuration for the danger zone alert system.
"""

# Video configuration
VIDEO_PATH = "./Danger_zone_alert/data/0_te21_cropped.mp4"
OUTPUT_PATH = None  # Set to a path to save output video, None to skip

# YOLO configuration
YOLO_MODEL = "yolo11n.pt"
YOLO_CLASS = [0]  # Class 0 = person
YOLO_CONFIDENCE = 0.4
YOLO_PERSIST = True

# Quadrilateral mask configuration
MASK_ALPHA = 0.25  # Transparency of the mask (0.0 to 1.0)
MASK_COLOR = (200, 200, 100)  # BGR color for the mask overlay (light cyan)
BORDER_COLOR = (0, 255, 255)  # BGR color for the border (yellow)

# Zone detection configuration
SAFE_POINT_COLOR = (0, 255, 0)  # Green circle for safe zone
DANGER_POINT_COLOR = (0, 0, 255)  # Red circle for danger zone
SAFE_POINT_RADIUS = 6
DANGER_POINT_RADIUS = 8

# Display configuration
DISPLAY_WINDOW_NAME = "Danger Zone Alert System"
DISPLAY_FPS = 30

# Alert configuration
SHOW_ALERTS = True
SAVE_STATISTICS = True
STATISTICS_FILE = "zone_violations_statistics.txt"
