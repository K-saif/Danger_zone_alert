"""
Zone Alert Manager Module
Manages detection, alerting, and tracking of persons in danger zones.
"""

import cv2
import time
from datetime import datetime
from collections import defaultdict

# ================================
# Speed Tracking Configuration
# ================================
FPS = 30  # Frame per second of video
FRAME_SKIP = 5  # Frames to skip between speed calculations

# Calibration constants for distance estimation
# These are based on height of person and pixel height at reference distance
REAL_HEIGHT = 1.76  # Average human height in meters
PIXEL_HEIGHT_REF = 384  # Pixel height of person near camera
K = REAL_HEIGHT * PIXEL_HEIGHT_REF  # Calibration constant


# ================================
# Speed Tracking Functions
# ================================
def estimate_distance_from_bbox(bbox):
    """
    Estimate distance from camera based on bounding box height.
    
    Args:
        bbox: Bounding box as (x1, y1, x2, y2)
    
    Returns:
        float: Estimated distance in meters, or None if invalid
    """
    x1, y1, x2, y2 = bbox
    pixel_height = y2 - y1
    if pixel_height <= 0:
        return None
    return K / pixel_height  # meters


def estimate_speed(track_id, frame_idx, distance, track_history):
    """
    Estimate speed of person based on distance history.
    
    Args:
        track_id: Unique tracking ID
        frame_idx: Current frame index
        distance: Current distance in meters
        track_history: Dictionary of track_id -> [(frame_idx, distance)]
    
    Returns:
        float: Estimated speed in m/s, or None if insufficient data
    """
    history = track_history[track_id]
    history.append((frame_idx, distance))
    
    # Keep last N frames
    WINDOW = 8
    if len(history) > WINDOW:
        history.pop(0)
    
    if len(history) < 2:
        return None
    
    speeds = []
    for i in range(1, len(history)):
        f0, d0 = history[i - 1]
        f1, d1 = history[i]
        
        dt = (f1 - f0) / FPS
        if dt > 0:
            speeds.append((d1 - d0) / dt)
    
    if not speeds:
        return None
    
    return sum(speeds) / len(speeds)


class PersonInZone:
    """Represents a person detected in the danger zone"""
    
    def __init__(self, track_id, entry_time):
        """
        Initialize a person in zone record.
        
        Args:
            track_id (int): Unique tracking ID
            entry_time (float): Timestamp when person entered the zone
        """
        self.track_id = track_id
        self.entry_time = entry_time
        self.exit_time = None
        self.duration = 0
        self.alert_shown = False
        
        # Speed tracking fields
        self.last_speed = None  # Last calculated speed in m/s
        self.last_distance = None  # Last estimated distance in meters
        self.total_distance = 0  # Total distance traveled while in zone (meters)
        self.prev_distance = None  # Previous distance measurement for accumulation
    
    def get_duration(self, current_time=None):
        """
        Get duration the person has been in the zone.
        
        Args:
            current_time (float): Current timestamp (uses time.time() if None)
        
        Returns:
            float: Duration in seconds
        """
        if current_time is None:
            current_time = time.time()
        
        if self.exit_time is not None:
            self.duration = self.exit_time - self.entry_time
            return self.duration
        else:
            return current_time - self.entry_time
    
    def mark_exit(self, exit_time=None):
        """
        Mark when the person exited the zone.
        
        Args:
            exit_time (float): Exit timestamp (uses time.time() if None)
        """
        if exit_time is None:
            exit_time = time.time()
        self.exit_time = exit_time
        self.duration = exit_time - self.entry_time


class ZoneAlertManager:
    """
    Manages alerts and tracking for persons in danger zones.
    
    Features:
    - Track persons entering and leaving zones
    - Record time spent in zone
    - Generate visual and console alerts
    - Maintain history of zone violations
    """
    
    def __init__(self, quadrilateral_tracker):
        """
        Initialize the zone alert manager.
        
        Args:
            quadrilateral_tracker: QuadrilateralTracker instance
        """
        self.tracker = quadrilateral_tracker
        self.persons_in_zone = {}  # {track_id: PersonInZone}
        self.alert_history = []  # List of all zone violation records
        self.current_in_zone = {}  # {track_id: True/False}
        
        # Speed tracking
        self.track_history = defaultdict(list)  # {track_id: [(frame_idx, distance)]}
        self.frame_idx = 0  # Current frame index
    
    def update(self, detection_results, frame):
        """
        Update zone detection based on YOLO detection results.
        
        Args:
            detection_results: YOLO detection results
            frame (np.ndarray): Current video frame
        
        Returns:
            tuple: (annotated_frame, alerts_triggered)
        """
        annotated_frame = frame.copy()
        alerts_triggered = []
        current_time = time.time()
        
        # Track which IDs were detected in this frame
        detected_ids = set()
        
        # Process each detection
        if detection_results.boxes is not None:
            for box in detection_results.boxes:
                bbox = box.xyxy[0].cpu().numpy()
                track_id = int(box.id[0]) if box.id is not None else None
                
                if track_id is not None:
                    detected_ids.add(track_id)
                
                # Check if in zone first
                is_in_zone = self.tracker.is_bbox_in_zone(bbox)
                
                # Estimate distance ONLY if in zone
                distance = None
                if is_in_zone:
                    distance = estimate_distance_from_bbox(bbox)
                
                # Draw bounding box and labels
                x1, y1, x2, y2 = map(int, bbox)
                
                if is_in_zone:
                    # Red box for persons in zone
                    box_color = (0, 0, 255)
                else:
                    # Green box for persons outside zone
                    box_color = (0, 255, 0)
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                
                # Prepare label
                label_parts = [f"ID {track_id}"] if track_id is not None else []
                
                if is_in_zone:
                    # Draw the bottom center point in red
                    annotated_frame = self.tracker.draw_bbox_bottom_center(
                        annotated_frame, bbox, color=(0, 0, 255), radius=8
                    )
                    
                    # Handle first time entering zone
                    if track_id is not None:
                        if track_id not in self.persons_in_zone:
                            # New person entering zone
                            self.persons_in_zone[track_id] = PersonInZone(track_id, current_time)
                            alerts_triggered.append({
                                'type': 'ENTRY',
                                'track_id': track_id,
                                'timestamp': current_time,
                                'message': f"🚨 ALERT! Person (ID: {track_id}) entered danger zone!"
                            })
                        
                        self.current_in_zone[track_id] = True
                        
                        # Calculate distance and speed for persons in zone
                        if distance is not None:
                            person = self.persons_in_zone[track_id]
                            person.last_distance = distance
                            
                            # Accumulate total distance traveled in zone
                            if person.prev_distance is not None:
                                distance_delta = abs(person.prev_distance - distance)
                                person.total_distance += distance_delta
                            
                            person.prev_distance = distance
                            
                            if self.frame_idx % FRAME_SKIP == 0:
                                speed = estimate_speed(track_id, self.frame_idx, distance, self.track_history)
                                if speed is not None:
                                    person.last_speed = speed
                            
                            # Add distance to label
                            label_parts.append(f"{distance:.2f}m")
                            
                            # Add speed to label if available
                            if person.last_speed is not None:
                                label_parts.append(f"{abs(person.last_speed):.2f}m/s")
                
                else:
                    # Draw the bottom center point in green
                    annotated_frame = self.tracker.draw_bbox_bottom_center(
                        annotated_frame, bbox, color=(0, 255, 0), radius=6
                    )
                    
                    if track_id is not None:
                        self.current_in_zone[track_id] = False
                
                # Draw label
                label = " | ".join(label_parts)
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + text_size[0] + 5, y1), box_color, -1)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1 + 2, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )
        
        # Check for persons who left the zone
        for track_id in list(self.persons_in_zone.keys()):
            if track_id not in detected_ids and self.current_in_zone.get(track_id, False):
                # Person left the zone
                person = self.persons_in_zone[track_id]
                person.mark_exit(current_time)
                
                duration = person.get_duration()
                self.current_in_zone[track_id] = False
                
                alerts_triggered.append({
                    'type': 'EXIT',
                    'track_id': track_id,
                    'timestamp': current_time,
                    'duration': duration,
                    'distance': person.total_distance,
                    'message': f"⚠ Person (ID: {track_id}) left danger zone (Duration: {duration:.2f}s, Distance: {person.total_distance:.2f}m)"
                })
                
                # Add to history
                self.alert_history.append({
                    'track_id': track_id,
                    'entry_time': person.entry_time,
                    'exit_time': person.exit_time,
                    'duration': duration,
                    'entry_datetime': datetime.fromtimestamp(person.entry_time),
                    'exit_datetime': datetime.fromtimestamp(person.exit_time),
                    'max_speed': person.last_speed,
                    'last_distance': person.last_distance,
                    'total_distance': person.total_distance
                })
        
        # Draw alert text if anyone is in zone
        persons_in_zone_count = sum(1 for v in self.current_in_zone.values() if v)
        if persons_in_zone_count > 0:
            annotated_frame = self._draw_alert_text(annotated_frame, persons_in_zone_count)
        
        # Increment frame counter
        self.frame_idx += 1
        
        return annotated_frame, alerts_triggered
    
    def _draw_alert_text(self, frame, count):
        """
        Draw alert text on frame showing persons in zone.
        
        Args:
            frame (np.ndarray): Video frame
            count (int): Number of persons in zone
        
        Returns:
            np.ndarray: Frame with alert text
        """
        result = frame.copy()
        
        # Draw red alert background
        cv2.rectangle(result, (10, 10), (400, 70), (0, 0, 255), -1)
        cv2.rectangle(result, (10, 10), (400, 70), (0, 0, 255), 2)
        
        # Draw alert text
        text = f"🚨 DANGER ZONE! {count} person(s) in zone"
        cv2.putText(result, text, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        return result
    
    def log_alert(self, alert):
        """
        Log an alert to console with timestamp.
        
        Args:
            alert (dict): Alert dictionary with message and metadata
        """
        timestamp = datetime.fromtimestamp(alert['timestamp']).strftime("%H:%M:%S")
        
        if alert['type'] == 'ENTRY':
            print(f"[{timestamp}] {alert['message']}")
        elif alert['type'] == 'EXIT':
            duration = alert.get('duration', 0)
            print(f"[{timestamp}] {alert['message']}")
            print(f"        └─ Time in zone: {duration:.2f} seconds")
    
    def get_statistics(self):
        """
        Get statistics about zone violations.
        
        Returns:
            dict: Statistics including total violations, average duration, etc.
        """
        total_violations = len(self.alert_history)
        total_persons = len(set(record['track_id'] for record in self.alert_history)) if self.alert_history else 0
        durations = [record['duration'] for record in self.alert_history]
        average_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            'total_violations': total_violations,
            'total_persons': total_persons,
            'average_duration': average_duration,
            'max_duration': max(durations) if durations else 0,
            'min_duration': min(durations) if durations else 0,
            'violations': self.alert_history
        }
    
    def finalize_zone_exits(self, current_time=None):
        """
        Mark all persons still in zone as exited (call at video end).
        
        Args:
            current_time (float): Current timestamp (uses time.time() if None)
        """
        if current_time is None:
            current_time = time.time()
        
        for track_id in list(self.persons_in_zone.keys()):
            person = self.persons_in_zone[track_id]
            if person.exit_time is None:
                person.mark_exit(current_time)
                duration = person.get_duration()
                
                # Add to history
                self.alert_history.append({
                    'track_id': track_id,
                    'entry_time': person.entry_time,
                    'exit_time': person.exit_time,
                    'duration': duration,
                    'entry_datetime': datetime.fromtimestamp(person.entry_time),
                    'exit_datetime': datetime.fromtimestamp(person.exit_time),
                    'max_speed': person.last_speed,
                    'total_distance': person.total_distance
                })
                
                print(f"[{datetime.fromtimestamp(current_time).strftime('%H:%M:%S')}] ⚠ Person (ID: {track_id}) still in danger zone (Duration: {duration:.2f}s, Distance: {person.total_distance:.2f}m) - Video ended")
    
    def print_statistics(self):
        """Print detailed violation information to console"""
        
        if not self.alert_history:
            print("\n" + "=" * 80)
            print("NO DANGER ZONE VIOLATIONS DETECTED")
            print("=" * 80 + "\n")
            return
        
        print("\n" + "=" * 80)
        print("DANGER ZONE VIOLATION DETAILS")
        print("=" * 80)
        
        for i, violation in enumerate(self.alert_history, 1):
            entry_dt = violation['entry_datetime']
            exit_dt = violation['exit_datetime']
            duration = violation['duration']
            max_speed = violation.get('max_speed', None)
            total_distance = violation.get('total_distance', 0)
            
            # Format with full timestamp including microseconds for precision
            entry_time_str = entry_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # milliseconds
            exit_time_str = exit_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]    # milliseconds
            
            print(f"\nViolation #{i}")
            print("-" * 80)
            print(f"Person ID:         {violation['track_id']}")
            print(f"Entry Time:        {entry_time_str}")
            print(f"Exit Time:         {exit_time_str}")
            print(f"Duration:          {duration:.2f} seconds ({int(duration)} sec {int((duration % 1) * 1000)} ms)")
            print(f"Distance Traveled: {total_distance:.2f} meters")
            
            # Print speed information if available
            if max_speed is not None:
                print(f"Max Speed:         {abs(max_speed):.2f} m/s ({abs(max_speed) * 3.6:.2f} km/h)")
        
        print("\n" + "=" * 80)
        print(f"Total Violations: {len(self.alert_history)}")
        print("=" * 80 + "\n")
    
    def reset(self):
        """Reset all tracking data"""
        self.persons_in_zone = {}
        self.current_in_zone = {}
        self.alert_history = []
        self.track_history = defaultdict(list)
        self.frame_idx = 0
