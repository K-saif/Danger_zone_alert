"""
Zone Alert Manager Module
Manages detection, alerting, and tracking of persons in danger zones.
"""

import cv2
import time
from datetime import datetime


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
                
                # Check if in zone
                is_in_zone = self.tracker.is_bbox_in_zone(bbox)
                
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
                
                else:
                    # Draw the bottom center point in green
                    annotated_frame = self.tracker.draw_bbox_bottom_center(
                        annotated_frame, bbox, color=(0, 255, 0), radius=6
                    )
                    
                    if track_id is not None:
                        self.current_in_zone[track_id] = False
        
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
                    'message': f"⚠ Person (ID: {track_id}) left danger zone (Duration: {duration:.2f}s)"
                })
                
                # Add to history
                self.alert_history.append({
                    'track_id': track_id,
                    'entry_time': person.entry_time,
                    'exit_time': person.exit_time,
                    'duration': duration,
                    'entry_datetime': datetime.fromtimestamp(person.entry_time),
                    'exit_datetime': datetime.fromtimestamp(person.exit_time)
                })
        
        # Draw alert text if anyone is in zone
        persons_in_zone_count = sum(1 for v in self.current_in_zone.values() if v)
        if persons_in_zone_count > 0:
            annotated_frame = self._draw_alert_text(annotated_frame, persons_in_zone_count)
        
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
                    'exit_datetime': datetime.fromtimestamp(person.exit_time)
                })
                
                print(f"[{datetime.fromtimestamp(current_time).strftime('%H:%M:%S')}] ⚠ Person (ID: {track_id}) still in danger zone (Duration: {duration:.2f}s) - Video ended")
    
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
            
            # Format with full timestamp including microseconds for precision
            entry_time_str = entry_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # milliseconds
            exit_time_str = exit_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]    # milliseconds
            
            print(f"\nViolation #{i}")
            print("-" * 80)
            print(f"Person ID: {violation['track_id']}")
            print(f"Entry Time:    {entry_time_str}")
            print(f"Exit Time:     {exit_time_str}")
            print(f"Duration:      {duration:.2f} seconds ({int(duration)} sec {int((duration % 1) * 1000)} ms)")
        
        print("\n" + "=" * 80)
        print(f"Total Violations: {len(self.alert_history)}")
        print("=" * 80 + "\n")
    
    def reset(self):
        """Reset all tracking data"""
        self.persons_in_zone = {}
        self.current_in_zone = {}
        self.alert_history = []
