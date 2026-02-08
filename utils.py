"""
Utilities Module
Helper functions for the danger zone alert system.
"""

import cv2
from datetime import datetime


class VideoWriter:
    """Handle video file writing"""
    
    def __init__(self, output_path, fps, frame_width, frame_height, codec='mp4v'):
        """
        Initialize video writer.
        
        Args:
            output_path (str): Path to save the video
            fps (float): Frames per second
            frame_width (int): Frame width
            frame_height (int): Frame height
            codec (str): Video codec
        """
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        self.frame_count = 0
    
    def write(self, frame):
        """
        Write a frame to the video.
        
        Args:
            frame (np.ndarray): Video frame
        
        Returns:
            bool: True if successful
        """
        success = self.writer.write(frame)
        if success:
            self.frame_count += 1
        return success
    
    def release(self):
        """Release the video writer"""
        self.writer.release()
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.release()
        except:
            pass


class Logger:
    """Simple logging utility"""
    
    def __init__(self, log_file=None):
        """
        Initialize logger.
        
        Args:
            log_file (str): Path to log file (optional)
        """
        self.log_file = log_file
        self.logs = []
    
    def log(self, message, level="INFO"):
        """
        Log a message.
        
        Args:
            message (str): Message to log
            level (str): Log level (INFO, WARNING, ERROR, ALERT)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"
        
        print(log_message)
        self.logs.append(log_message)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(log_message + "\n")
    
    def info(self, message):
        """Log info message"""
        self.log(message, "INFO")
    
    def warning(self, message):
        """Log warning message"""
        self.log(message, "WARNING")
    
    def error(self, message):
        """Log error message"""
        self.log(message, "ERROR")
    
    def alert(self, message):
        """Log alert message"""
        self.log(message, "ALERT")
    
    def save_to_file(self, filepath):
        """
        Save all logs to a file.
        
        Args:
            filepath (str): Path to save logs
        """
        with open(filepath, 'w') as f:
            for log_message in self.logs:
                f.write(log_message + "\n")


def get_frame_info(cap):
    """
    Get video frame information.
    
    Args:
        cap (cv2.VideoCapture): Video capture object
    
    Returns:
        dict: Frame information (fps, width, height, total_frames)
    """
    return {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }


def format_time(seconds):
    """
    Format seconds to readable time format.
    
    Args:
        seconds (float): Seconds
    
    Returns:
        str: Formatted time (HH:MM:SS)
    """
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
