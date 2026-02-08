"""
Quadrilateral Tracker Module
Handles interactive quadrilateral drawing and masking for video ROI definition.
"""

import cv2
import numpy as np


class QuadrilateralTracker:
    """
    Track video with an interactive quadrilateral region of interest.
    
    Provides functionality to:
    - Extract the first frame from a video
    - Interactively draw a quadrilateral on the frame
    - Apply a semi-transparent mask overlay to frames
    """
    
    def __init__(self, video_path):
        """
        Initialize the quadrilateral tracker.
        
        Args:
            video_path (str): Path to the video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.first_frame = None
        self.quadrilateral = []
        
    def get_first_frame(self):
        """
        Extract the first frame from the video.
        
        Returns:
            np.ndarray: The first frame of the video, or None if failed
        """
        ret, frame = self.cap.read()
        if ret:
            self.first_frame = frame.copy()
            return frame
        return None
    
    def mouse_callback(self, event, x, y, flags, param):
        """
        Handle mouse events for interactive quadrilateral drawing.
        
        Left-click: Add a point
        Right-click: Remove the last point
        
        Args:
            event: OpenCV mouse event type
            x, y: Mouse coordinates
            flags: Additional flags
            param: Additional parameters
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.quadrilateral) < 4:
                self.quadrilateral.append((x, y))
                print(f"Point {len(self.quadrilateral)} added: ({x}, {y})")
                self._update_display()
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.quadrilateral:
                self.quadrilateral.pop()
                print(f"Last point removed. Remaining points: {len(self.quadrilateral)}")
                self._update_display()
    
    def _update_display(self):
        """Update the display with current quadrilateral points"""
        frame_copy = self.first_frame.copy()
        
        # Draw all points
        for i, point in enumerate(self.quadrilateral):
            cv2.circle(frame_copy, point, 5, (0, 255, 0), -1)
            cv2.putText(frame_copy, str(i + 1), (point[0] + 10, point[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw lines between consecutive points
        for i in range(len(self.quadrilateral) - 1):
            cv2.line(frame_copy, self.quadrilateral[i], 
                    self.quadrilateral[i + 1], (0, 255, 0), 2)
        
        # Close the quadrilateral if all 4 points are set
        if len(self.quadrilateral) == 4:
            cv2.line(frame_copy, self.quadrilateral[3], 
                    self.quadrilateral[0], (0, 255, 0), 2)
        
        cv2.imshow("Draw Quadrilateral - Left Click to Add Points", frame_copy)
    
    def draw_quadrilateral(self):
        """
        Interactive drawing of quadrilateral on first frame.
        
        Allows user to:
        - Left-click to add points
        - Right-click to remove the last point
        - Press SPACE to confirm when all 4 points are added
        - Press ESC to cancel
        
        Returns:
            bool: True if quadrilateral was successfully drawn, False if cancelled
        """
        if self.first_frame is None:
            print("Error: First frame not extracted")
            return False
        
        cv2.imshow("Draw Quadrilateral - Left Click to Add Points", self.first_frame)
        cv2.setMouseCallback("Draw Quadrilateral - Left Click to Add Points", self.mouse_callback)
        
        self._print_instructions()
        
        confirmed = False
        while not confirmed:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("Drawing cancelled")
                self.quadrilateral = []
                cv2.destroyWindow("Draw Quadrilateral - Left Click to Add Points")
                return False
            elif key == 32:  # SPACE - confirm when 4 points are added
                if len(self.quadrilateral) == 4:
                    confirmed = True
                    break
                else:
                    remaining = 4 - len(self.quadrilateral)
                    print(f"⚠ Please add {remaining} more point(s) before pressing SPACE")
        
        print(f"✓ Quadrilateral points confirmed: {self.quadrilateral}\n")
        cv2.destroyWindow("Draw Quadrilateral - Left Click to Add Points")
        return True
    
    @staticmethod
    def _print_instructions():
        """Print user instructions for quadrilateral drawing"""
        print("\n" + "=" * 60)
        print("DRAW QUADRILATERAL REGION OF INTEREST")
        print("=" * 60)
        print("Instructions:")
        print("  - Left-click to add points (draw 4 points for a quadrilateral)")
        print("  - Right-click to remove the last point")
        print("  - Press SPACE to confirm when all 4 points are added")
        print("  - Press ESC to cancel")
        print("=" * 60 + "\n")
    
    def apply_quadrilateral_mask(self, inference_image, alpha=0.3, 
                                  mask_color=(200, 200, 100), border_color=(0, 255, 255)):
        """
        Apply the quadrilateral as a light color mask on the inference image.
        
        Args:
            inference_image (np.ndarray): The image to apply the mask on
            alpha (float): Transparency of the mask (0.0 to 1.0). Default: 0.3
            mask_color (tuple): BGR color for the mask overlay. Default: light cyan
            border_color (tuple): BGR color for the border. Default: yellow
        
        Returns:
            np.ndarray: Image with quadrilateral mask applied
        """
        if len(self.quadrilateral) != 4:
            return inference_image
        
        result = inference_image.copy()
        
        # Create a mask with the quadrilateral
        mask = np.zeros(inference_image.shape[:2], dtype=np.uint8)
        quad_points = np.array(self.quadrilateral, dtype=np.int32)
        cv2.fillPoly(mask, [quad_points], 255)
        
        # Apply colored mask
        overlay = result.copy()
        overlay[mask == 255] = mask_color
        
        # Blend with original image
        result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)
        
        # Draw quadrilateral border
        cv2.polylines(result, [quad_points], True, border_color, 2)
        
        return result
    
    def get_quadrilateral_points(self):
        """
        Get the current quadrilateral points.
        
        Returns:
            list: List of (x, y) tuples representing the quadrilateral vertices
        """
        return self.quadrilateral
    
    def set_quadrilateral_points(self, points):
        """
        Manually set quadrilateral points.
        
        Args:
            points (list): List of (x, y) tuples representing the quadrilateral vertices
        """
        if len(points) != 4:
            print("Warning: Quadrilateral should have exactly 4 points")
        self.quadrilateral = points
    
    def reset_quadrilateral(self):
        """Reset the quadrilateral points"""
        self.quadrilateral = []
    
    def is_point_in_quadrilateral(self, point):
        """
        Check if a point is inside the quadrilateral.
        
        Args:
            point (tuple): (x, y) coordinates
        
        Returns:
            bool: True if point is inside, False otherwise
        """
        if len(self.quadrilateral) != 4:
            return False
        
        quad_points = np.array(self.quadrilateral, dtype=np.int32)
        result = cv2.pointPolygonTest(quad_points, point, False)
        return result >= 0
    
    def get_bbox_bottom_center(self, bbox):
        """
        Get the center point of the bottom line of a bounding box.
        
        Args:
            bbox (list/tuple): Bounding box coordinates [x1, y1, x2, y2]
        
        Returns:
            tuple: (center_x, center_y) of the bottom line
        """
        x1, y1, x2, y2 = bbox[:4]
        center_x = (x1 + x2) / 2
        center_y = y2  # Bottom of the bbox
        return (int(center_x), int(center_y))
    
    def is_bbox_in_zone(self, bbox):
        """
        Check if a bounding box's bottom center is inside the quadrilateral zone.
        
        Uses the center of the bottom line of the bbox as the decision point.
        
        Args:
            bbox (list/tuple): Bounding box coordinates [x1, y1, x2, y2]
        
        Returns:
            bool: True if bbox bottom center is inside the zone, False otherwise
        """
        if len(self.quadrilateral) != 4:
            return False
        
        bottom_center = self.get_bbox_bottom_center(bbox)
        return self.is_point_in_quadrilateral(bottom_center)
    
    def draw_bbox_bottom_center(self, image, bbox, color=(0, 255, 0), radius=5):
        """
        Draw the bottom center point of a bbox on the image.
        
        Args:
            image (np.ndarray): Image to draw on
            bbox (list/tuple): Bounding box coordinates [x1, y1, x2, y2]
            color (tuple): BGR color for the point. Default: green
            radius (int): Radius of the circle. Default: 5
        
        Returns:
            np.ndarray: Image with bottom center point drawn
        """
        bottom_center = self.get_bbox_bottom_center(bbox)
        result = image.copy()
        cv2.circle(result, bottom_center, radius, color, -1)
        return result
    
    def restart_video(self):
        """Restart video capture from the beginning"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def release(self):
        """Release video capture and close all windows"""
        self.cap.release()
        cv2.destroyAllWindows()
