"""
Danger Zone Alert System
Main entry point for the danger zone detection and alert system.
"""

import cv2
import sys
from ultralytics import YOLO
from quadrilateral_tracker import QuadrilateralTracker
from zone_alert_manager import ZoneAlertManager
from utils import Logger, VideoWriter, get_frame_info
import config


def main():
    """Main execution function"""
    
    # Initialize logger
    logger = Logger()
    logger.info("Starting Danger Zone Alert System...")
    
    # Load YOLO model
    logger.info(f"Loading YOLO model: {config.YOLO_MODEL}")
    try:
        model = YOLO(config.YOLO_MODEL)
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {str(e)}")
        return
    
    # Initialize video capture
    logger.info(f"Opening video: {config.VIDEO_PATH}")
    tracker = QuadrilateralTracker(config.VIDEO_PATH)
    
    # Get first frame and let user draw quadrilateral
    first_frame = tracker.get_first_frame()
    if first_frame is None:
        logger.error("Could not read the first frame")
        return
    
    logger.info(f"Video resolution: {first_frame.shape[1]}x{first_frame.shape[0]}")
    
    # Interactive quadrilateral drawing
    if not tracker.draw_quadrilateral():
        logger.warning("User cancelled quadrilateral drawing")
        tracker.release()
        return
    
    # Show preview
    logger.info("Displaying first frame with quadrilateral mask...")
    result_image = tracker.apply_quadrilateral_mask(first_frame, alpha=config.MASK_ALPHA,
                                                     mask_color=config.MASK_COLOR,
                                                     border_color=config.BORDER_COLOR)
    cv2.imshow("First Frame Preview", result_image)
    cv2.waitKey(2000)
    cv2.destroyWindow("First Frame Preview")
    
    # Initialize zone alert manager
    alert_manager = ZoneAlertManager(tracker)
    
    # Get video information
    video_info = get_frame_info(tracker.cap)
    logger.info(f"Video info - FPS: {video_info['fps']}, Total frames: {video_info['total_frames']}")
    
    # Restart video and initialize video writer if output path is set
    tracker.restart_video()
    video_writer = None
    if config.OUTPUT_PATH:
        logger.info(f"Output video will be saved to: {config.OUTPUT_PATH}")
        video_writer = VideoWriter(
            config.OUTPUT_PATH,
            video_info['fps'],
            video_info['width'],
            video_info['height']
        )
    
    # Main processing loop
    logger.info("Starting video processing...")
    logger.info("Press 'Q' to stop processing\n")
    
    frame_count = 0
    try:
        while tracker.cap.isOpened():
            success, frame = tracker.cap.read()
            
            if not success:
                break
            
            frame_count += 1
            
            # Run YOLO tracking
            results = model.track(
                frame,
                persist=config.YOLO_PERSIST,
                classes=config.YOLO_CLASS,
                conf=config.YOLO_CONFIDENCE,
                verbose=False
            )
            
            # Visualize detections
            annotated_frame = results[0].plot()
            
            # Apply quadrilateral mask
            annotated_frame = tracker.apply_quadrilateral_mask(
                annotated_frame,
                alpha=config.MASK_ALPHA,
                mask_color=config.MASK_COLOR,
                border_color=config.BORDER_COLOR
            )
            
            # Update zone alerts
            annotated_frame, alerts = alert_manager.update(results[0], annotated_frame)
            
            # Log alerts
            for alert in alerts:
                alert_manager.log_alert(alert)
            
            # Write to output video if enabled
            if video_writer:
                video_writer.write(annotated_frame)
            
            # Display frame
            cv2.imshow(config.DISPLAY_WINDOW_NAME, annotated_frame)
            
            # Log progress every 30 frames
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count} frames...")
            
            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User stopped processing")
                break
    
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
    
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        
        # Finalize any persons still in zone at video end
        alert_manager.finalize_zone_exits()
        
        tracker.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        alert_manager.print_statistics()
        logger.info(f"Total frames processed: {frame_count}")
        logger.info("Danger Zone Alert System stopped")


if __name__ == "__main__":
    main()