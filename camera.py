"""
Camera module for capturing webcam frames.
Handles real-time video capture and frame management.
"""

import cv2
import numpy as np
import os
import hashlib
import logging
from typing import Optional, Tuple
import config

# Configure logging
logger = logging.getLogger(__name__)


class Camera:
    """Handles webcam capture and frame management."""
    
    def __init__(self, camera_index: int = config.CAMERA_INDEX):
        """
        Initialize camera.
        
        Args:
            camera_index: Index of the camera to use (default: 0)
        """
        self.camera_index = camera_index
        self.cap = None
        self.last_frame_hash = None
        self._ensure_temp_dir()
        
    def _ensure_temp_dir(self):
        """Create temporary directory if it doesn't exist."""
        if not os.path.exists(config.TEMP_DIR):
            os.makedirs(config.TEMP_DIR)
            logger.info(f"Created temporary directory: {config.TEMP_DIR}")
    
    def open(self) -> bool:
        """
        Open camera connection.
        
        Returns:
            bool: True if camera opened successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera at index {self.camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            
            logger.info(f"Camera opened successfully at index {self.camera_index}")
            return True
            
        except Exception as e:
            logger.error(f"Error opening camera: {e}")
            return False
    
    def close(self, cleanup_temp=False):
        """Close camera connection and cleanup resources.
        
        Args:
            cleanup_temp: Whether to delete temp files (default: False)
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Camera closed")
        
        cv2.destroyAllWindows()
        
        # Cleanup temp directory only if requested
        if cleanup_temp and config.AUTO_CLEANUP:
            self._cleanup_temp_files()
    
    def _cleanup_temp_files(self):
        """Remove temporary files."""
        try:
            temp_image_path = os.path.join(config.TEMP_DIR, config.TEMP_IMAGE_NAME)
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
                logger.info(f"Cleaned up temporary file: {temp_image_path}")
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {e}")
    
    def _compute_frame_hash(self, frame: np.ndarray) -> str:
        """
        Compute hash of frame to detect duplicates.
        
        Args:
            frame: Input frame
            
        Returns:
            str: MD5 hash of the frame
        """
        return hashlib.md5(frame.tobytes()).hexdigest()
    
    def is_duplicate_frame(self, frame: np.ndarray) -> bool:
        """
        Check if frame is duplicate of the last captured frame.
        
        Args:
            frame: Frame to check
            
        Returns:
            bool: True if frame is duplicate, False otherwise
        """
        current_hash = self._compute_frame_hash(frame)
        
        if self.last_frame_hash is None:
            return False
        
        is_duplicate = current_hash == self.last_frame_hash
        
        if is_duplicate:
            logger.debug("Duplicate frame detected")
        
        return is_duplicate
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from camera.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        if self.cap is None or not self.cap.isOpened():
            logger.error("Camera is not opened")
            return False, None
        
        try:
            ret, frame = self.cap.read()
            
            if not ret:
                logger.warning("Failed to read frame from camera")
                return False, None
            
            return True, frame
            
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return False, None
    
    def capture_frame(self, frame: np.ndarray) -> Optional[str]:
        """
        Capture and save frame to temporary storage.
        
        Args:
            frame: Frame to capture
            
        Returns:
            Optional[str]: Path to saved image, or None if failed
        """
        try:
            # Check for duplicate frame
            if self.is_duplicate_frame(frame):
                logger.warning("Attempted to capture duplicate frame - skipping")
                return None
            
            # Save frame
            temp_image_path = os.path.join(config.TEMP_DIR, config.TEMP_IMAGE_NAME)
            cv2.imwrite(temp_image_path, frame)
            
            # Update last frame hash
            self.last_frame_hash = self._compute_frame_hash(frame)
            
            logger.info(f"Frame captured and saved to: {temp_image_path}")
            return temp_image_path
            
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None
    
    def delete_temp_image(self):
        """Delete temporary image file."""
        try:
            temp_image_path = os.path.join(config.TEMP_DIR, config.TEMP_IMAGE_NAME)
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
                logger.info(f"Deleted temporary image: {temp_image_path}")
        except Exception as e:
            logger.warning(f"Error deleting temporary image: {e}")
    
    def show_frame(self, frame: np.ndarray, window_name: str = "Camera"):
        """
        Display frame in a window.
        
        Args:
            frame: Frame to display
            window_name: Name of the window
        """
        cv2.imshow(window_name, frame)
    
    def wait_for_key(self, delay: int = 1) -> int:
        """
        Wait for key press.
        
        Args:
            delay: Delay in milliseconds
            
        Returns:
            int: Key code of pressed key
        """
        return cv2.waitKey(delay) & 0xFF
    
    def run_capture_loop(self) -> Optional[str]:
        """
        Run interactive capture loop.
        Shows live camera feed and waits for user to capture frame.
        
        Returns:
            Optional[str]: Path to captured image, or None if cancelled
        """
        if not self.open():
            logger.error("Failed to open camera")
            return None
        
        try:
            logger.info(f"Press '{config.CAPTURE_KEY}' to capture, '{config.QUIT_KEY}' to quit")
            
            while True:
                ret, frame = self.read_frame()
                
                if not ret:
                    logger.error("Failed to read frame")
                    break
                
                # Add instructions to frame
                display_frame = frame.copy()
                cv2.putText(
                    display_frame,
                    f"Press '{config.CAPTURE_KEY}' to capture, '{config.QUIT_KEY}' to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                self.show_frame(display_frame, "Facial Recognition - Camera")
                
                key = self.wait_for_key(1)
                
                if key == ord(config.CAPTURE_KEY):
                    image_path = self.capture_frame(frame)
                    if image_path:
                        logger.info("Frame captured successfully")
                        return image_path
                    else:
                        logger.warning("Failed to capture frame (possibly duplicate)")
                
                elif key == ord(config.QUIT_KEY):
                    logger.info("Capture cancelled by user")
                    break
            
            return None
            
        except Exception as e:
            logger.error(f"Error in capture loop: {e}")
            return None
            
        finally:
            self.close()


def main():
    """Test camera module."""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT
    )
    
    camera = Camera()
    image_path = camera.run_capture_loop()
    
    if image_path:
        print(f"Image captured: {image_path}")
    else:
        print("No image captured")


if __name__ == "__main__":
    main()
