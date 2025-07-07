"""Image processing utilities for Streamlit UI."""

import base64
import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict


class ImageProcessor:
    @staticmethod
    def decode_base64_image(img_b64: str) -> Optional[np.ndarray]:
        """Decode base64 image to numpy array."""
        if not img_b64 or not isinstance(img_b64, str):
            print(f"Invalid image data: expected string, got {type(img_b64)}")
            return None
        
        try:
            img_data = base64.b64decode(img_b64)
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                print("Failed to decode image from buffer")
                return None
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error decoding image: {e}")
            return None

    @staticmethod
    def draw_detections(img: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detection annotations on image."""
        img_copy = img.copy()
        
        for det in detections:
            if 'center_px' in det:
                x, y = map(int, det['center_px'])
                y = img.shape[0] - y  # Flip Y coordinate
                
                # Draw circle
                cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)
                
                # Draw label
                label = f"{det.get('class', '')} ({det.get('confidence', 0):.2f})"
                cv2.putText(img_copy, label, (x + 10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img_copy

    @staticmethod
    def process_detection_data(data: Dict) -> Tuple[Optional[np.ndarray], str]:
        """Process detection data and return image with status."""
        detections = data.get("detections", [])
        is_raw_image = data.get("raw_image", False)
        
        # Get the image data (always in "image" field)
        img_b64 = data.get("image")
        if not img_b64:
            return None, "No image data available"
        
        # Decode the image
        img = ImageProcessor.decode_base64_image(img_b64)
        if img is None:
            return None, "Failed to decode image"
        
        # If it's a raw image, return it as-is
        if is_raw_image:
            return img, "Raw camera image"
        
        # If we have detections, show image with annotations
        if detections:
            img_with_detections = ImageProcessor.draw_detections(img, detections)
            return img_with_detections, f"Detections: {len(detections)}"
        
        # If processed image but no detections, return plain image
        return img, "No detections found"
