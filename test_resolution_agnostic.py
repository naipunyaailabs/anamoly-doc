#!/usr/bin/env python3
"""
Test script to verify resolution-agnostic processing capability
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

import src.core.logic_engine as logic

def test_resolution_agnostic():
    """Test the resolution-agnostic processing with different frame sizes"""
    
    # Test different resolutions
    test_resolutions = [
        (480, 640),   # Standard VGA
        (720, 1280),  # HD
        (1080, 1920), # Full HD
        (448, 832),   # Previously problematic resolution
        (360, 640),   # Mobile resolution
        (240, 320),   # Low resolution
        (1440, 2560), # 2K
        (2160, 3840), # 4K
    ]
    
    print("Testing resolution-agnostic processing...")
    
    for i, (h, w) in enumerate(test_resolutions):
        print(f"\nTest {i+1}: {w}x{h}")
        
        # Create a test frame with some simple pattern
        frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        
        # Add some distinguishable features
        # Add a rectangle in the center
        center_x, center_y = w // 2, h // 2
        cv2.rectangle(frame, (center_x-50, center_y-50), (center_x+50, center_y+50), (255, 0, 0), 2)
        
        # Test the resize function
        try:
            resized_frame, scale_ratio, padding = logic.resize_frame_with_aspect_ratio(frame)
            
            print(f"  Original size: {w}x{h}")
            print(f"  Resized size: {resized_frame.shape[1]}x{resized_frame.shape[0]}")
            print(f"  Scale ratio: {scale_ratio:.4f}")
            print(f"  Padding: {padding}")
            
            # Verify that the resized frame is 640x640
            assert resized_frame.shape[0] == 640, f"Height is {resized_frame.shape[0]}, expected 640"
            assert resized_frame.shape[1] == 640, f"Width is {resized_frame.shape[1]}, expected 640"
            
            # Test coordinate scaling back
            # Create some test boxes in the resized coordinate system
            test_boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
            scaled_boxes = logic.scale_boxes_to_original(test_boxes, scale_ratio, padding, (h, w))
            
            print(f"  Test box in resized coords: {test_boxes[0]}")
            print(f"  Scaled back to original coords: {scaled_boxes[0]}")
            
            # Verify that scaled boxes are within original frame bounds
            assert scaled_boxes[0][0] >= 0 and scaled_boxes[0][0] <= w, "X1 out of bounds"
            assert scaled_boxes[0][1] >= 0 and scaled_boxes[0][1] <= h, "Y1 out of bounds"
            assert scaled_boxes[0][2] >= 0 and scaled_boxes[0][2] <= w, "X2 out of bounds"
            assert scaled_boxes[0][3] >= 0 and scaled_boxes[0][3] <= h, "Y2 out of bounds"
            
            print(f"  ✓ Test passed")
            
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            return False
    
    print("\nAll resolution tests passed!")
    return True

if __name__ == "__main__":
    test_resolution_agnostic()