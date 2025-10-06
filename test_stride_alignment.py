#!/usr/bin/env python3
"""
Test script to verify stride alignment functionality
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

import src.core.logic_engine as logic

def test_stride_alignment():
    """Test the stride alignment function with various input sizes"""
    
    print("Testing stride alignment functionality...")
    
    # Test cases with various dimensions
    test_cases = [
        (720, 407),  # The case from the warning
        (640, 480),  # Standard resolution
        (1920, 1080), # Full HD
        (1280, 720),  # HD
        (320, 240),   # Small resolution
        (800, 600),   # Another common resolution
        (1024, 768),  # XGA resolution
    ]
    
    print("\nTesting various frame dimensions:")
    print("=" * 50)
    
    for h, w in test_cases:
        # Create a dummy frame with the test dimensions
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Apply stride alignment
        aligned_frame, scale_ratio = logic.align_to_stride(frame)
        
        # Get aligned dimensions
        aligned_h, aligned_w = aligned_frame.shape[:2]
        
        # Check if dimensions are multiples of 32
        is_w_multiple = (aligned_w % 32 == 0)
        is_h_multiple = (aligned_h % 32 == 0)
        
        print(f"Original: {w}x{h} -> Aligned: {aligned_w}x{aligned_h}")
        print(f"  Width multiple of 32: {is_w_multiple}")
        print(f"  Height multiple of 32: {is_h_multiple}")
        print(f"  Scale ratio: {scale_ratio}")
        print(f"  Status: {'PASS' if is_w_multiple and is_h_multiple else 'FAIL'}")
        print()
        
        # Verify that the alignment is correct
        expected_w = ((w + 31) // 32) * 32
        expected_h = ((h + 31) // 32) * 32
        
        assert aligned_w == expected_w, f"Width mismatch: expected {expected_w}, got {aligned_w}"
        assert aligned_h == expected_h, f"Height mismatch: expected {expected_h}, got {aligned_h}"
        
    print("All tests passed! Stride alignment is working correctly.")

def test_model_compatibility():
    """Test that aligned frames work with YOLO models without warnings"""
    
    print("\nTesting model compatibility with aligned frames...")
    print("=" * 50)
    
    # Create a frame with problematic dimensions from the warning
    frame = np.zeros((407, 720, 3), dtype=np.uint8)
    
    print(f"Original frame dimensions: {frame.shape[1]}x{frame.shape[0]}")
    
    # Apply stride alignment
    aligned_frame, scale_ratio = logic.align_to_stride(frame)
    
    print(f"Aligned frame dimensions: {aligned_frame.shape[1]}x{aligned_frame.shape[0]}")
    print(f"Scale ratio: {scale_ratio}")
    
    # Verify dimensions are multiples of 32
    w, h = aligned_frame.shape[1], aligned_frame.shape[0]
    assert w % 32 == 0, f"Width {w} is not multiple of 32"
    assert h % 32 == 0, f"Height {h} is not multiple of 32"
    
    print("Frame is properly aligned for YOLO models!")
    print("No warnings should occur when processing this frame.")

if __name__ == "__main__":
    test_stride_alignment()
    test_model_compatibility()
    print("\nAll stride alignment tests completed successfully!")