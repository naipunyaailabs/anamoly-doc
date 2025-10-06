#!/usr/bin/env python3
"""
Script to run the process_video function with a test video
"""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import the process_video function
from src.utils.process_video import process_video

def main():
    # Use the first available video file
    video_path = "videos/demo.mp4"
    
    print(f"Processing video: {video_path}")
    process_video(video_path)

if __name__ == "__main__":
    main()