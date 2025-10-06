import cv2
import numpy as np

def create_test_video():
    # Create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('videos/test_video.mp4', fourcc, 30.0, (640, 480))
    
    # Create 100 frames of test video
    for i in range(100):
        # Create a black image
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some moving objects
        # Draw a moving circle
        center_x = 100 + i * 2
        center_y = 240
        cv2.circle(frame, (center_x, center_y), 30, (0, 255, 0), -1)
        
        # Draw a moving rectangle
        rect_x = 300 + i
        rect_y = 100
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + 50, rect_y + 50), (255, 0, 0), -1)
        
        # Add text
        cv2.putText(frame, f'Frame {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write the frame
        out.write(frame)
    
    # Release everything
    out.release()
    print("Test video created: videos/test_video.mp4")

if __name__ == "__main__":
    create_test_video()