import requests
import time

def test_video_processing():
    url = "http://localhost:8000/process-video/"
    
    # Open the test video file
    with open("videos/test_video.mp4", "rb") as video_file:
        files = {"file": ("test_video.mp4", video_file, "video/mp4")}
        
        print("Sending video for processing...")
        response = requests.post(url, files=files)
        
        if response.status_code == 200:
            print("Video processing started successfully!")
            print("Response:", response.json())
        else:
            print(f"Error: {response.status_code}")
            print("Response:", response.text)

if __name__ == "__main__":
    test_video_processing()