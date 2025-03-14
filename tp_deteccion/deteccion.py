import cv2
import config

def detect_video(path):
    window_name = "Video"
    cv2.namedWindow(window_name)
    capture = cv2.VideoCapture(path)
    
    if not capture.isOpened():
        print("Error: Could not open video stream.")
        return None
    
    return capture

def check_exit():
    return cv2.waitKey(1) & 0xFF == ord('q')
    
def main():
    capture = detect_video(config.VIDEO_STREAM_PATH)
    while True:
    
        sucess, frame = capture.read()
        cv2.imshow('Video', frame)
        
        if check_exit():
            break

    capture.release()
    
main()