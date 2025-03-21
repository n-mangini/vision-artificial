import cv2

def handle_video_capture(window_name, path):
    cv2.namedWindow(window_name)
    capture = cv2.VideoCapture(path)

    if not capture.isOpened():
        print("Error: Could not open video stream.")
        return None

    return capture


def check_frame_exit():
    return cv2.waitKey(1) & 0xFF == ord('c')


def apply_color_convertion(frame, color):
    return cv2.cvtColor(frame, color)



def threshold_frame(frame, slider_max, binary, trackbar_value):
    _, th = cv2.threshold(frame, trackbar_value, slider_max, binary)
    return th