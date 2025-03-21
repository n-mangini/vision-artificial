import cv2

def create_trackbar(trackbar_name, window_name, slider_max):
    value = 0
    count = slider_max
    cv2.createTrackbar(trackbar_name, window_name,
                       value, count, lambda x: None)

def get_trackbar_pos(trackbar_name, window_name):
    return int(cv2.getTrackbarPos(
            trackbar_name, window_name))