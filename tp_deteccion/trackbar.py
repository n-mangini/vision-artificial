import cv2


def on_thresh_change(value):
    global trackbar_thresh_value
    trackbar_thresh_value = value


def on_denoise_change(value):
    global trackbar_denoise_value
    # Avoid 0 kernel size
    if value == 0:
        trackbar_denoise_value = 1
    else:
        trackbar_denoise_value = value
