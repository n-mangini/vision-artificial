import cv2

trackbar_values = {
    'thresh': 60,
    'denoise': 1
}

def on_thresh_change(value):
    print("threshold: ", value)
    trackbar_values['thresh'] = value


def on_denoise_change(value):
    # Avoid 0 kernel size
    print("denoise: ", value)
    if value == 0:
        trackbar_values['denoise'] = 1
        print("denoise: 1")
    else:
        trackbar_values['denoise'] = value
        print("denoise: ", value)
