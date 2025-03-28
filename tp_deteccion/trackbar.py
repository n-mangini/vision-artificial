import cv2

trackbar_values = {
    'thresh': 60,
    'denoise': 1,
    'min_area': 1,
    'max_area': 99999,
    'shape_tolerance': 0
}


def on_thresh_change(value):
    print("threshold: ", value)
    trackbar_values['thresh'] = value


def on_min_area_change(value):
    print("min area: ", value)
    trackbar_values['min_area'] = value


def on_max_area_change(value):
    print("max area: ", value)
    trackbar_values['max_area'] = value


def on_shape_tolerance_change(value):
    # tolerances are float values but trackbar values are integers
    float_value = value / 100.0
    print("shape tolerance: ", float_value)
    trackbar_values['shape_tolerance'] = float_value


def on_denoise_change(value):
    # Avoid 0 kernel size
    print("denoise: ", value)
    if value == 0:
        trackbar_values['denoise'] = 1
        print("denoise: 1")
    else:
        trackbar_values['denoise'] = value
        print("denoise: ", value)
