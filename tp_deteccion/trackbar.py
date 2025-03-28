trackbar_values = {
    'thresh': 60,
    'denoise': 1,
    'min_area': 500,
    'max_area': 99999,
    'min_shape_score': 0,
    'show_all_contours': 0,
    'show_filtered_contours': 1,
}


def on_toggle_change(name):
    def callback(value):
        print(f"{name}: ", value)
        trackbar_values[name] = value
    return callback


def on_min_shape_score_change(value):
    # tolerances are float values but trackbar values are integers
    float_value = value / 100.0
    print("min shape score: ", float_value)
    trackbar_values['min_shape_score'] = float_value


def on_denoise_change(value):
    # Avoid 0 kernel size
    print("denoise: ", value)
    if value == 0:
        trackbar_values['denoise'] = 1
        print("denoise: 1")
    else:
        trackbar_values['denoise'] = value
        print("denoise: ", value)
