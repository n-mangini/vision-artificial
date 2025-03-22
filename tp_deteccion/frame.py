import cv2


def handle_video_capture(window_name, path):
    cv2.namedWindow(window_name)
    capture = cv2.VideoCapture(path)

    if not capture.isOpened():
        print("Error: Could not open video stream.")
        return None

    return capture


def video_capture_read(capture):
    success, frame = capture.read()
    if not success:
        print("Error: Could not read frame.")
        return None
    return frame


def check_frame_exit():
    return cv2.waitKey(1) & 0xFF == ord('c')


def apply_color_convertion(frame, color):
    return cv2.cvtColor(frame, color)


def get_threshold_frame(frame, slider_max, binary, trackbar_value):
    _, th = cv2.threshold(frame, trackbar_value, slider_max, binary)
    return th


"""
    Applies morphological operations to denoise (remove noise) an image by removing small noise and filling small holes.
    - Opening: removes small noise by applying erosion first, then restores the main structure with dilation.
    - Closing: fills small holes by applying dilation first, then smooths the result with erosion.

    @param frame: The input frame (grayscale or binary).
    @param kernel_size: The size of the structuring element (square kernel).
    
    @return: A denoised image where small noise is removed and small holes are filled.

"""


def get_denoised_frame(frame, method, kernel_size):
    kernel = cv2.getStructuringElement(
        method, (kernel_size, kernel_size))
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing
