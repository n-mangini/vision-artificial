import cv2
import config
from frame import handle_video_capture, check_frame_exit, apply_color_convertion, threshold_frame
from trackbar import create_trackbar, get_trackbar_pos


def main():
    main_window_name = 'Form Detection'
    cv2.namedWindow(main_window_name)
    
    capture = handle_video_capture(main_window_name, config.VIDEO_PATH)

    trackbar_thresh_name = 'Threshold'
    thresh_slider_max = 255
    create_trackbar(trackbar_thresh_name, main_window_name, thresh_slider_max)

    while True:
        success, frame = capture.read()
        
        monochromatic_frame = apply_color_convertion(
            frame=frame, color=cv2.COLOR_RGB2GRAY)

        trackbar_thresh_value = get_trackbar_pos(
            trackbar_thresh_name, main_window_name)

        binary = threshold_frame(frame=monochromatic_frame, slider_max=thresh_slider_max,
                                       binary=cv2.THRESH_BINARY,
                                       trackbar_value=trackbar_thresh_value)

        cv2.imshow(main_window_name, monochromatic_frame)
        cv2.imshow('Treshold', binary)
        if check_frame_exit():
            break

    capture.release()


main()
