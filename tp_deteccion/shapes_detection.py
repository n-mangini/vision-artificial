import cv2
import config
from frame import handle_video_capture, video_capture_read, check_frame_exit, apply_color_convertion, threshold_frame
from trackbar import create_trackbar, get_trackbar_pos


def main():
    main_window_name = 'Form Detection'
    binary_window_name = 'Binary'

    capture = handle_video_capture(main_window_name, config.VIDEO_PATH)

    trackbar_thresh_name = 'Threshold'
    thresh_slider_max = 255
    create_trackbar(trackbar_thresh_name, main_window_name,
                    slider_default=60, slider_max=thresh_slider_max)

    while True:
        main_frame = video_capture_read(capture)

        monochromatic_frame = apply_color_convertion(
            frame=main_frame, color=cv2.COLOR_RGB2GRAY)

        trackbar_thresh_value = get_trackbar_pos(
            trackbar_thresh_name, main_window_name)

        binary_frame = threshold_frame(frame=monochromatic_frame, slider_max=thresh_slider_max,
                                       binary=cv2.THRESH_BINARY,
                                       trackbar_value=trackbar_thresh_value)

        cv2.imshow(main_window_name, main_frame)
        cv2.imshow(binary_window_name, binary_frame)
        if check_frame_exit():
            break

    capture.release()


main()
