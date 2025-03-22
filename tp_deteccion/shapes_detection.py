import cv2
import config
from config import choose_camera_by_OS
from frame import handle_video_capture, video_capture_read, check_frame_exit, apply_color_convertion, get_threshold_frame, get_denoised_frame
from trackbar import create_trackbar, get_trackbar_pos


def main():
    main_frame_name = 'Form Detection'
    processed_frame_name = 'Processed Detection'

    capture = handle_video_capture(main_frame_name, choose_camera_by_OS())

    # Threshold trackbar
    trackbar_thresh_name = 'Threshold'
    thresh_slider_max = 255
    create_trackbar(trackbar_thresh_name, main_frame_name,
                    slider_default=60, slider_max=thresh_slider_max)

    # Threshold trackbar
    trackbar_kernel_name = 'Kernel denoise'
    contour_kernel_max = 10
    create_trackbar(trackbar_kernel_name, main_frame_name,
                    slider_default=1, slider_max=contour_kernel_max)

    while True:
        main_frame = video_capture_read(capture)

        # Apply monocromatic
        binary_frame = apply_color_convertion(
            frame=main_frame, color=cv2.COLOR_RGB2GRAY)

        # Apply threshold
        trackbar_thresh_value = get_trackbar_pos(
            trackbar_thresh_name, main_frame_name)

        threshold_frame = get_threshold_frame(frame=binary_frame, slider_max=thresh_slider_max,
                                              binary=cv2.THRESH_BINARY,
                                              trackbar_value=trackbar_thresh_value)

        # Apply noise reduction
        trackbar_denoise_value = get_trackbar_pos(
            trackbar_kernel_name, main_frame_name)
        
        denoised_frame = get_denoised_frame(frame=threshold_frame, method=cv2.MORPH_ELLIPSE, kernel_size=trackbar_denoise_value)

        cv2.imshow(main_frame_name, main_frame)
        cv2.imshow(processed_frame_name, denoised_frame)
        if check_frame_exit():
            break

    capture.release()


main()
