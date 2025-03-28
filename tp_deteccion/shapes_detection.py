import cv2
from config import choose_camera_by_OS
from frame import handle_video_capture, video_capture_read, check_frame_exit, apply_color_convertion, get_threshold_frame, get_denoised_frame
from trackbar import on_denoise_change, trackbar_values, on_min_shape_score_change, on_toggle_change
from contour import filter_contours_by_area, draw_contours, COLOR_GREEN, COLOR_RED, get_saved_contour, get_shape_center

FIGURE_TEMPLATES = {
    "Triangle": get_saved_contour('tp_deteccion/resources/triangle.png'),
    "Square": get_saved_contour('tp_deteccion/resources/square.png'),
    "Star": get_saved_contour('tp_deteccion/resources/star.jpeg'),
    "Circle": get_saved_contour('tp_deteccion/resources/circle.jpg')
}


def main():
    main_frame_name = 'Shape Detection'
    processed_frame_name = 'Processed Detection'

    capture = handle_video_capture(main_frame_name, choose_camera_by_OS())

    # Create trackbars
    thresh_trackbar_name = 'Threshold'
    thresh_slider_max = 255
    cv2.createTrackbar(thresh_trackbar_name, main_frame_name,
                       trackbar_values['thresh'], thresh_slider_max, on_toggle_change('thresh'))

    kernel_trackbar_name = 'Denoise'
    kernel_contour_max = 10
    cv2.createTrackbar(kernel_trackbar_name, main_frame_name,
                       trackbar_values['denoise'], kernel_contour_max, on_denoise_change)

    min_area_trackbar_name = 'Min Area'
    min_area_max = 10000
    cv2.createTrackbar(min_area_trackbar_name, main_frame_name,
                       trackbar_values['min_area'], min_area_max, on_toggle_change('min_area'))

    max_area_trackbar_name = 'Max Area'
    max_area_max = 99999
    cv2.createTrackbar(max_area_trackbar_name, main_frame_name,
                       trackbar_values['max_area'], max_area_max, on_toggle_change('max_area'))

    shapes_tolerance_trackbar_name = 'Shape Tolerance'
    shape_tolerance_max = 100  # = 1.0 when converted to float
    cv2.createTrackbar(shapes_tolerance_trackbar_name, main_frame_name,
                       trackbar_values["min_shape_score"], shape_tolerance_max, on_min_shape_score_change)

    cv2.createTrackbar('Show All Contours', main_frame_name,
                       trackbar_values['show_all_contours'], 1, on_toggle_change('show_all_contours'))
    cv2.createTrackbar('Show Filtered Contours', main_frame_name,
                       trackbar_values['show_filtered_contours'], 1, on_toggle_change('show_filtered_contours'))

    while capture.isOpened():
        main_frame = video_capture_read(capture)

        # Apply monocromatic
        binary_frame = apply_color_convertion(
            frame=main_frame, color=cv2.COLOR_RGB2GRAY)

        # Apply threshold
        threshold_frame = get_threshold_frame(frame=binary_frame, slider_max=thresh_slider_max,
                                              binary=cv2.THRESH_BINARY,
                                              trackbar_value=trackbar_values['thresh'])

        # Apply noise reduction
        denoised_frame = get_denoised_frame(
            frame=threshold_frame, method=cv2.MORPH_ELLIPSE, kernel_size=trackbar_values['denoise'])

        # Contours
        contours, hierarchy = cv2.findContours(
            image=denoised_frame, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        filtered_contours = filter_contours_by_area(
            contours, min_area=trackbar_values['min_area'], max_area=trackbar_values['max_area']
        )

        if trackbar_values['show_filtered_contours']:
            for contour in filtered_contours:
                text_x, text_y = get_shape_center(contour)

                best_shape = "Unknown"
                best_score = float("inf")

                for name, ref_contour in FIGURE_TEMPLATES.items():
                    score = cv2.matchShapes(
                        contour, ref_contour, cv2.CONTOURS_MATCH_I1, 0)

                    if score <= trackbar_values['min_shape_score'] and score < best_score:
                        best_score = score
                        best_shape = name

                cv2.putText(main_frame, text=best_shape, org=(text_x, text_y),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                            color=COLOR_GREEN if best_shape != "Unknown" else COLOR_RED, thickness=2)

                color = COLOR_GREEN if best_shape != "Unknown" else COLOR_RED
                draw_contours(main_frame, [contour], color)

        if trackbar_values['show_all_contours']:
            draw_contours(main_frame, contours, COLOR_GREEN)

        cv2.imshow(main_frame_name, main_frame)
        cv2.imshow(processed_frame_name, denoised_frame)
        if check_frame_exit():
            break

    capture.release()


main()
