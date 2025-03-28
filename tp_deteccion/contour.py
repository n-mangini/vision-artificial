import cv2

COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)


def filter_contours_by_area(contours, min_area, max_area):
    filtered_contours = []
    for cnt in contours:
        if min_area <= cv2.contourArea(cnt) <= max_area:
            filtered_contours.append(cnt)
    return filtered_contours


def draw_contours(frame, contours, color):
    # contourIdx=-1: Draw all contours
    cv2.drawContours(frame, contours, contourIdx=-1,
                     color=color, thickness=2)
