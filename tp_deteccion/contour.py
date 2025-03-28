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


def get_saved_contour(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        image=image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    return contours[1]


def match_shapes(frame_contours, contour_to_compare):
    return cv2.matchShapes(
        contour1=frame_contours, contour2=contour_to_compare, method=cv2.CONTOURS_MATCH_I1, parameter=0)


def get_shape_center(contour):
    x, y, w, h = cv2.boundingRect(contour)
    center_x = x + w // 2
    center_y = y + h // 2
    return (center_x, center_y)
