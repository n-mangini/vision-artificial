import cv2
import numpy as np
from config import handle_video_capture, choose_camera_by_OS

points = []
homography_matrix = None
modo = 'visualizacion'

dst_points = np.array([
    [0, 0],
    [300, 0],
    [300, 300],
    [0, 300]
], dtype=np.float32)


def order_points(puntos):
    rect = np.zeros((4, 2), dtype="float32")
    s = puntos.sum(axis=1)
    # top-left
    rect[0] = puntos[np.argmin(s)]
    # bottom-right
    rect[2] = puntos[np.argmax(s)]
    diff = np.diff(puntos, axis=1)
    # top-right
    rect[1] = puntos[np.argmin(diff)]
    # bottom-left
    rect[3] = puntos[np.argmax(diff)]
    return rect


def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        print(f"Seleccionado: ({x}, {y})")


def detect_qr_points(frame):
    detector = cv2.QRCodeDetector()
    retval, points = detector.detect(frame)
    if retval:
        return order_points(points[0].astype(np.float32))
    return None


def find_homography(src_points):
    homography_matrix, _ = cv2.findHomography(src_points, dst_points)
    return homography_matrix


def apply_homography(frame, homography_matrix):
    if homography_matrix is not None:
        warped = cv2.warpPerspective(frame, homography_matrix, (300, 300))
        cv2.imshow("Vista frontal", warped)


def is_invertible(matrix):
    return (
        matrix is not None and
        matrix.shape == (3, 3) and
        np.linalg.cond(matrix) < 1 / np.finfo(matrix.dtype).eps
    )


def draw_grid(frame, homography_matrix, grid_size=3):
    if not is_invertible(homography_matrix):
        cv2.putText(frame, "Homografía inválida", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return

    step = 300 // grid_size
    grid_points = [[i * step, j * step]
                   for i in range(grid_size + 1) for j in range(grid_size + 1)]
    grid_points = np.array(grid_points, dtype=np.float32).reshape(-1, 1, 2)
    inv_h = np.linalg.inv(homography_matrix)
    transformed_points = cv2.perspectiveTransform(grid_points, inv_h)
    transformed_points = transformed_points.reshape(
        (grid_size + 1, grid_size + 1, 2)).astype(int)

    for i in range(grid_size + 1):
        for j in range(grid_size):
            cv2.line(frame, tuple(transformed_points[i][j]), tuple(
                transformed_points[i][j + 1]), (255, 0, 0), 1)

    for j in range(grid_size + 1):
        for i in range(grid_size):
            cv2.line(frame, tuple(transformed_points[i][j]), tuple(
                transformed_points[i + 1][j]), (255, 0, 0), 1)


main_frame_name = "Webcam"
capture = handle_video_capture(main_frame_name, choose_camera_by_OS())


print("Presioná 'q' para modo QR continuo, 'h' para modo manual, 'ESC' para salir")

while True:
    homography_matrix = None
    ret, frame = capture.read()
    if not ret:
        break
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('q'):
        modo = 'qr'
        print("Modo automático activado")
    elif key == ord('h'):
        modo = 'manual'
        print("Modo manual activado")
        points = []
        cv2.setMouseCallback("Webcam", click_event)

    if modo == 'qr':
        pts = detect_qr_points(frame)
        if pts is not None:
            homography_matrix = find_homography(pts)

    elif modo == 'manual' and len(points) == 4 and homography_matrix is None:
        homography_matrix = find_homography(np.array(points, dtype=np.float32))

    draw_grid(frame, homography_matrix)
    apply_homography(frame, homography_matrix)

    cv2.imshow("Webcam", frame)

capture.release()
cv2.destroyAllWindows()
