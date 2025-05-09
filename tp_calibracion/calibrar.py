import numpy as np
import cv2 as cv
import os

def main():
    print("""
    Calibrador de cámara simple
    ---------------------------
    Instrucciones:
    ESPACIO: captura imagen (cuando se detecte el patrón)
    C: calibra con las imágenes capturadas
    S: guarda parámetros de calibración
    A: aplica calibración a imagen de prueba
    ESC: salir
    """)

    # Configuración
    ESC = 27
    SPACE = 32
    C_KEY = ord('c')
    S_KEY = ord('s')
    A_KEY = ord('a')
    
    # Parámetros del tablero de ajedrez (esquinas interiores)
    chessBoard = (7, 7)  # Si tu tablero es diferente, cámbialo aquí
    square_size = 1.0    # Unidades arbitrarias, ajustar si necesitas medidas reales
    
    # Criterios para refinamiento de esquinas
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Crear ventanas
    cv.namedWindow("Cámara", cv.WINDOW_NORMAL)
    cv.namedWindow("Detecciones", cv.WINDOW_NORMAL)
    
    # Preparar puntos de objeto 3D (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((chessBoard[0] * chessBoard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessBoard[0], 0:chessBoard[1]].T.reshape(-1, 2) * square_size
    
    # Arrays para almacenar puntos de objeto y puntos de imagen
    objpoints = []  # puntos 3D en espacio real
    imgpoints = []  # puntos 2D en el plano de la imagen
    
    # Iniciar cámara
    cam = cv.VideoCapture(0)
    width = cam.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cam.get(cv.CAP_PROP_FRAME_HEIGHT)
    print(f"Resolución de cámara: {width} x {height}")
    
    # Imagen para mostrar las detecciones (negra inicialmente)
    img_size = (640, 480)
    detections_img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
    
    # Variables para calibración
    ret_calib = False
    mtx = None
    dist = None
    
    # Contador de imágenes capturadas
    img_count = 0
    
    while True:
        # Capturar imagen de la cámara
        ret, frame = cam.read()
        if not ret:
            print("Error al capturar imagen")
            break
        
        # Convertir a escala de grises
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Buscar esquinas del tablero
        ret, corners = cv.findChessboardCorners(gray, chessBoard, None)
        
        # Dibujar frame con información
        frame_display = frame.copy()
        
        # Si se encuentra el patrón, dibujar esquinas
        if ret:
            # Refinar esquinas
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Dibujar esquinas
            cv.drawChessboardCorners(frame_display, chessBoard, corners2, ret)
            
            # Mostrar estado
            cv.putText(frame_display, "Patron detectado - ESPACIO para capturar", 
                      (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv.putText(frame_display, "No se detecta patron", 
                      (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Mostrar contador de imágenes
        cv.putText(frame_display, f"Imagenes capturadas: {img_count}", 
                  (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Mostrar imagen
        cv.imshow("Cámara", frame_display)
        cv.imshow("Detecciones", detections_img)
        
        # Esperar tecla
        key = cv.waitKey(1)
        
        # Procesar teclas
        if key == ESC:
            print("Saliendo...")
            break
        
        elif key == SPACE:
            # Capturar imagen si se detecta el patrón
            if ret:
                img_count += 1
                print(f"Imagen {img_count} capturada")
                
                # Guardar puntos
                objpoints.append(objp)
                imgpoints.append(corners2)
                
                # Actualizar imagen de detecciones
                # Fundir la imagen anterior para simular un efecto de persistencia
                detections_img = cv.addWeighted(detections_img, 0.7, np.zeros_like(detections_img), 0, 0)
                
                # Redimensionar frame para que coincida con detections_img si es necesario
                frame_resized = cv.resize(frame, (detections_img.shape[1], detections_img.shape[0]))
                
                # Dibujar esquinas en la imagen de detecciones
                temp_img = frame_resized.copy()
                cv.drawChessboardCorners(temp_img, chessBoard, 
                                       cv.resize(corners2, (0, 0), fx=detections_img.shape[1]/frame.shape[1], 
                                                fy=detections_img.shape[0]/frame.shape[0]), ret)
                
                # Añadir a imagen de detecciones
                detections_img = cv.addWeighted(detections_img, 1, temp_img, 0.3, 0)
        
        elif key == C_KEY:
            # Calibrar cámara con las imágenes capturadas
            if len(objpoints) > 0:
                print("Calibrando cámara...")
                
                ret_calib, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
                    objpoints, imgpoints, gray.shape[::-1], None, None)
                
                if ret_calib:
                    print("\n=== RESULTADOS DE CALIBRACIÓN ===")
                    print(f"Error de reproyección: {ret_calib}")
                    print("\nMatriz de la cámara:")
                    print(mtx)
                    print("\nCoeficientes de distorsión [k1, k2, p1, p2, k3]:")
                    print(dist.ravel())
                    
                    # Calcular error de reproyección
                    mean_error = 0
                    for i in range(len(objpoints)):
                        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
                        mean_error += error
                    
                    print(f"Error total: {mean_error/len(objpoints)}")
                else:
                    print("Calibración fallida")
            else:
                print("No hay suficientes imágenes para calibrar")
        
        elif key == S_KEY:
            # Guardar parámetros de calibración
            if ret_calib:
                # Crear directorio si no existe
                if not os.path.exists("calibration_results"):
                    os.makedirs("calibration_results")
                
                # Guardar parámetros
                np.savez("calibration_results/camera_params.npz", 
                        mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
                
                # Guardar en formato XML
                fs = cv.FileStorage("calibration_results/camera_params.xml", cv.FILE_STORAGE_WRITE)
                fs.write("camera_matrix", mtx)
                fs.write("distortion_coefficients", dist)
                fs.release()
                
                print("Parámetros guardados en 'calibration_results'")
            else:
                print("No hay parámetros de calibración para guardar")
        
        elif key == A_KEY:
            # Aplicar calibración a imagen actual
            if ret_calib:
                # Obtener nuevos parámetros de cámara
                h, w = frame.shape[:2]
                newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
                
                # Corregir distorsión
                dst = cv.undistort(frame, mtx, dist, None, newcameramtx)
                
                # Recortar región de interés
                x, y, w, h = roi
                if w > 0 and h > 0:  # Verificar ROI válido
                    dst = dst[y:y+h, x:x+w]
                
                # Mostrar imagen original y corregida
                comparison = np.hstack((frame, cv.resize(dst, (frame.shape[1], frame.shape[0]))))
                cv.namedWindow("Comparación", cv.WINDOW_NORMAL)
                cv.imshow("Comparación", comparison)
                
                # Guardar imagen corregida
                if not os.path.exists("calibration_results"):
                    os.makedirs("calibration_results")
                cv.imwrite("calibration_results/calibrated.jpg", dst)
                print("Imagen calibrada guardada como 'calibration_results/calibrated.jpg'")
            else:
                print("No hay parámetros de calibración disponibles")
    
    # Liberar recursos
    cam.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()