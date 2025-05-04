import cv2
import numpy as np
from typing import Tuple, List
from scipy.optimize import minimize


class FullDistortionModel:
    """Modelo completo de distorsión sin estado interno"""
    
    @staticmethod
    def calibrate_from_image(image: np.ndarray) -> Tuple[float, float, float, float, float]:
        """Auto-calibración completa desde una sola imagen"""
        
        # Crear matriz de cámara inicial
        h, w = image.shape[:2]
        camera_matrix = np.array([
            [w, 0, w/2],
            [0, w, h/2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Calibrar todos los parámetros de distorsión
        k1, k2, p1, p2 = FullDistortionModel._calibrate_all_parameters(image, camera_matrix)
        
        # No usamos k3 (distorsión radial de alto orden)
        k3 = 0.0
        
        print(f"Calibración radial: k1={k1:.6f}, k2={k2:.6f}")
        print(f"Calibración tangencial: p1={p1:.6f}, p2={p2:.6f}")
        
        return k1, k2, k3, p1, p2
    
    @staticmethod
    def _calibrate_all_parameters(image: np.ndarray, camera_matrix: np.ndarray) -> Tuple[float, float, float, float]:
        """Calibra todos los parámetros de distorsión (k1, k2, p1, p2)"""
        
        # Preparar imagen
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Detectar líneas largas
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=300, maxLineGap=10)
        
        if lines is None or len(lines) < 3:
            print("No se encontraron suficientes líneas para calibrar")
            return 0.0, 0.0, 0.0, 0.0
        
        # Recolectar puntos de líneas largas
        line_points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            if line_length > 400:  # Solo líneas largas
                # Extraer puntos a lo largo de la línea
                points = FullDistortionModel._extract_line_points(edges, x1, y1, x2, y2)
                if len(points) > 10:
                    line_points.extend(points)
        
        if len(line_points) < 100:
            print("No se encontraron suficientes puntos para calibrar")
            return 0.0, 0.0, 0.0, 0.0
        
        line_points = np.array(line_points)
        
        # Optimizar parámetros
        def objective_function(params):
            k1, k2, p1, p2 = params
            
            # Convertir puntos y aplicar corrección
            points_reshaped = line_points.reshape(-1, 1, 2).astype(np.float32)
            dist_coeffs = np.array([k1, k2, p1, p2, 0])
            
            try:
                undistorted = cv2.undistortPoints(points_reshaped, camera_matrix, 
                                                 dist_coeffs, P=camera_matrix)
                undistorted = undistorted.reshape(-1, 2)
            except:
                return float('inf')
            
            # Medir qué tan rectas quedan las líneas
            total_error = 0
            group_size = 50
            
            for i in range(0, len(undistorted), group_size):
                group = undistorted[i:i+group_size]
                if len(group) > 10:
                    # Ajustar línea recta
                    [vx, vy, x, y] = cv2.fitLine(group.astype(np.float32), 
                                                 cv2.DIST_L2, 0, 0.01, 0.01)
                    
                    # Calcular desviación promedio
                    for point in group:
                        d = abs(vx * (point[1] - y) - vy * (point[0] - x)) / np.sqrt(vx**2 + vy**2)
                        total_error += d
            
            return total_error
        
        # Encontrar parámetros óptimos para todos los coeficientes
        result = minimize(objective_function, [0.0, 0.0, 0.0, 0.0], method='Nelder-Mead',
                         options={'xatol': 1e-6, 'fatol': 1e-6})
        
        k1, k2, p1, p2 = result.x
        
        # Solo limitar valores extremos para evitar errores numéricos
        k1 = max(min(k1, 10.0), -10.0)
        k2 = max(min(k2, 10.0), -10.0)
        p1 = max(min(p1, 1.0), -1.0)
        p2 = max(min(p2, 1.0), -1.0)
        
        return k1, k2, p1, p2
    
    @staticmethod
    def _extract_line_points(edge_image: np.ndarray, x1: int, y1: int, 
                           x2: int, y2: int, num_points: int = 50) -> List[np.ndarray]:
        """Extrae puntos a lo largo de una línea en la imagen de bordes"""
        points = []
        
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            
            # Buscar el borde más cercano en un radio pequeño
            found = False
            for radius in range(5):
                if found:
                    break
                for dx in range(-radius, radius+1):
                    for dy in range(-radius, radius+1):
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < edge_image.shape[1] and 
                            0 <= ny < edge_image.shape[0] and 
                            edge_image[ny, nx] > 0):
                            points.append([nx, ny])
                            found = True
                            break
        
        return np.array(points) if points else np.array([])
    
    @staticmethod
    def remove_distortion(points: np.ndarray, camera_matrix: np.ndarray,
                         k1: float, k2: float, k3: float,
                         p1: float, p2: float) -> np.ndarray:
        """Remueve ambas distorsiones de puntos"""
        dist_coeffs = np.array([k1, k2, p1, p2, k3])
        points_reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        undistorted = cv2.undistortPoints(points_reshaped, camera_matrix, dist_coeffs, P=camera_matrix)
        return undistorted.reshape(-1, 2)
    
    @staticmethod
    def undistort_image(image: np.ndarray, k1: float, k2: float, k3: float,
                       p1: float, p2: float) -> np.ndarray:
        """Corrige una imagen completa de distorsión usando los parámetros especificados"""
        h, w = image.shape[:2]
        
        # Crear matriz de cámara
        camera_matrix = np.array([
            [w, 0, w/2],
            [0, w, h/2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Crear vector de distorsión
        dist_coeffs = np.array([k1, k2, p1, p2, k3])
        
        # Crear mapa de corrección
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_32FC1
        )
        
        # Aplicar corrección
        undistorted_image = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
        
        return undistorted_image
    
    @staticmethod
    def visualize_detection(image: np.ndarray):
        """Visualiza la detección de líneas para debug (opcional)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Detectar líneas
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=300, maxLineGap=10)
        
        # Crear imagen de visualización
        line_image = image.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Mostrar resultados
        cv2.imshow('Bordes detectados', edges)
        cv2.imshow('Líneas detectadas', line_image)
        print("Líneas detectadas para calibración. Presiona cualquier tecla para continuar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()