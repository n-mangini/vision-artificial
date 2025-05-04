import cv2
import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple

class RadialDistortion:
    """Clase estática para manejar distorsión radial sin estado interno"""
    
    @staticmethod
    def remove_distortion(points: np.ndarray, camera_matrix: np.ndarray,
                         k1: float, k2: float, k3: float) -> np.ndarray:
        """Remueve distorsión radial de puntos usando OpenCV"""
        dist_coeffs = np.array([k1, k2, 0, 0, k3])
        points_reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        undistorted = cv2.undistortPoints(points_reshaped, camera_matrix, dist_coeffs, P=camera_matrix)
        return undistorted.reshape(-1, 2)
    
    @staticmethod
    def calibrate_from_lines(curved_lines: List[np.ndarray], camera_matrix: np.ndarray) -> Tuple[float, float]:
        """Calibra parámetros k1, k2 a partir de líneas curvas"""
        def objective_function(params):
            k1, k2 = params
            total_error = 0
            
            for line_points in curved_lines:
                undistorted_points = RadialDistortion.remove_distortion(line_points, camera_matrix, k1, k2, 0.0)
                error = RadialDistortion._measure_line_straightness(undistorted_points)
                total_error += error
            
            return total_error
        
        result = minimize(objective_function, [0.0, 0.0], method='Nelder-Mead')
        return result.x[0], result.x[1]
    
    @staticmethod
    def _measure_line_straightness(points: np.ndarray) -> float:
        """Mide qué tan recta es una línea (menor valor = más recta)"""
        if len(points) < 3:
            return float('inf')
            
        # Ajustar una línea recta a los puntos
        [vx, vy, x, y] = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Calcular distancias perpendiculares a la línea
        distances = []
        for point in points:
            # Ecuación de la línea: vx(x - x0) = vy(y - y0)
            # Distancia punto a línea
            d = abs(vx * (point[1] - y) - vy * (point[0] - x)) / np.sqrt(vx**2 + vy**2)
            distances.append(d)
        
        return np.mean(distances)