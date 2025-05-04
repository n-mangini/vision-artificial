import cv2
import numpy as np
from scipy.optimize import minimize
from typing import Tuple

class TangentialDistortion:
    """Clase estática para manejar distorsión tangencial sin estado interno"""
    
    @staticmethod
    def remove_distortion(points: np.ndarray, camera_matrix: np.ndarray,
                         p1: float, p2: float) -> np.ndarray:
        """Remueve distorsión tangencial de puntos usando OpenCV"""
        dist_coeffs = np.array([0, 0, p1, p2, 0])
        points_reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        undistorted = cv2.undistortPoints(points_reshaped, camera_matrix, dist_coeffs, P=camera_matrix)
        return undistorted.reshape(-1, 2)
    
    @staticmethod
    def calibrate_from_pattern(pattern_points: np.ndarray, image_points: np.ndarray, 
                              camera_matrix: np.ndarray) -> Tuple[float, float]:
        """Calibra parámetros p1, p2 a partir de un patrón conocido"""
        def objective_function(params):
            p1, p2 = params
            
            # Proyectar puntos 3D usando parámetros actuales
            dist_coeffs = np.array([0, 0, p1, p2, 0])
            rvec = np.zeros(3)
            tvec = np.array([0, 0, 500])  # Asunción inicial
            
            _, rvec, tvec = cv2.solvePnP(pattern_points, image_points, 
                                       camera_matrix, dist_coeffs,
                                       rvec, tvec, useExtrinsicGuess=True)
            
            projected_points, _ = cv2.projectPoints(pattern_points, rvec, tvec,
                                                   camera_matrix, dist_coeffs)
            
            # Calcular error de reproyección
            error = np.sum(np.linalg.norm(image_points - projected_points.reshape(-1, 2), axis=1))
            return error
        
        result = minimize(objective_function, [0.0, 0.0], method='Nelder-Mead')
        return result.x[0], result.x[1]