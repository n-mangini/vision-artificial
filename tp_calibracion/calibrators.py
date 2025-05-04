import cv2
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class LineDetector:
    @staticmethod
    def detect_lines(image, method='hough', min_line_length=150, min_distance=30):
        """Detección de líneas optimizada"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Suavizado para reducir ruido
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detección de bordes más sensible
        edges = cv2.Canny(blurred, 30, 150, apertureSize=3)
        
        if method == 'hough':
            # Hough probabilística con parámetros ajustados
            lines = cv2.HoughLinesP(
                edges, 
                rho=1, 
                theta=np.pi/180, 
                threshold=30,
                minLineLength=min_line_length,
                maxLineGap=min_distance
            )
            
            # Filtrar líneas cercanas
            if lines is not None:
                filtered_lines = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Calcular longitud
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if length > min_line_length:
                        filtered_lines.append(line)
                
                if filtered_lines:
                    lines = np.array(filtered_lines).reshape(-1, 1, 4)
        
        elif method == 'lsd':
            # Line Segment Detector
            lsd = cv2.createLineSegmentDetector(0)
            detected_lines = lsd.detect(edges)
            lines = detected_lines[0] if detected_lines[0] is not None else None
        else:
            raise ValueError("Método de detección inválido")
            
        return lines if lines is not None and len(lines) > 0 else np.array([])

    @staticmethod
    def auto_detect(image, min_line_count=8):
        """Detección automática mejorada"""
        methods = ['hough', 'lsd']
        best_lines = np.array([])
        
        for method in methods:
            lines = LineDetector.detect_lines(image, method)
            if len(lines) >= min_line_count:
                return lines
            elif len(lines) > len(best_lines):
                best_lines = lines
        
        if len(best_lines) < min_line_count:
            raise ValueError(f"Error: No se encontraron suficientes líneas ({len(best_lines)}/{min_line_count})")
            
        return best_lines

class DistortionCalculator:
    @staticmethod
    def calculate_line_deviation(points):
        """Calcula qué tan rectas son las líneas detectadas"""
        if len(points) < 3:
            return float('inf')
            
        # Ajustar una línea a los puntos
        [vx, vy, x0, y0] = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Calcular distancias perpendiculares
        a, b, c = vy, -vx, vx*y0 - vy*x0
        denominator = np.sqrt(a**2 + b**2)
        
        if denominator == 0:
            return float('inf')
            
        distances = np.abs(a*points[:, 0] + b*points[:, 1] + c) / denominator
        
        return np.mean(distances)
    
    @staticmethod
    def optimize_distortion(points, camera_matrix):
        """Optimiza los coeficientes de distorsión usando un enfoque más robusto"""
        h, w = camera_matrix[1, 2] * 2, camera_matrix[0, 2] * 2
        
        def objective(params):
            k1, k2, p1, p2, k3 = params
            dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
            
            try:
                # Corregir puntos
                undistorted = cv2.undistortPoints(
                    points.reshape(-1, 1, 2),
                    camera_matrix,
                    dist_coeffs,
                    P=camera_matrix
                ).reshape(-1, 2)
                
                # Calcular error total
                total_error = 0
                point_groups = np.array_split(undistorted, len(undistorted) // 50)
                
                for group in point_groups:
                    if len(group) > 2:
                        error = DistortionCalculator.calculate_line_deviation(group)
                        if not np.isinf(error):
                            total_error += error
                
                # Penalizar coeficientes extremos
                penalty = 0
                for param in params:
                    if abs(param) > 1.0:
                        penalty += (abs(param) - 1.0) ** 2
                
                return total_error + penalty
                
            except:
                return float('inf')
        
        # Múltiples inicializaciones para evitar mínimos locales
        best_result = None
        best_error = float('inf')
        
        initial_guesses = [
            [0.0, 0.0, 0.0, 0.0, 0.0],  # Sin distorsión
            [-0.5, 0.0, 0.0, 0.0, 0.0], # Barrel
            [0.5, 0.0, 0.0, 0.0, 0.0],  # Pincushion
        ]
        
        bounds = [(-2, 2), (-1, 1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)]
        
        for initial_guess in initial_guesses:
            result = minimize(
                objective, 
                initial_guess, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000}
            )
            
            if result.fun < best_error:
                best_error = result.fun
                best_result = result
        
        return best_result.x

class CameraCalibrator:
    @staticmethod
    def calibrate(image, show_steps=True, min_lines=6):
        """Flujo de calibración mejorado"""
        if show_steps:
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title('Imagen Original')
            plt.axis('off')
            plt.show()
        
        # 1. Detección de líneas
        lines = LineDetector.auto_detect(image, min_line_count=min_lines)
        
        if show_steps:
            line_img = image.copy()
            colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0] if lines.ndim == 3 else line
                color = colors[i % len(colors)]
                cv2.line(line_img, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
            plt.title(f'Líneas Detectadas ({len(lines)} líneas)')
            plt.axis('off')
            plt.show()
        
        # 2. Extraer puntos de las líneas
        points = []
        for line in lines:
            if lines.ndim == 3:
                x1, y1, x2, y2 = line[0]
            else:
                x1, y1, x2, y2 = line
            
            # Muestrear puntos uniformemente
            num_samples = int(np.sqrt((x2-x1)**2 + (y2-y1)**2) / 5)
            num_samples = min(max(num_samples, 10), 100)
            
            for t in np.linspace(0, 1, num_samples):
                points.append([x1 + t*(x2-x1), y1 + t*(y2-y1)])
        
        points = np.array(points, dtype=np.float32)
        
        # 3. Matriz de cámara inicial
        h, w = image.shape[:2]
        # Estimar focal length basado en el tamaño de la imagen
        focal = max(h, w) * 1.2
        camera_matrix = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]], dtype=np.float64)
        
        # 4. Optimizar coeficientes de distorsión
        dist_coeffs = DistortionCalculator.optimize_distortion(points, camera_matrix)
        
        # 5. Corregir imagen
        undistorted_img = cv2.undistort(image, camera_matrix, dist_coeffs)
        
        if show_steps:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax1.set_title('Imagen Original')
            ax1.axis('off')
            
            ax2.imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
            ax2.set_title('Imagen Corregida')
            ax2.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return {
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs,
            'undistorted_image': undistorted_img,
            'radial_coeffs': {'k1': dist_coeffs[0], 'k2': dist_coeffs[1], 'k3': dist_coeffs[4]},
            'tangential_coeffs': {'p1': dist_coeffs[2], 'p2': dist_coeffs[3]}
        }
    
    @staticmethod
    def show_calibration_results(results):
        """Muestra resultados detallados"""
        print("\n=== RESULTADOS DE CALIBRACIÓN ===")
        print("\nMatriz de Cámara:")
        print(results['camera_matrix'])
        
        print("\nCoeficientes de Distorsión:")
        print(f"Array completo: {results['dist_coeffs']}")
        
        print("\nCoeficientes Radiales:")
        print(f"k1 (principal): {results['radial_coeffs']['k1']:.6f}")
        print(f"k2 (secundario): {results['radial_coeffs']['k2']:.6f}")
        print(f"k3 (terciario): {results['radial_coeffs']['k3']:.6f}")
        
        print("\nCoeficientes Tangenciales:")
        print(f"p1: {results['tangential_coeffs']['p1']:.6f}")
        print(f"p2: {results['tangential_coeffs']['p2']:.6f}")
        
        # Interpretación
        print("\n=== INTERPRETACIÓN ===")
        k1 = results['radial_coeffs']['k1']
        if k1 > 0.1:
            print("Distorsión POSITIVA (pincushion) dominante")
        elif k1 < -0.1:
            print("Distorsión NEGATIVA (barrel) dominante")
        else:
            print("Distorsión radial mínima")

def main():
    # Cargar imagen
    image = cv2.imread("./img/aleph.png")
    if image is None:
        print("Error: No se encontró la imagen")
        return
    
    try:
        print("Iniciando calibración...")
        results = CameraCalibrator.calibrate(image, show_steps=True)
        
        # Mostrar resultados
        CameraCalibrator.show_calibration_results(results)
        
        # Guardar resultados
        np.savez("calibration_results.npz",
                camera_matrix=results['camera_matrix'],
                dist_coeffs=results['dist_coeffs'])
        cv2.imwrite("corrected_image.png", results['undistorted_image'])
        
        print("\nArchivos guardados:")
        print("- calibration_results.npz")
        print("- corrected_image.png")
        
    except Exception as e:
        print(f"Error en calibración: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()