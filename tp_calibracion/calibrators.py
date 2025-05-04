import cv2
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class LineDetector:
    @staticmethod
    def detect_lines(image, method='hough', min_line_length=200):
        """Multi-method line detection with auto-fallback"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        if method == 'hough':
            # Standard Hough Line Transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                  minLineLength=min_line_length,
                                  maxLineGap=20)
            if lines is not None:
                lines = lines.reshape(-1, 1, 4)  # Ensure consistent shape
        elif method == 'lsd':
            # Line Segment Detector (more advanced)
            lsd = cv2.createLineSegmentDetector(0)
            detected_lines = lsd.detect(edges)
            lines = detected_lines[0] if detected_lines[0] is not None else np.array([])
        else:
            raise ValueError("Invalid detection method")
            
        return lines if lines is not None and len(lines) > 0 else np.array([])

    @staticmethod
    def auto_detect(image, min_line_count=10):
        """Automatically tries multiple detection methods"""
        methods = ['hough', 'lsd']
        best_lines = np.array([])
        
        for method in methods:
            lines = LineDetector.detect_lines(image, method)
            if len(lines) >= min_line_count:
                if len(lines) > len(best_lines):
                    best_lines = lines
        
        if len(best_lines) >= min_line_count:
            return best_lines
        
        # If automatic fails, prompt user for manual input
        print("Automatic detection failed. Please manually mark lines.")
        return LineDetector.manual_line_input(image)

    @staticmethod
    def manual_line_input(image):
        """Allows user to manually draw lines"""
        print("Click to mark line start and end points (press 'q' when done)")
        
        lines = []
        clone = image.copy()
        
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if 'start' not in param:
                    param['start'] = (x, y)
                    cv2.circle(clone, (x, y), 5, (0,255,0), -1)
                else:
                    param['end'] = (x, y)
                    cv2.line(clone, param['start'], param['end'], (0,255,0), 2)
                    lines.append([param['start'][0], param['start'][1], x, y])
                    del param['start']
                cv2.imshow("Manual Line Input", clone)
        
        cv2.imshow("Manual Line Input", clone)
        cv2.setMouseCallback("Manual Line Input", click_event, {})
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        return np.array(lines).reshape(-1, 1, 4)

class CameraCalibrator:
    @staticmethod
    def calibrate(image, auto_detect=True, show_steps=True):
        """Complete calibration workflow"""
        if show_steps:
            cv2.imshow("Original Image", image)
            cv2.waitKey(500)
        
        # 1. Line Detection
        if auto_detect:
            lines = LineDetector.auto_detect(image)
        else:
            lines = LineDetector.detect_lines(image)
        
        if len(lines) < 4:
            raise ValueError("Insufficient lines detected for calibration")
        
        if show_steps:
            line_img = image.copy()
            for line in lines:
                x1, y1, x2, y2 = line[0] if lines.ndim == 3 else line
                cv2.line(line_img, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
            cv2.imshow("Detected Lines", line_img)
            cv2.waitKey(1000)
        
        # 2. Sample points along lines
        points = []
        for line in lines:
            if lines.ndim == 3:
                x1, y1, x2, y2 = line[0]
            else:
                x1, y1, x2, y2 = line
            
            # More points for better accuracy
            for t in np.linspace(0, 1, 100):
                points.append([x1 + t*(x2-x1), y1 + t*(y2-y1)])
        
        points = np.array(points, dtype=np.float32)
        
        # 3. Set up camera matrix
        h, w = image.shape[:2]
        camera_matrix = np.array([[w, 0, w/2], [0, h, h/2], [0, 0, 1]], dtype=np.float64)
        
        # 4. Combined calibration
        def objective(params):
            k1, k2, p1, p2, k3 = params
            dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
            
            try:
                undistorted = cv2.undistortPoints(
                    points.reshape(-1, 1, 2),
                    camera_matrix,
                    dist_coeffs,
                    P=camera_matrix
                ).reshape(-1, 2)
                
                error = CameraCalibrator._calculate_deviation(undistorted)
                return error  # Minimize this value
            except:
                return float('inf')
        
        # Start with small distortion
        initial_guess = [0.0, 0.0, 0.0, 0.0, 0.0]
        # Reasonable bounds for distortion coefficients
        bounds = [(-1, 1), (-1, 1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)]
        result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
        
        # 5. Apply correction
        dist_coeffs = np.array(result.x, dtype=np.float64)
        undistorted_img = cv2.undistort(image, camera_matrix, dist_coeffs)
        
        if show_steps:
            cv2.imshow("Corrected Image", undistorted_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return {
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs,
            'undistorted_image': undistorted_img
        }

    @staticmethod
    def _calculate_deviation(points):
        """Calculates how straight the points are"""
        total_error = 0
        valid_groups = 0
        
        # Split points into groups (assuming 100 points per line)
        for i in range(0, len(points), 100):
            group = points[i:i+100]
            if len(group) >= 3:
                line = cv2.fitLine(group, cv2.DIST_L2, 0, 0.01, 0.01)
                vx, vy, x0, y0 = line.flatten()
                
                # Calculate perpendicular distances
                distances = np.abs((group[:,0]-x0)*vy - (group[:,1]-y0)*vx)/np.sqrt(vx**2 + vy**2)
                total_error += np.mean(distances)
                valid_groups += 1
        
        return total_error / valid_groups if valid_groups > 0 else float('inf')

def main():
    # Try loading the distorted calibration image first
    image = cv2.imread("calibration_image.jpg")  # Change to your image path
    if image is None:
        image = cv2.imread("./img/aleph.png")
    if image is None:
        print("Error: No image found")
        return
    
    try:
        print("Starting calibration...")
        results = CameraCalibrator.calibrate(image, show_steps=True)
        
        print("\nCalibration Results:")
        print(f"Camera Matrix:\n{results['camera_matrix']}")
        print(f"Distortion Coefficients (k1,k2,p1,p2,k3): {results['dist_coeffs']}")
        
        # Save results
        np.savez("calibration_results.npz",
                camera_matrix=results['camera_matrix'],
                dist_coeffs=results['dist_coeffs'])
        cv2.imwrite("corrected_image.png", results['undistorted_image'])
        
        # Show comparison
        plt.figure(figsize=(12, 6))
        plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original')
        plt.subplot(122), plt.imshow(cv2.cvtColor(results['undistorted_image'], cv2.COLOR_BGR2RGB)), plt.title('Corrected')
        plt.show()
        
    except Exception as e:
        print(f"Calibration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()