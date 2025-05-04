import cv2
import numpy as np
from scipy.optimize import minimize

def detect_strong_lines(image, min_line_length=300, show_detection=False):
    """Detects straight lines in the image with adjustable sensitivity"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines with adaptive threshold
    lines = cv2.HoughLinesP(edges, 
                          rho=1, 
                          theta=np.pi/180, 
                          threshold=50, 
                          minLineLength=min_line_length,
                          maxLineGap=20)
    
    if lines is None:
        raise ValueError("No strong lines detected. Try with a different image or adjust parameters.")
    
    if show_detection:
        line_img = image.copy()
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.imshow("Detected Lines", line_img)
        cv2.waitKey(1000)
    
    return lines

def sample_line_points(lines, num_points=50):
    """Samples points along detected lines"""
    line_points = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        for t in np.linspace(0, 1, num_points):
            line_points.append([x1 + t*(x2-x1), y1 + t*(y2-y1)])
    return np.array(line_points, dtype=np.float32)

def compute_line_straightness(points, group_size=50):
    """Measures how straight the points are when grouped"""
    error = 0
    valid_groups = 0
    
    for i in range(0, len(points), group_size):
        group = points[i:i+group_size]
        if len(group) > 10:
            line_params = cv2.fitLine(group, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x0, y0 = line_params.flatten()
            distances = np.abs((group[:,0]-x0)*vy - (group[:,1]-y0)*vx)/np.sqrt(vx**2 + vy**2)
            error += np.mean(distances)
            valid_groups += 1
    
    return error / valid_groups if valid_groups > 0 else float('inf')

def calibrate_radial_distortion(image, initial_k=(0.1, -0.1, 0), show_detection=False):
    """Calibrates radial distortion parameters (k1, k2, k3)"""
    lines = detect_strong_lines(image, show_detection=show_detection)
    line_points = sample_line_points(lines)
    
    h, w = image.shape[:2]
    camera_matrix = np.array([[max(w,h), 0, w/2], 
                            [0, max(w,h), h/2], 
                            [0, 0, 1]], dtype=np.float32)
    
    def objective(k_params):
        k1, k2, k3 = k_params
        dist_coeffs = np.array([k1, k2, 0, 0, k3])
        
        try:
            undistorted = cv2.undistortPoints(
                line_points.reshape(-1, 1, 2),
                camera_matrix,
                dist_coeffs,
                P=camera_matrix
            ).reshape(-1, 2)
            
            return compute_line_straightness(undistorted)
        except:
            return float('inf')
    
    bounds = [(-1.0, 1.0), (-1.0, 1.0), (-0.2, 0.2)]
    result = minimize(objective, initial_k, method='L-BFGS-B', bounds=bounds)
    
    # Filter insignificant coefficients
    k1, k2, k3 = result.x
    return np.array([
        k1 if abs(k1) > 1e-5 else 0,
        k2 if abs(k2) > 1e-5 else 0,
        0, 0,  # p1, p2 (tangential handled separately)
        k3 if abs(k3) > 1e-5 else 0
    ])

def calibrate_tangential_distortion(image, initial_p=(0.001, -0.001), show_detection=False):
    """Calibrates tangential distortion parameters (p1, p2)"""
    lines = detect_strong_lines(image, show_detection=show_detection)
    line_points = sample_line_points(lines)
    
    h, w = image.shape[:2]
    camera_matrix = np.array([[max(w,h), 0, w/2], 
                            [0, max(w,h), h/2], 
                            [0, 0, 1]], dtype=np.float32)
    
    def objective(p_params):
        p1, p2 = p_params
        dist_coeffs = np.array([0, 0, p1, p2, 0])  # Only tangential
        
        try:
            undistorted = cv2.undistortPoints(
                line_points.reshape(-1, 1, 2),
                camera_matrix,
                dist_coeffs,
                P=camera_matrix
            ).reshape(-1, 2)
            
            return compute_line_straightness(undistorted)
        except:
            return float('inf')
    
    bounds = [(-0.1, 0.1), (-0.1, 0.1)]
    result = minimize(objective, initial_p, method='L-BFGS-B', bounds=bounds)
    
    p1, p2 = result.x
    return (p1 if abs(p1) > 1e-6 else 0, 
            p2 if abs(p2) > 1e-6 else 0)

def main():
    try:
        # Load image
        image_path = "./distorted_calibration.png"  # Change to your image path
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        print("=== Camera Calibration ===")
        print("1. Detecting lines...")
        
        # Calibrate radial distortion
        print("2. Calibrating radial distortion...")
        radial_coeffs = calibrate_radial_distortion(image, show_detection=True)
        k1, k2, _, _, k3 = radial_coeffs
        print(f"  Radial coefficients: k1={k1:.6f}, k2={k2:.6f}, k3={k3:.6f}")
        
        # Calibrate tangential distortion
        print("3. Calibrating tangential distortion...")
        p1, p2 = calibrate_tangential_distortion(image)
        print(f"  Tangential coefficients: p1={p1:.6f}, p2={p2:.6f}")
        
        # Combine all coefficients
        dist_coeffs = np.array([k1, k2, p1, p2, k3])
        
        # Create camera matrix
        h, w = image.shape[:2]
        camera_matrix = np.array([[w, 0, w/2], [0, h, h/2], [0, 0, 1]])
        
        # Apply correction
        print("4. Applying correction...")
        undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
        
        # Show results
        cv2.imshow("Original (Distorted)", image)
        cv2.imshow("Corrected", undistorted)
        print("5. Press any key to close windows...")
        cv2.waitKey(0)
        
        # Save results
        np.savez("calibration_results.npz",
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs)
        print("6. Calibration data saved to calibration_results.npz")
        
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()