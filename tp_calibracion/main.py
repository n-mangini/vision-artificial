import cv2
import sys
from full_distortion_model import FullDistortionModel

def main():
    path = "./aleph.png"

    image = cv2.imread(path)
    
    if image is None:
        print(f"Error: No se pudo cargar la imagen {path}")
        sys.exit(1)
    
    # Redimensionar la imagen al 50%
    scale_percent = 50  # porcentaje del tamaño original
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    image_resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    print(f"Imagen redimensionada de {image.shape[:2]} a {image_resized.shape[:2]}")
    
    # Usar la imagen redimensionada para el resto del proceso
    image = image_resized
    
    # Auto-calibrar desde la imagen
    print("Calibrando parámetros de distorsión...")
    k1, k2, k3, p1, p2 = FullDistortionModel.calibrate_from_image(image)
    
    print(f"Parámetros encontrados:")
    print(f"  Radial: k1={k1:.6f}, k2={k2:.6f}, k3={k3:.6f}")
    print(f"  Tangencial: p1={p1:.6f}, p2={p2:.6f}")
    
    # Corregir la imagen con los parámetros calculados
    print("Corrigiendo imagen...")
    undistorted_image = FullDistortionModel.undistort_image(image, k1, k2, k3, p1, p2)
    
    # Mostrar las imágenes
    cv2.imshow('Imagen Original', image)
    cv2.imshow('Imagen Corregida', undistorted_image)
    
    print("\nPresiona cualquier tecla para cerrar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()