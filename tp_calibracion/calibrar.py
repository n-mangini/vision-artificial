import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

def calibrar_camara(ruta_imagenes, patron_size=(7, 7), tamano_cuadro=1.0, visualizar=True):
    """
    Calibra una cámara usando un patrón de tablero de ajedrez.
    
    Args:
        ruta_imagenes: Ruta a las imágenes del patrón
        patron_size: Tamaño del patrón (esquinas interiores)
        tamano_cuadro: Tamaño de cada cuadro en unidades arbitrarias
        visualizar: Si se debe visualizar el proceso
    
    Returns:
        ret: Error de reproyección
        mtx: Matriz de la cámara
        dist: Coeficientes de distorsión [k1, k2, p1, p2, k3]
        rvecs: Vectores de rotación
        tvecs: Vectores de traslación
    """
    # Verificar si la ruta existe
    if not os.path.exists(ruta_imagenes):
        print(f"Error: La carpeta {ruta_imagenes} no existe.")
        print(f"Directorio actual: {os.getcwd()}")
        print("Contenido del directorio actual:")
        for item in os.listdir('.'):
            print(f"  - {item}")
        return None, None, None, None, None
    
    # Lista todas las imágenes
    imagenes = glob.glob(os.path.join(ruta_imagenes, '*.jpg'))
    imagenes += glob.glob(os.path.join(ruta_imagenes, '*.jpeg'))
    imagenes += glob.glob(os.path.join(ruta_imagenes, '*.png'))
    
    if len(imagenes) == 0:
        print(f"No se encontraron imágenes en {ruta_imagenes}")
        print(f"Contenido de la carpeta {ruta_imagenes}:")
        for item in os.listdir(ruta_imagenes):
            print(f"  - {item}")
        return None, None, None, None, None
    
    print(f"Se encontraron {len(imagenes)} imágenes para calibración:")
    for img in imagenes:
        print(f"  - {os.path.basename(img)}")
    
    # Preparar arrays para almacenar puntos de objeto y puntos de imagen
    objpoints = []  # Puntos 3D en espacio real
    imgpoints = []  # Puntos 2D en el plano de la imagen
    
    # Preparar puntos de objeto (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
    objp = np.zeros((patron_size[0] * patron_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:patron_size[0], 0:patron_size[1]].T.reshape(-1, 2) * tamano_cuadro
    
    # Arrays para almacenar las dimensiones de las imágenes y detecciones
    img_shapes = []
    detecciones = []
    
    # Probar con diferentes tamaños de patrón si es necesario
    patrones_alternativos = [(8, 6), (9, 6), (6, 9), (8, 8)]
    
    # Encontrar las esquinas del tablero en cada imagen
    for idx, fname in enumerate(imagenes):
        img = cv2.imread(fname)
        if img is None:
            print(f"No se pudo leer la imagen: {fname}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Encontrar esquinas del tablero
        ret, corners = cv2.findChessboardCorners(gray, patron_size, None)
        patron_actual = patron_size
        
        # Si no se encuentra, intentar con patrones alternativos
        if not ret:
            print(f"No se detectó el patrón {patron_size} en {os.path.basename(fname)}")
            
            # Mejora: Intentar con diferentes niveles de preprocesamiento
            # 1. Probar con ecualización de histograma
            gray_eq = cv2.equalizeHist(gray)
            ret, corners = cv2.findChessboardCorners(gray_eq, patron_size, None)
            
            if not ret:
                # 2. Probar con filtro gaussiano
                gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
                ret, corners = cv2.findChessboardCorners(gray_blur, patron_size, None)
            
            # 3. Si todavía no funciona, probar con diferentes tamaños de patrón
            if not ret:
                for alt_patron in patrones_alternativos:
                    print(f"  Probando con patrón {alt_patron}...")
                    ret, corners = cv2.findChessboardCorners(gray, alt_patron, None)
                    if ret:
                        print(f"  ¡Patrón {alt_patron} detectado correctamente!")
                        patron_actual = alt_patron
                        # Actualizar objp para este patrón específico
                        objp_alt = np.zeros((alt_patron[0] * alt_patron[1], 3), np.float32)
                        objp_alt[:, :2] = np.mgrid[0:alt_patron[0], 0:alt_patron[1]].T.reshape(-1, 2) * tamano_cuadro
                        break
        
        # Si se encuentran, agregar puntos de objeto y puntos refinados de imagen
        if ret:
            # Criterios para algoritmo de esquinas de subpíxel
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            
            # Refinar esquinas
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Usar el objp correspondiente al patrón detectado
            if patron_actual == patron_size:
                objpoints.append(objp)
            else:
                objp_alt = np.zeros((patron_actual[0] * patron_actual[1], 3), np.float32)
                objp_alt[:, :2] = np.mgrid[0:patron_actual[0], 0:patron_actual[1]].T.reshape(-1, 2) * tamano_cuadro
                objpoints.append(objp_alt)
                
            imgpoints.append(corners2)
            detecciones.append((idx, corners2))
            img_shapes.append(gray.shape[::-1])  # (ancho, alto)
            
            print(f"Patrón detectado en {os.path.basename(fname)}")
            
            # Dibujar y mostrar las esquinas
            if visualizar:
                img_copy = img.copy()
                cv2.drawChessboardCorners(img_copy, patron_actual, corners2, ret)
                plt.figure(figsize=(10, 7))
                plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
                plt.title(f'Detección en {os.path.basename(fname)}')
                plt.show()
    
    if len(objpoints) == 0:
        print("No se detectaron patrones en ninguna imagen")
        print("Sugerencias:")
        print("1. Asegúrate de que el tamaño del patrón es correcto (# de esquinas interiores)")
        print("2. Verifica que las imágenes contienen un tablero de ajedrez completo y visible")
        print("3. Prueba con patrones más pequeños o diferentes configuraciones")
        return None, None, None, None, None
    
    print(f"Se detectaron patrones en {len(objpoints)}/{len(imagenes)} imágenes")
    
    # Verificar si hay múltiples resoluciones
    if len(set(img_shapes)) > 1:
        print("ADVERTENCIA: Las imágenes tienen diferentes resoluciones:")
        for shape in set(img_shapes):
            count = img_shapes.count(shape)
            print(f"  - Resolución {shape[0]}x{shape[1]}: {count} imágenes")
            
        print("Agrupando imágenes por resolución para calibración separada...")
        
        # Agrupar por resolución
        calibraciones_por_resolucion = {}
        for res in set(img_shapes):
            indices = [i for i, shape in enumerate(img_shapes) if shape == res]
            if len(indices) >= 4:  # Necesitamos al menos 4 imágenes para una buena calibración
                res_objpoints = [objpoints[i] for i in indices]
                res_imgpoints = [imgpoints[i] for i in indices]
                
                print(f"Calibrando para resolución {res[0]}x{res[1]} con {len(indices)} imágenes...")
                
                try:
                    # Calibración para esta resolución específica
                    flags = 0
                    if len(indices) >= 10:
                        flags = cv2.CALIB_RATIONAL_MODEL
                    
                    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                        res_objpoints, res_imgpoints, res, None, None,
                        flags=flags
                    )
                    
                    # Almacenar resultados
                    calibraciones_por_resolucion[res] = {
                        'ret': ret,
                        'mtx': mtx,
                        'dist': dist,
                        'rvecs': rvecs,
                        'tvecs': tvecs,
                        'imagen_count': len(indices)
                    }
                    
                    print(f"Calibración exitosa para resolución {res[0]}x{res[1]}, error: {ret}")
                except Exception as e:
                    print(f"Error en calibración para resolución {res[0]}x{res[1]}: {e}")
        
        # Elegir la mejor calibración (la que tiene más imágenes o menor error)
        if calibraciones_por_resolucion:
            # Ordenar por número de imágenes (descendente) y luego por error (ascendente)
            mejor_res = max(calibraciones_por_resolucion.items(), 
                           key=lambda x: (x[1]['imagen_count'], -x[1]['ret']))
            
            res_elegida = mejor_res[0]
            calibracion_elegida = mejor_res[1]
            
            print(f"\nSe seleccionó la calibración para resolución {res_elegida[0]}x{res_elegida[1]}")
            print(f"Error de reproyección: {calibracion_elegida['ret']}")
            
            # Devolver la mejor calibración
            return (calibracion_elegida['ret'], calibracion_elegida['mtx'], 
                   calibracion_elegida['dist'], calibracion_elegida['rvecs'], 
                   calibracion_elegida['tvecs'])
        else:
            print("No se pudo realizar la calibración para ninguna resolución")
            return None, None, None, None, None
    else:
        # Todas las imágenes tienen la misma resolución
        img_size = img_shapes[0]
        print(f"Todas las imágenes tienen la misma resolución: {img_size[0]}x{img_size[1]}")
        
        print("Calibrando cámara con los patrones detectados...")
        try:
            # Usar flags específicos para evitar errores
            flags = 0
            # Añadir k3 solo si hay suficientes imágenes (más de 10 generalmente)
            if len(objpoints) >= 10:
                flags = cv2.CALIB_RATIONAL_MODEL
                print("Utilizando modelo racional (incluye k3)")
            else:
                print("Utilizando modelo estándar (no incluye k3)")
            
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, img_size, None, None,
                flags=flags
            )
            
            # Verificar que dist tiene el formato correcto
            if len(dist.ravel()) > 5:
                print(f"ADVERTENCIA: Se obtuvieron {len(dist.ravel())} coeficientes de distorsión, se esperaban 5.")
                print("Ajustando a los 5 primeros coeficientes...")
                dist = dist[:, :5]  # Tomar solo los primeros 5 coeficientes
            
            # Mostrar resultados
            print("\n=== RESULTADOS DE CALIBRACIÓN ===")
            print(f"Error de reproyección: {ret}")
            print("\nMatriz de la cámara:")
            print(mtx)
            print("\nCoeficientes de distorsión [k1, k2, p1, p2, k3]:")
            print(dist.ravel())
            
            if visualizar and len(detecciones) > 0:
                visualizar_cobertura(detecciones, img_size)
                visualizar_distorsion(mtx, dist, img_size)
            
            return ret, mtx, dist, rvecs, tvecs
        except Exception as e:
            print(f"Error durante la calibración: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None, None

def visualizar_cobertura(detecciones, img_size):
    """
    Visualiza la cobertura de detecciones en la imagen
    """
    plt.figure(figsize=(10, 7))
    
    for idx, corners in detecciones:
        # Extraer los puntos x, y
        x = corners[:, 0, 0]
        y = corners[:, 0, 1]
        
        # Dibujar los puntos con diferentes colores para cada imagen
        plt.scatter(x, y, alpha=0.5, label=f'Imagen {idx+1}')
    
    plt.xlim(0, img_size[0])
    plt.ylim(img_size[1], 0)  # Invertir eje y para que coincida con la orientación de la imagen
    plt.title('Cobertura de detecciones en todas las imágenes')
    plt.xlabel('X (píxeles)')
    plt.ylabel('Y (píxeles)')
    plt.grid(True)
    if len(detecciones) <= 20:  # Mostrar leyenda solo si no hay demasiadas imágenes
        plt.legend()
    plt.show()

def visualizar_distorsion(mtx, dist, img_size):
    """
    Visualiza el efecto de la distorsión en una cuadrícula uniforme
    """
    # Crear una cuadrícula regular
    x, y = np.meshgrid(np.linspace(0, img_size[0], 20), np.linspace(0, img_size[1], 20))
    
    # Aplanar la cuadrícula en puntos
    points = np.vstack((x.flatten(), y.flatten())).T.reshape(-1, 1, 2).astype(np.float32)
    
    # Aplicar la distorsión inversa para ver cómo se distorsionarían los puntos
    undistorted = cv2.undistortPoints(points, mtx, dist, P=mtx)
    
    # Mostrar la cuadrícula original y la distorsionada
    plt.figure(figsize=(15, 7))
    
    # Cuadrícula original
    plt.subplot(121)
    plt.scatter(points[:, 0, 0], points[:, 0, 1], color='blue', s=10)
    plt.title('Cuadrícula Original')
    plt.xlim(0, img_size[0])
    plt.ylim(img_size[1], 0)
    plt.grid(True)
    
    # Cuadrícula distorsionada
    plt.subplot(122)
    plt.scatter(undistorted[:, 0, 0], undistorted[:, 0, 1], color='red', s=10)
    plt.title('Efecto de la Distorsión')
    plt.xlim(0, img_size[0])
    plt.ylim(img_size[1], 0)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Asegurarse de que tenemos exactamente 5 coeficientes
    distCoeffs = dist.ravel()
    if len(distCoeffs) >= 5:
        k1, k2, p1, p2, k3 = distCoeffs[:5]
    else:
        k1, k2, p1, p2 = distCoeffs[:4]
        k3 = 0
    
    # Mostrar qué tipo de distorsión radial es predominante
    print("\nAnálisis de distorsión radial:")
    if abs(k1) > abs(k2) and abs(k1) > abs(k3):
        tipo = "barril" if k1 < 0 else "cojín"
        print(f"Predominante: k1 = {k1:.6f} (distorsión de {tipo})")
    elif abs(k2) > abs(k1) and abs(k2) > abs(k3):
        tipo = "complejo"
        print(f"Predominante: k2 = {k2:.6f} (distorsión compleja)")
    elif abs(k3) > abs(k1) and abs(k3) > abs(k2):
        tipo = "de orden superior"
        print(f"Predominante: k3 = {k3:.6f} (distorsión {tipo})")
    else:
        print("No hay un coeficiente de distorsión radial claramente predominante.")
    
    print("\nAnálisis de distorsión tangencial:")
    if abs(p1) > 0.0001 or abs(p2) > 0.0001:
        print(f"Distorsión tangencial presente: p1 = {p1:.6f}, p2 = {p2:.6f}")
    else:
        print("Distorsión tangencial mínima.")

def aplicar_calibracion(ruta_imagenes, ruta_salida, mtx, dist):
    """
    Aplica la calibración a las imágenes y las guarda en la ruta de salida
    
    Args:
        ruta_imagenes: Ruta a las imágenes a calibrar
        ruta_salida: Ruta donde guardar las imágenes calibradas
        mtx: Matriz de la cámara
        dist: Coeficientes de distorsión
    """
    # Verificar si la ruta existe
    if not os.path.exists(ruta_imagenes):
        print(f"Error: La carpeta {ruta_imagenes} no existe.")
        return
    
    # Crear directorio de salida si no existe
    if not os.path.exists(ruta_salida):
        os.makedirs(ruta_salida)
        print(f"Se creó el directorio de salida: {ruta_salida}")
    
    # Lista todas las imágenes
    imagenes = glob.glob(os.path.join(ruta_imagenes, '*.jpg'))
    imagenes += glob.glob(os.path.join(ruta_imagenes, '*.jpeg'))
    imagenes += glob.glob(os.path.join(ruta_imagenes, '*.png'))
    
    if len(imagenes) == 0:
        print(f"No se encontraron imágenes en {ruta_imagenes}")
        print(f"Contenido de la carpeta {ruta_imagenes}:")
        for item in os.listdir(ruta_imagenes):
            print(f"  - {item}")
        return
    
    print(f"\nAplicando calibración a {len(imagenes)} imágenes...")
    
    for fname in imagenes:
        img = cv2.imread(fname)
        if img is None:
            print(f"No se pudo leer la imagen: {fname}")
            continue
            
        h, w = img.shape[:2]
        
        # Obtener la nueva matriz de la cámara
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        
        # Rectificar la imagen (eliminar distorsión)
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        
        # Recortar la región de interés
        x, y, w, h = roi
        if roi[2] > 0 and roi[3] > 0:  # Asegurarse de que roi es válido
            dst = dst[y:y+h, x:x+w]
        
        # Guardar la imagen resultante
        output_fname = os.path.join(ruta_salida, os.path.basename(fname))
        cv2.imwrite(output_fname, dst)
        print(f"Imagen calibrada guardada: {output_fname}")
        
        # Mostrar la comparación original vs calibrada
        if len(imagenes) <= 5:  # Limitar la visualización a pocas imágenes
            plt.figure(figsize=(15, 7))
            plt.subplot(121)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title('Original')
            plt.subplot(122)
            plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
            plt.title('Calibrada')
            plt.tight_layout()
            plt.show()
    
    print(f"Imágenes calibradas guardadas en {ruta_salida}")

def main():
    print("\n===== CALIBRADOR DE CÁMARA INTERACTIVO =====")
    print("Versión: 3.0 - Optimizado para múltiples resoluciones")
    print("Fecha: Mayo 2025")
    print("="*45)
    
    # Obtener la ruta del script actual
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir == '':  # Si es una cadena vacía, usar el directorio actual
        script_dir = os.getcwd()
    print(f"Directorio del script: {script_dir}")
    
    # Definir rutas relativas al directorio del script
    ruta_calibracion = os.path.join(script_dir, 'img')
    ruta_a_calibrar = os.path.join(script_dir, 'to_calibrate')
    ruta_salida = os.path.join(script_dir, 'calibrated')
    
    # Verificar la existencia de las carpetas
    print("\nVerificando directorios:")
    for ruta, nombre in [(ruta_calibracion, "Imágenes de calibración"), 
                         (ruta_a_calibrar, "Imágenes a calibrar")]:
        if os.path.exists(ruta):
            print(f"✓ {nombre}: {ruta} (Existe)")
        else:
            print(f"✗ {nombre}: {ruta} (No existe)")
    
    # Configuración del patrón de calibración (tablero de ajedrez)
    # Basado en los resultados anteriores, parece que el patrón es 7x7
    patron_size = (7, 7)  # Número de esquinas interiores en el patrón (ancho, alto)
    
    print(f"\nBuscando imágenes de calibración en: {ruta_calibracion}")
    print(f"Patrón de calibración inicial: tablero de ajedrez de {patron_size[0]}x{patron_size[1]} esquinas interiores")
    print("Se probarán automáticamente diferentes patrones si el inicial no se detecta.")
    
    # Realizar la calibración
    ret, mtx, dist, rvecs, tvecs = calibrar_camara(ruta_calibracion, patron_size, visualizar=True)
    
    if mtx is not None and dist is not None:
        # Aplicar la calibración a las nuevas imágenes
        aplicar_calibracion(ruta_a_calibrar, ruta_salida, mtx, dist)
        
        # Guardar los parámetros en un archivo
        print("\nGuardando parámetros de calibración...")
        parametros_path = os.path.join(script_dir, 'calibracion_parametros.npz')
        np.savez(
            parametros_path, 
            mtx=mtx, 
            dist=dist, 
            rvecs=rvecs, 
            tvecs=tvecs, 
            patron_size=patron_size
        )
        print(f"Parámetros guardados en: {parametros_path}")
        
        # También guardar en formato YAML/XML para compatibilidad con otras herramientas
        fs = cv2.FileStorage(os.path.join(script_dir, 'calibracion_parametros.xml'), cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", mtx)
        fs.write("distortion_coefficients", dist)
        fs.release()
        print(f"Parámetros guardados en formato XML: {os.path.join(script_dir, 'calibracion_parametros.xml')}")
    else:
        print("La calibración falló. No se pudieron obtener los parámetros de la cámara.")
        print("Intenta con un patrón de calibración diferente o mejora la calidad de las imágenes.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError inesperado: {e}")
        import traceback
        traceback.print_exc()