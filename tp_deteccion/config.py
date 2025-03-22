import platform
# config.py - Configuration file for capture path

# RTMP stream URL
WSL_CAMERA = "/dev/video0"
WINDOWS_CAMERA = 0

def detect_os():
    system = platform.system()
    if system == "Windows":
        return "windows"
    elif system == "Darwin":
        return "mac"
    elif "microsoft" in platform.uname().release.lower():  # Detect WSL
        return "wsl"
    else:
        return "linux"
    
def choose_camera_by_OS():
    os = detect_os()
    if os == "windows":
        return WINDOWS_CAMERA
    else:
        return WSL_CAMERA