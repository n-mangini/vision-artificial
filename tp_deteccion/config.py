import platform

LINUX_CAMERA_PATH = "/dev/video0"
WINDOWS_CAMERA_PATH = 0


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
    if os == "wsl" or os == "linux":
        return LINUX_CAMERA_PATH
    else:
        return WINDOWS_CAMERA_PATH
