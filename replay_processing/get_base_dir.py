import os, sys

def get_base_dir():
    # When frozen by PyInstaller, use temp extraction folder
    if hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS

    # When running from source
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))