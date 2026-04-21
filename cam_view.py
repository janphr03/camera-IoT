import os

from Backend.app import CameraApp


if __name__ == "__main__":
    CameraApp().run(open_browser=os.getenv("OPEN_BROWSER") == "1")
