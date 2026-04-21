from pathlib import Path
import base64
import os
import threading
import time
import webbrowser

import cv2
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from openai import OpenAI
from picamera2 import Picamera2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


class CameraApp:
    def __init__(self, host="0.0.0.0", port=5000):
        self.host = host
        self.port = port

        template_dir = PROJECT_ROOT / "Frontend" / "templates"
        self.app = Flask(__name__, template_folder=str(template_dir))
        self.app.add_url_rule("/", "index", self.index)
        self.app.add_url_rule(
            "/camera_feed/<camera_id>", "camera_feed", self.camera_feed
        )

        self.client = OpenAI() if os.getenv("OPENAI_API_KEY") else None
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            controls={"FrameDurationLimits": (33333, 33333)},
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(1)

        self.jpeg_quality = 68
        self.min_area = 1400
        self.padding = 25
        self.previous_gray = None
        self.classification_interval = 5
        self.last_classification_time = 0
        self.current_label = "Noch kein Objekt"
        self.classification_in_progress = False
        self.label_lock = threading.Lock()
        self.default_color_mode = os.getenv("CAMERA_COLOR_MODE", "swap_rb")

    def index(self):
        return render_template("index.html")

    def encode_frame(self, frame_bgr):
        ok, buffer = cv2.imencode(
            ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        return buffer.tobytes() if ok else None

    def normalize_frame_for_opencv(self, frame, color_mode):
        if color_mode == "raw":
            return frame

        if color_mode == "rgb_to_bgr":
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if color_mode == "swap_rb":
            return frame[:, :, [2, 1, 0]]

        return frame[:, :, [2, 1, 0]]

    def classify_object_with_openai(self, roi):
        if self.client is None:
            return "OpenAI API-Key fehlt"

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        ok, buffer = cv2.imencode(".jpg", roi_rgb)
        if not ok:
            return self.current_label

        image_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{image_base64}"

        try:
            response = self.client.responses.create(
                model="gpt-4.1-mini",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "Nenne nur das Hauptobjekt in diesem Bildausschnitt. "
                                    "Antworte extrem kurz, nur 1 bis 3 Woerter, auf Deutsch. "
                                    "Beispiele: Mensch, Hund, Katze, Auto, Vogel, Auto, Paket, Unbekanntes Objekt."
                                ),
                            },
                            {
                                "type": "input_image",
                                "image_url": data_url,
                            },
                        ],
                    }
                ],
            )
            label = response.output_text.strip()
            return label if label else "Unbekanntes Objekt"
        except Exception as exc:
            print(f"[{time.strftime('%H:%M:%S')}] OpenAI-Fehler: {exc}")
            return self.current_label

    def classify_object_in_background(self, roi):
        try:
            label = self.classify_object_with_openai(roi)
            with self.label_lock:
                self.current_label = label
            print(f"[{time.strftime('%H:%M:%S')}] Bewegung erkannt: {label}")
        finally:
            self.classification_in_progress = False

    def maybe_start_classification(self, roi):
        current_time = time.time()
        if self.classification_in_progress:
            return

        if current_time - self.last_classification_time < self.classification_interval:
            return

        self.last_classification_time = current_time
        self.classification_in_progress = True
        thread = threading.Thread(
            target=self.classify_object_in_background,
            args=(roi.copy(),),
            daemon=True,
        )
        thread.start()

    def draw_light_motion_overlay(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.previous_gray is None:
            self.previous_gray = gray
            return frame_bgr

        frame_delta = cv2.absdiff(self.previous_gray, gray)
        _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        frame_height, frame_width = frame_bgr.shape[:2]
        min_x = frame_width
        min_y = frame_height
        max_x = 0
        max_y = 0
        motion_detected = False

        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + width)
            max_y = max(max_y, y + height)
            motion_detected = True

        self.previous_gray = gray

        if not motion_detected:
            return frame_bgr

        min_x = max(0, min_x - self.padding)
        min_y = max(0, min_y - self.padding)
        max_x = min(frame_width, max_x + self.padding)
        max_y = min(frame_height, max_y + self.padding)

        roi = frame_bgr[min_y:max_y, min_x:max_x]
        self.maybe_start_classification(roi)

        overlay_color = (40, 120, 255)
        cv2.rectangle(frame_bgr, (min_x, min_y), (max_x, max_y), overlay_color, 2)
        with self.label_lock:
            label = self.current_label

        label_origin_y = max(28, min_y - 10)
        cv2.putText(
            frame_bgr,
            label,
            (min_x, label_origin_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            overlay_color,
            2,
        )
        return frame_bgr

    def generate_stream(self, overlay=False, color_mode=None):
        self.previous_gray = None
        color_mode = color_mode or self.default_color_mode

        while True:
            frame = self.picam2.capture_array()
            frame_bgr = self.normalize_frame_for_opencv(frame, color_mode)

            if overlay:
                frame_bgr = self.draw_light_motion_overlay(frame_bgr)

            frame_bytes = self.encode_frame(frame_bgr)
            if frame_bytes is None:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    def camera_feed(self, camera_id):
        if camera_id != "pi-main":
            return jsonify({"error": "Kamera nicht gefunden."}), 404

        overlay = request.args.get("overlay", "0") == "1"
        color_mode = request.args.get("color", self.default_color_mode)
        return Response(
            self.generate_stream(overlay=overlay, color_mode=color_mode),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def open_browser(self):
        webbrowser.open(f"http://127.0.0.1:{self.port}")

    def run(self, open_browser=False):
        if open_browser:
            threading.Timer(1.0, self.open_browser).start()

        self.app.run(host=self.host, port=self.port, threaded=True)
