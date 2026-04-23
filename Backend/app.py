from pathlib import Path
import base64
import json
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
        self.app.add_url_rule("/camera_events", "camera_events", self.camera_events)

        self.client = OpenAI() if os.getenv("OPENAI_API_KEY") else None
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            controls={"FrameDurationLimits": (33333, 33333)},
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(1)

        self.jpeg_quality = 58
        self.min_area = 650
        self.padding = 20
        self.previous_gray = None
        self.frame_index = 0
        self.motion_analysis_stride = 3
        self.last_detection_box = None
        self.last_motion_time = 0
        self.motion_hold_seconds = 0.55
        self.classification_interval = 5
        self.last_classification_time = 0
        self.current_label = "Noch kein Objekt"
        self.classification_in_progress = False
        self.label_lock = threading.Lock()
        self.label_event = threading.Event()
        self.default_color_mode = os.getenv("CAMERA_COLOR_MODE", "raw")

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
                self.label_event.set()
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

    def find_motion_box(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.previous_gray is None:
            self.previous_gray = gray
            return None

        frame_delta = cv2.absdiff(self.previous_gray, gray)
        _, thresh = cv2.threshold(frame_delta, 20, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        relevant_contours = [
            contour
            for contour in contours
            if cv2.contourArea(contour) >= self.min_area
        ]

        self.previous_gray = gray

        if not relevant_contours:
            return None

        frame_height, frame_width = frame_bgr.shape[:2]
        largest_contour = max(relevant_contours, key=cv2.contourArea)
        min_x, min_y, width, height = cv2.boundingRect(largest_contour)
        max_x = min_x + width
        max_y = min_y + height

        min_x = max(0, min_x - self.padding)
        min_y = max(0, min_y - self.padding)
        max_x = min(frame_width, max_x + self.padding)
        max_y = min(frame_height, max_y + self.padding)
        return min_x, min_y, max_x, max_y

    def draw_detection_box(self, frame_bgr, detection_box):
        min_x, min_y, max_x, max_y = detection_box
        overlay_color = (255, 190, 80)
        fill_color = (70, 40, 8)
        corner_length = max(18, min(44, (max_x - min_x) // 4, (max_y - min_y) // 4))

        tint = frame_bgr.copy()
        cv2.rectangle(tint, (min_x, min_y), (max_x, max_y), fill_color, -1)
        cv2.addWeighted(tint, 0.14, frame_bgr, 0.86, 0, frame_bgr)

        line_thickness = 3
        cv2.line(frame_bgr, (min_x, min_y), (min_x + corner_length, min_y), overlay_color, line_thickness)
        cv2.line(frame_bgr, (min_x, min_y), (min_x, min_y + corner_length), overlay_color, line_thickness)
        cv2.line(frame_bgr, (max_x, min_y), (max_x - corner_length, min_y), overlay_color, line_thickness)
        cv2.line(frame_bgr, (max_x, min_y), (max_x, min_y + corner_length), overlay_color, line_thickness)
        cv2.line(frame_bgr, (min_x, max_y), (min_x + corner_length, max_y), overlay_color, line_thickness)
        cv2.line(frame_bgr, (min_x, max_y), (min_x, max_y - corner_length), overlay_color, line_thickness)
        cv2.line(frame_bgr, (max_x, max_y), (max_x - corner_length, max_y), overlay_color, line_thickness)
        cv2.line(frame_bgr, (max_x, max_y), (max_x, max_y - corner_length), overlay_color, line_thickness)
        return overlay_color

    def draw_light_motion_overlay(self, frame_bgr):
        self.frame_index += 1
        should_analyze = self.frame_index % self.motion_analysis_stride == 0

        if should_analyze:
            detection_box = self.find_motion_box(frame_bgr)
            if detection_box is not None:
                self.last_detection_box = detection_box
                self.last_motion_time = time.time()
                min_x, min_y, max_x, max_y = detection_box
                roi = frame_bgr[min_y:max_y, min_x:max_x]
                self.maybe_start_classification(roi)

        if (
            self.last_detection_box is None
            or time.time() - self.last_motion_time > self.motion_hold_seconds
        ):
            return frame_bgr

        overlay_color = self.draw_detection_box(frame_bgr, self.last_detection_box)
        return frame_bgr

    def generate_stream(self, overlay=False, color_mode=None):
        self.previous_gray = None
        self.frame_index = 0
        self.last_detection_box = None
        self.last_motion_time = 0
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

    def generate_camera_events(self):
        last_sent_label = None

        while True:
            self.label_event.wait(timeout=20)
            self.label_event.clear()

            with self.label_lock:
                label = self.current_label

            if label == last_sent_label:
                yield ": keepalive\n\n"
                continue

            last_sent_label = label
            payload = {
                "label": label,
                "timestamp": time.strftime("%H:%M:%S"),
            }
            yield f"data: {json.dumps(payload)}\n\n"

    def camera_events(self):
        return Response(
            self.generate_camera_events(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    def open_browser(self):
        webbrowser.open(f"http://127.0.0.1:{self.port}")

    def run(self, open_browser=False):
        if open_browser:
            threading.Timer(1.0, self.open_browser).start()

        self.app.run(host=self.host, port=self.port, threaded=True)
