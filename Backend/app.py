from pathlib import Path
import base64
import json
import threading
import time

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
        self.app.add_url_rule("/api/cameras", "camera_options", self.camera_options)
        self.app.add_url_rule("/api/status", "status", self.status)
        self.app.add_url_rule(
            "/api/cameras/<camera_id>",
            "update_camera",
            self.update_camera,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/camera_feed/<camera_id>", "camera_feed", self.camera_feed
        )

        self.client = OpenAI()
        self.camera_config_path = PROJECT_ROOT / "Backend" / "cameras.json"
        self.cameras = self.load_camera_config()

        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2)

        self.previous_gray = None
        self.min_area = 1200
        self.padding = 35
        self.last_classification_time = 0
        self.classification_interval = 5
        self.current_label = "Noch keine Klassifikation"
        self.last_motion_status = "Initialisiere"
        self.last_detection_box = None
        self.latest_raw_frame = None
        self.latest_overlay_frame = None
        self.latest_frame_timestamp = 0.0
        self.frame_lock = threading.Lock()

        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()

    def load_camera_config(self):
        default_config = [
            {
                "id": "pi-main",
                "name": "Eingang Nord",
                "location": "Haupteingang",
                "resolution": "640 x 480",
                "state": "online",
                "description": "Raspberry-Pi-Kameramodul",
            }
        ]

        if not self.camera_config_path.exists():
            self.camera_config_path.write_text(
                json.dumps(default_config, indent=2), encoding="utf-8"
            )
            return default_config

        try:
            return json.loads(self.camera_config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self.camera_config_path.write_text(
                json.dumps(default_config, indent=2), encoding="utf-8"
            )
            return default_config

    def save_camera_config(self):
        self.camera_config_path.write_text(
            json.dumps(self.cameras, indent=2), encoding="utf-8"
        )

    def index(self):
        return render_template("index.html")

    def camera_options(self):
        return jsonify({"cameras": self.cameras})

    def status(self):
        return jsonify(
            {
                "motion": self.last_motion_status,
                "label": self.current_label,
                "classification_interval": self.classification_interval,
                "last_frame_timestamp": self.latest_frame_timestamp,
                "cameras": [
                    {
                        "id": camera["id"],
                        "motion": self.last_motion_status,
                        "label": self.current_label,
                        "state": camera["state"],
                    }
                    for camera in self.cameras
                ],
            }
        )

    def update_camera(self, camera_id):
        payload = request.get_json(silent=True) or {}
        new_name = (payload.get("name") or "").strip()

        if not new_name:
            return jsonify({"error": "Name darf nicht leer sein."}), 400

        for camera in self.cameras:
            if camera["id"] == camera_id:
                camera["name"] = new_name[:40]
                self.save_camera_config()
                return jsonify({"camera": camera})

        return jsonify({"error": "Kamera nicht gefunden."}), 404

    def classify_object_with_openai(self, roi):
        ok, buffer = cv2.imencode(".jpg", roi)
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
                                    "Beispiele: Mensch, Hund, Katze, Auto, Vogel, Unbekanntes Objekt."
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
            print(f"OpenAI-Fehler: {exc}")
            return self.current_label

    def analyze_frame(self, frame_bgr):
        overlay_frame = frame_bgr.copy()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.previous_gray is None:
            self.previous_gray = gray
            self.last_motion_status = "System bereit"
            return overlay_frame

        frame_delta = cv2.absdiff(self.previous_gray, gray)
        _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=3)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        motion_detected = False
        frame_height, frame_width = frame_bgr.shape[:2]
        min_x = frame_width
        min_y = frame_height
        max_x = 0
        max_y = 0

        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)
            motion_detected = True

        self.last_detection_box = None
        if motion_detected:
            min_x = max(0, min_x - self.padding)
            min_y = max(0, min_y - self.padding)
            max_x = min(frame_width, max_x + self.padding)
            max_y = min(frame_height, max_y + self.padding)
            self.last_detection_box = (min_x, min_y, max_x, max_y)

            current_time = time.time()
            if current_time - self.last_classification_time >= self.classification_interval:
                roi = frame_bgr[min_y:max_y, min_x:max_x]
                self.current_label = self.classify_object_with_openai(roi)
                self.last_classification_time = current_time
                print(f"[{time.strftime('%H:%M:%S')}] Erkannt: {self.current_label}")

            cv2.rectangle(overlay_frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
            label_y = max(25, min_y - 10)
            cv2.putText(
                overlay_frame,
                self.current_label,
                (min_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            self.last_motion_status = "Bewegung erkannt"
        else:
            self.last_motion_status = "Keine Bewegung"

        status_color = (0, 0, 255) if motion_detected else (0, 255, 0)
        cv2.putText(
            overlay_frame,
            self.last_motion_status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
        )

        self.previous_gray = gray
        return overlay_frame

    def encode_frame(self, frame_bgr):
        ok, buffer = cv2.imencode(".jpg", frame_bgr)
        return buffer.tobytes() if ok else None

    def capture_loop(self):
        while True:
            frame = self.picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            overlay_frame = self.analyze_frame(frame_bgr)
            raw_bytes = self.encode_frame(frame_bgr)
            overlay_bytes = self.encode_frame(overlay_frame)

            if raw_bytes and overlay_bytes:
                with self.frame_lock:
                    self.latest_raw_frame = raw_bytes
                    self.latest_overlay_frame = overlay_bytes
                    self.latest_frame_timestamp = time.time()

    def generate_stream(self, overlay=False):
        while True:
            with self.frame_lock:
                frame_bytes = (
                    self.latest_overlay_frame if overlay else self.latest_raw_frame
                )

            if frame_bytes is None:
                time.sleep(0.05)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            time.sleep(0.05)

    def camera_feed(self, camera_id):
        if not any(camera["id"] == camera_id for camera in self.cameras):
            return jsonify({"error": "Kamera nicht gefunden."}), 404

        overlay = request.args.get("overlay", "0") == "1"
        return Response(
            self.generate_stream(overlay=overlay),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def run(self):
        self.app.run(host=self.host, port=self.port, threaded=True)
