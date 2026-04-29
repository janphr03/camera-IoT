from pathlib import Path
import base64
import copy
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
STATE_FILE = PROJECT_ROOT / "Backend" / "app_state.json"
MAX_DETECTIONS = 80
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
        self.app.add_url_rule(
            "/app_state", "app_state", self.app_state, methods=["GET", "POST"]
        )

        self.state_file = STATE_FILE
        self.state_lock = threading.Lock()
        self.state = self.load_state()
        latest_detection = self.state.get("latest_detection", {})

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
        self.jpeg_encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        self.classification_jpeg_quality = 60
        self.classification_encode_params = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            self.classification_jpeg_quality,
        ]
        self.classification_max_side = 320
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
        self.current_label = latest_detection.get("label") or "Noch kein Objekt"
        self.current_detection_timestamp = latest_detection.get("timestamp")
        self.classification_in_progress = False
        self.label_lock = threading.Lock()
        self.label_event = threading.Event()

    def default_state(self):
        return {
            "settings": {
                "overlay": False,
                "fit": False,
                "mirror": False,
            },
            "latest_detection": {
                "label": "Noch kein Objekt",
                "timestamp": None,
            },
            "detections": [],
            "event_counter": 0,
            "events_today": 0,
            "events_today_date": time.strftime("%Y-%m-%d"),
        }

    def normalize_state(self, data):
        state = self.default_state()
        if not isinstance(data, dict):
            return state

        settings = data.get("settings")
        if isinstance(settings, dict):
            for key in state["settings"]:
                state["settings"][key] = bool(settings.get(key, state["settings"][key]))

        detections = data.get("detections")
        if isinstance(detections, list):
            clean_detections = []
            for detection in detections[-MAX_DETECTIONS:]:
                normalized_detection = self.normalize_detection(detection)
                if normalized_detection is not None:
                    clean_detections.append(normalized_detection)
            state["detections"] = clean_detections

        event_counter = data.get("event_counter")
        if isinstance(event_counter, int) and event_counter >= 0:
            state["event_counter"] = max(event_counter, len(state["detections"]))
        else:
            state["event_counter"] = len(state["detections"])

        today = time.strftime("%Y-%m-%d")
        events_today = data.get("events_today")
        events_today_date = data.get("events_today_date")
        if (
            isinstance(events_today, int)
            and events_today >= 0
            and events_today_date == today
        ):
            state["events_today"] = events_today
            state["events_today_date"] = today
        else:
            state["events_today"] = sum(
                1
                for detection in state["detections"]
                if (detection.get("timestamp") or "").startswith(today)
            )
            state["events_today_date"] = today

        latest_detection = self.normalize_detection(data.get("latest_detection"))
        if latest_detection is not None:
            state["latest_detection"] = latest_detection
        elif state["detections"]:
            state["latest_detection"] = state["detections"][-1]

        return state

    def normalize_detection(self, detection):
        if not isinstance(detection, dict):
            return None

        label = detection.get("label")
        if not isinstance(label, str) or not label.strip():
            return None

        timestamp = detection.get("timestamp")
        timestamp = timestamp.strip() if isinstance(timestamp, str) else None

        normalized = {
            "id": detection.get("id") if isinstance(detection.get("id"), int) else 0,
            "label": label.strip()[:80],
            "timestamp": timestamp,
            "camera_id": detection.get("camera_id") or "pi-main",
        }

        box = detection.get("box")
        if isinstance(box, dict):
            try:
                x = int(box.get("x", 0))
                y = int(box.get("y", 0))
                width = max(0, int(box.get("width", 0)))
                height = max(0, int(box.get("height", 0)))
            except (TypeError, ValueError):
                x = y = width = height = 0
            normalized["box"] = {
                "x": max(0, x),
                "y": max(0, y),
                "width": width,
                "height": height,
                "area": width * height,
            }

        return normalized

    def summarize_state(self, state):
        detections = state.get("detections", [])
        label_counts = {}

        for detection in detections:
            label = detection.get("label") or "Unbekannt"
            label_counts[label] = label_counts.get(label, 0) + 1

        top_label = None
        top_label_count = 0
        if label_counts:
            top_label, top_label_count = max(
                label_counts.items(), key=lambda item: (item[1], item[0])
            )

        latest_detection = state.get("latest_detection") or {}
        return {
            "total_events": state.get("event_counter", len(detections)),
            "stored_events": len(detections),
            "events_today": state.get("events_today", 0),
            "unique_labels": len(label_counts),
            "top_label": top_label or "Keine Daten",
            "top_label_count": top_label_count,
            "last_seen_at": latest_detection.get("timestamp"),
            "history_limit": MAX_DETECTIONS,
        }

    def state_response_unlocked(self):
        state = copy.deepcopy(self.state)
        state["analytics"] = self.summarize_state(state)
        state["recent_detections"] = state.get("detections", [])[-20:][::-1]
        return state

    def write_state_unlocked(self, state):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self.state_file.with_suffix(".json.tmp")
        with temp_file.open("w", encoding="utf-8") as file:
            json.dump(state, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temp_file.replace(self.state_file)

    def load_state(self):
        if not self.state_file.exists():
            state = self.default_state()
            self.write_state_unlocked(state)
            return state

        try:
            with self.state_file.open("r", encoding="utf-8") as file:
                return self.normalize_state(json.load(file))
        except (OSError, json.JSONDecodeError):
            state = self.default_state()
            self.write_state_unlocked(state)
            return state

    def app_state(self):
        if request.method == "GET":
            with self.state_lock:
                state = self.state_response_unlocked()
            return jsonify(state)

        payload = request.get_json(silent=True) or {}
        with self.state_lock:
            state = self.normalize_state(self.state)
            settings = payload.get("settings")
            if isinstance(settings, dict):
                for key in state["settings"]:
                    if key in settings:
                        state["settings"][key] = bool(settings[key])

            self.state = state
            self.write_state_unlocked(self.state)
            response_state = self.state_response_unlocked()

        return jsonify(response_state)

    def persist_detection(self, label, detection_box=None):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        detection = {
            "id": int(time.time() * 1000),
            "label": label,
            "timestamp": timestamp,
            "camera_id": "pi-main",
        }
        if detection_box is not None:
            min_x, min_y, max_x, max_y = detection_box
            width = max(0, max_x - min_x)
            height = max(0, max_y - min_y)
            detection["box"] = {
                "x": min_x,
                "y": min_y,
                "width": width,
                "height": height,
                "area": width * height,
            }

        with self.state_lock:
            state = self.normalize_state(self.state)
            today = time.strftime("%Y-%m-%d")
            if state.get("events_today_date") != today:
                state["events_today"] = 0
                state["events_today_date"] = today
            state["latest_detection"] = detection
            state["detections"].append(detection)
            state["detections"] = state["detections"][-MAX_DETECTIONS:]
            state["event_counter"] += 1
            state["events_today"] += 1
            self.state = state
            self.write_state_unlocked(self.state)
            response_state = self.state_response_unlocked()

        return detection, response_state

    def index(self):
        return render_template("index.html")

    def encode_frame(self, frame_bgr):
        ok, buffer = cv2.imencode(".jpg", frame_bgr, self.jpeg_encode_params)
        return buffer.tobytes() if ok else None

    def classify_object_with_openai(self, roi):
        if self.client is None:
            return "OpenAI API-Key fehlt"

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        height, width = roi_rgb.shape[:2]
        largest_side = max(height, width)
        if largest_side > self.classification_max_side:
            scale = self.classification_max_side / largest_side
            roi_rgb = cv2.resize(
                roi_rgb,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )

        ok, buffer = cv2.imencode(".jpg", roi_rgb, self.classification_encode_params)
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

    def classify_object_in_background(self, roi, detection_box):
        try:
            label = self.classify_object_with_openai(roi)
            detection, state = self.persist_detection(label, detection_box)
            with self.label_lock:
                self.current_label = detection["label"]
                self.current_detection_timestamp = detection["timestamp"]
                self.current_detection_payload = {
                    "detection": detection,
                    "analytics": state["analytics"],
                    "recent_detections": state["recent_detections"],
                }
                self.label_event.set()
            print(f"[{time.strftime('%H:%M:%S')}] Bewegung erkannt: {label}")
        finally:
            self.classification_in_progress = False

    def maybe_start_classification(self, roi, detection_box):
        current_time = time.time()
        if self.classification_in_progress:
            return

        if current_time - self.last_classification_time < self.classification_interval:
            return

        self.last_classification_time = current_time
        self.classification_in_progress = True
        thread = threading.Thread(
            target=self.classify_object_in_background,
            args=(roi.copy(), detection_box),
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
        self.previous_gray = gray

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        largest_contour = None
        largest_area = self.min_area
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= largest_area:
                largest_area = area
                largest_contour = contour

        if largest_contour is None:
            return None

        frame_height, frame_width = frame_bgr.shape[:2]
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

        roi = frame_bgr[min_y:max_y, min_x:max_x]
        if roi.size:
            tint = roi.copy()
            tint[:] = fill_color
            cv2.addWeighted(tint, 0.14, roi, 0.86, 0, roi)

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
                self.maybe_start_classification(roi, detection_box)

        if (
            self.last_detection_box is None
            or time.time() - self.last_motion_time > self.motion_hold_seconds
        ):
            return frame_bgr

        self.draw_detection_box(frame_bgr, self.last_detection_box)
        return frame_bgr

    def generate_stream(self, overlay=False):
        self.previous_gray = None
        self.frame_index = 0
        self.last_detection_box = None
        self.last_motion_time = 0

        while True:
            frame = self.picam2.capture_array()
            frame_bgr = frame

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
        return Response(
            self.generate_stream(overlay=overlay),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    def generate_camera_events(self):
        last_sent_detection = None

        while True:
            self.label_event.wait(timeout=20)
            self.label_event.clear()

            with self.label_lock:
                label = self.current_label
                timestamp = self.current_detection_timestamp or time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                detection_payload = getattr(self, "current_detection_payload", None)

            current_detection = (label, timestamp)
            if current_detection == last_sent_detection:
                yield ": keepalive\n\n"
                continue

            last_sent_detection = current_detection
            payload = detection_payload or {
                "detection": {"label": label, "timestamp": timestamp},
                "analytics": {},
                "recent_detections": [],
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
