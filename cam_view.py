from flask import Flask, Response
from picamera2 import Picamera2
from openai import OpenAI
from dotenv import load_dotenv
import cv2
import time
import base64
import os


load_dotenv()


class CameraApp:
    def __init__(self, host="0.0.0.0", port=5000):
        self.host = host
        self.port = port

        self.app = Flask(__name__)
        self.app.add_url_rule("/", "index", self.index)
        self.app.add_url_rule("/video_feed", "video_feed", self.video_feed)

        self.client = OpenAI()

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
        self.current_label = "Kein Objekt erkannt"

    def index(self):
        return """
        <html>
            <head>
                <title>Raspberry Camera</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        text-align: center;
                        background: #111;
                        color: white;
                        margin-top: 30px;
                    }
                    img {
                        border: 3px solid white;
                        max-width: 90%;
                        height: auto;
                    }
                </style>
            </head>
            <body>
                <h1>Raspberry Live Camera</h1>
                <img src="/video_feed" />
            </body>
        </html>
        """

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
                                    "Antworte extrem kurz, nur 1 bis 3 Wörter, auf Deutsch. "
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

        except Exception as e:
            print(f"OpenAI-Fehler: {e}")
            return self.current_label

    def process_motion(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.previous_gray is None:
            self.previous_gray = gray
            return frame_bgr

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

        if motion_detected:
            min_x = max(0, min_x - self.padding)
            min_y = max(0, min_y - self.padding)
            max_x = min(frame_width, max_x + self.padding)
            max_y = min(frame_height, max_y + self.padding)

            cv2.rectangle(frame_bgr, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)

            current_time = time.time()
            if current_time - self.last_classification_time >= self.classification_interval:
                roi = frame_bgr[min_y:max_y, min_x:max_x]
                self.current_label = self.classify_object_with_openai(roi)
                self.last_classification_time = current_time
                print(f"[{time.strftime('%H:%M:%S')}] Erkannt: {self.current_label}")

            label_y = max(25, min_y - 10)
            cv2.putText(
                frame_bgr,
                self.current_label,
                (min_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        status_text = "Bewegung erkannt" if motion_detected else "Keine Bewegung"
        status_color = (0, 0, 255) if motion_detected else (0, 255, 0)

        cv2.putText(
            frame_bgr,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
        )

        self.previous_gray = gray
        return frame_bgr

    def generate_frames(self):
        while True:
            frame = self.picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_bgr = self.process_motion(frame_bgr)

            ok, buffer = cv2.imencode(".jpg", frame_bgr)
            if not ok:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    def video_feed(self):
        return Response(
            self.generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    def run(self):
        self.app.run(host=self.host, port=self.port, threaded=True)


if __name__ == "__main__":
    CameraApp().run()