# RaspCamIoT

Dieses Projekt startet einen Flask-Server auf dem Raspberry Pi und stellt eine kleine Ueberwachungskamera-App im Browser bereit. Die Anwendung bietet jetzt ein Home-Menue, eine reine Live-Preview ohne aktive Erkennung, eine explizite Kamera-Auswahl und einen separaten Ueberwachungsmodus mit Bewegungs- und Objekterkennung per OpenAI API.

## Projektstruktur

```text
RaspCamIoT/
├── Backend/
│   └── app.py
├── Frontend/
│   └── templates/
│       └── index.html
├── cam_view.py
├── requirements.txt
└── .env
```

## Voraussetzungen

- Python 3.10 oder neuer
- Raspberry Pi mit funktionierender Kamera
- Raspberry-Pi-System mit installierter `picamera2`-Unterstützung
- OpenAI API Key

## Projekt neu einrichten

### 1. Projekt laden

Repository klonen oder Projektordner kopieren:

```bash
git clone <REPOSITORY-URL>
cd RaspCamIoT
```

### 2. Virtuelle Umgebung erstellen

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Python-Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

Hinweis: `picamera2` läuft in der Regel auf Raspberry Pi OS. Falls die Installation auf einem normalen Laptop oder Mac fehlschlägt, ist das erwartbar. Das Projekt ist für den Raspberry Pi gedacht.

### 4. `.env` im Projektroot anlegen

Die `.env` muss direkt im Hauptordner `RaspCamIoT` liegen, also auf derselben Ebene wie `cam_view.py`.

Beispielinhalt:

```env
OPENAI_API_KEY=dein_api_key_hier
```

### 5. Anwendung starten

```bash
python3 cam_view.py
```

Standardmäßig startet der Server auf Port `5000`.

## Im Browser oeffnen

Im lokalen Netzwerk den Raspberry Pi im Browser öffnen:

```text
http://<IP-DES-RASPBERRY-PI>:5000
```

Beispiel:

```text
http://192.168.178.50:5000
```

## Aktueller Ablauf in der App

1. Startseite mit Home-Dashboard oeffnen.
2. Verfuegbare Kamera auswaehlen.
3. Zunaechst nur die Preview ansehen.
4. Danach bewusst `Ueberwachung starten` aktivieren.
5. Im aktiven Modus wird Bewegung markiert und ein Objektlabel eingeblendet.

## Typischer Ablauf für dich oder Kolleg*innen

1. Projekt neu laden oder pullen.
2. In den Ordner `RaspCamIoT` wechseln.
3. Virtuelle Umgebung aktivieren.
4. Falls nötig `pip install -r requirements.txt` ausführen.
5. Prüfen, ob die `.env` im Projektroot vorhanden ist.
6. Raspberry-Kamera anschließen und aktivieren.
7. `python3 cam_view.py` starten.
8. Im Browser `http://<raspberry-ip>:5000` öffnen.

## Hinweise

- Das Frontend liegt in `Frontend/templates/index.html`.
- Das Backend liegt in `Backend/app.py`.
- Die `.env` wird automatisch aus dem Projektroot geladen.
- Wenn keine Kamera erkannt wird, liegt das meist an der Raspberry-Pi-Konfiguration oder an fehlender `picamera2`-Unterstützung.
