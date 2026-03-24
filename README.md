# Aag — AI Fire Detection System

**Real-time intelligent fire monitoring using computer vision and deep learning.**

---

## Overview

**Aag** is a full-stack AI-powered fire detection system that uses your webcam to detect fire in real time. It combines a **FastAPI Python backend** for image analysis with a **pure-HTML/React frontend** for live monitoring. No database, no complex setup — just instant, stateless fire detection.

The system uses **HSV (Hue-Saturation-Value) color-space analysis** to identify fire signatures in live video frames, with a brightness validation pass to reduce false positives. It is also **YOLO-ready**, meaning you can plug in a custom-trained YOLO model for even higher accuracy.

---

## Features

- **Live Webcam Feed** — Captures real-time video frames from your device's camera
- **HSV Fire Detection** — Detects red, orange, and yellow fire signatures using multi-range color thresholding and morphological cleanup
- **YOLO-Ready Architecture** — Easily switch to a custom YOLO model (e.g., `fire_model.pt`) for improved detection
- **Bounding Box Overlay** — Draws a red bounding box on the detected fire region in the live canvas feed
- **Confidence Scoring** — Calculates and displays a real-time confidence percentage (scaled by fire area and brightness)
- **Multi-Layer Alert System**:
  - 🔔 Audio alarm (`pieper.mp3`) with Web Audio API fallback
  - 🔴 Full-screen red flash overlay
  - 📳 Shaking card animation
  - 🚨 Blinking `FIRE DETECTED — EVACUATE IMMEDIATELY` banner
- **Live Event Log** — Timestamped log of all detection events, camera states, and system messages
- **System Status Dashboard** — Real-time indicators for backend API, camera, alarm, and detection engine status
- **Demo Mode** — If the backend is offline, the frontend runs in probabilistic demo mode so you can still explore the UI
- **Zero-Overhead** — Stateless, no database, no session storage. Sub-200ms response time.

---

## Project Structure

```
Aag/
├── backend/
│   ├── main.py              # FastAPI backend — fire detection API
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── index.html           # React app (CDN-loaded) — full dashboard UI
│   ├── serve.py             # Simple Python HTTP server for local dev
│   └── public/
│       └── audio/
│           └── pieper.mp3   # Alarm audio file
└── README.md
```

---

## How It Works

### Backend — Fire Detection Pipeline (`main.py`)

1. The frontend sends a **base64-encoded JPEG frame** to the `/detect` endpoint every second.
2. The backend decodes the image using OpenCV.
3. The frame is converted to **HSV color space**.
4. Four color masks are applied to isolate fire-like pixels:
   - Lower red range: H `0–15`
   - Upper red range: H `160–180` (wraps in HSV)
   - Orange range: H `5–35`
   - Yellow range: H `20–45`
   - All require: Saturation `≥ 100`, Value (brightness) `≥ 150`
5. **Morphological operations** (opening + dilation) remove noise.
6. **Confidence score** is calculated from:
   - Fire pixel ratio (fire area / total frame area)
   - Brightness ratio of fire-region pixels (pixels > 180 brightness)
   - Formula: `confidence = (fire_ratio × 10) × (0.5 + 0.5 × brightness_ratio)`
7. A **bounding box** is drawn around the largest fire contour.
8. If confidence ≥ **20%**, fire is declared detected.

### Frontend — Dashboard UI (`index.html`)

- Built with **React 18** (loaded via CDN) and **Babel Standalone** for JSX.
- Captures a frame from the video element every **1 second** using an HTML5 canvas.
- Sends the base64 frame to the FastAPI backend via `fetch`.
- On detection, triggers animations, the alarm, and logs the event.
- Falls back to **demo mode** (random probabilistic fire events) if the backend is unreachable.

---

## Running the Project

### Prerequisites

- Python 3.9+
- A webcam

### 1. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Start the Backend Server

```bash
cd backend
python main.py
```

The FastAPI server will start at `http://localhost:8000`.

You can verify it is running by visiting: [http://localhost:8000/health](http://localhost:8000/health)

### 3. Start the Frontend Server

Open a new terminal and run:

```bash
cd frontend
python serve.py
```

This will start a local HTTP server on port **8001** and automatically open the app in your browser at `http://localhost:8001/index.html`.

> **Why use `serve.py` instead of opening `index.html` directly?**
> Opening `index.html` as a `file://` URL will block the audio from loading due to browser security restrictions. The `serve.py` server serves it over HTTP so audio and API calls work correctly.

---

## 🔌 API Reference

### `GET /health`

Returns the current server status and detection mode.

**Response:**

```json
{
  "status": "healthy",
  "detection_mode": "HSV"
}
```

### `POST /detect`

Accepts a base64-encoded JPEG image and returns a fire detection result.

**Request Body:**

```json
{
  "frame": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

**Response:**

```json
{
  "fire_detected": true,
  "confidence": 0.7821,
  "bbox": [0.2, 0.3, 0.4, 0.35],
  "message": "[!] FIRE DETECTED"
}
```

> `bbox` is `[x, y, width, height]` normalized to 0–1 relative to the frame dimensions.

---

## Optional: YOLO Model Integration

To use a custom YOLO fire detection model instead of HSV:

1. Install Ultralytics:
   ```bash
   pip install ultralytics
   ```
2. Place your trained model file (e.g., `fire_model.pt`) in the `backend/` folder.
3. In `backend/main.py`, uncomment these lines:
   ```python
   from ultralytics import YOLO
   model = YOLO("fire_model.pt")
   USE_YOLO = True
   ```
4. Restart the backend.

---

## Tech Stack

| Layer            | Technology                                         |
| ---------------- | -------------------------------------------------- |
| Frontend         | HTML5, React 18 (CDN), Babel Standalone, CSS3      |
| Backend          | Python, FastAPI, Uvicorn                           |
| Computer Vision  | OpenCV, NumPy                                      |
| Image Processing | Pillow                                             |
| Fonts            | Google Fonts — Orbitron, Rajdhani, Share Tech Mono |
| Audio            | HTML5 Audio API + Web Audio API (fallback)         |
| AI (Optional)    | Ultralytics YOLO                                   |

---

## Configuration

| Parameter              | Location              | Default                 | Description                        |
| ---------------------- | --------------------- | ----------------------- | ---------------------------------- |
| `CONFIDENCE_THRESHOLD` | `backend/main.py`     | `0.20`                  | Minimum confidence to declare fire |
| `BACKEND_URL`          | `frontend/index.html` | `http://localhost:8000` | Backend API URL                    |
| Detection interval     | `frontend/index.html` | `1000ms`                | Frame capture rate                 |
| Frontend port          | `frontend/serve.py`   | `8001`                  | Local server port                  |
| Backend port           | `backend/main.py`     | `8000`                  | FastAPI server port                |

---

## Notes & Limitations

- **HSV detection is color-based** and may produce false positives on brightly lit orange/red objects (e.g., sunsets, certain clothing). Adjust the `CONFIDENCE_THRESHOLD` or switch to YOLO for more accuracy.
- **Camera permission** must be granted by the browser for the live feed to work.
- **Audio autoplay** is restricted by modern browsers until the user interacts with the page. A silent audio unlock is attempted on Start button click.
- The system runs at **1 Hz** (one detection per second) by default to balance performance and responsiveness.

---

## License

This project is for educational and research purposes.

---

> Built with by Arvind | PyroWatch (Aag) — AI Fire Detection System
