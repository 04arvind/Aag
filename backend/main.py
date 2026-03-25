"""
AI Fire Detection Backend
FastAPI + OpenCV with HSV-based fire detection (with optional YOLO support)
"""

import cv2
import numpy as np
import base64
import logging
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Fire Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
USE_YOLO = False
class FrameRequest(BaseModel):
    frame: str 
class DetectionResponse(BaseModel):
    fire_detected: bool
    confidence: float
    bbox: Optional[list] = None  # [x, y, w, h] normalized
    message: str


def decode_base64_frame(b64_string: str) -> np.ndarray:
    """Decode a base64 image string to an OpenCV frame."""
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    img_bytes = base64.b64decode(b64_string)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode image frame")
    return frame


def detect_fire_hsv(frame: np.ndarray) -> dict:
    """
    HSV-based fire detection using color thresholding.
    Fire appears as bright red/orange/yellow pixels.
    Returns detection result with confidence score.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 150])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 100, 150])
    upper_red2 = np.array([180, 255, 255])
    lower_orange = np.array([5, 100, 150])
    upper_orange = np.array([35, 255, 255])
    lower_yellow = np.array([20, 100, 150])
    upper_yellow = np.array([45, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    fire_mask = cv2.bitwise_or(mask_red1, mask_red2)
    fire_mask = cv2.bitwise_or(fire_mask, mask_orange)
    fire_mask = cv2.bitwise_or(fire_mask, mask_yellow)

    kernel = np.ones((5, 5), np.uint8)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_DILATE, kernel)

    total_pixels = frame.shape[0] * frame.shape[1]
    fire_pixels = np.sum(fire_mask > 0)
    fire_ratio = fire_pixels / total_pixels

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bright_pixels = np.sum(gray[fire_mask > 0] > 180) if fire_pixels > 0 else 0
    brightness_ratio = bright_pixels / fire_pixels if fire_pixels > 0 else 0

    raw_confidence = min(fire_ratio * 10, 1.0)  # scale: 10% coverage = 100% confidence
    adjusted_confidence = raw_confidence * (0.5 + 0.5 * brightness_ratio)

    bbox = None
    if fire_pixels > 100:
        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            h_img, w_img = frame.shape[:2]
            bbox = [
                round(x / w_img, 4),
                round(y / h_img, 4),
                round(w / w_img, 4),
                round(h / h_img, 4)
            ]

    CONFIDENCE_THRESHOLD = 0.20  # HSV threshold (lower since it's color-based)
    fire_detected = adjusted_confidence >= CONFIDENCE_THRESHOLD

    return {
        "fire_detected": fire_detected,
        "confidence": round(float(adjusted_confidence), 4),
        "bbox": bbox
    }


def detect_fire_yolo(frame: np.ndarray) -> dict:
    """YOLO-based fire detection — requires a trained fire model."""
    results = model(frame, conf=0.5, verbose=False)
    fire_detected = False
    confidence = 0.0
    bbox = None

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf > confidence:
                confidence = conf
                fire_detected = True
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                h_img, w_img = frame.shape[:2]
                bbox = [
                    round(x1 / w_img, 4),
                    round(y1 / h_img, 4),
                    round((x2 - x1) / w_img, 4),
                    round((y2 - y1) / h_img, 4)
                ]

    return {
        "fire_detected": fire_detected,
        "confidence": round(confidence, 4),
        "bbox": bbox
    }


@app.get("/")
async def root():
    return {"status": "online", "service": "AI Fire Detection API", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy", "detection_mode": "YOLO" if USE_YOLO else "HSV"}


@app.post("/detect", response_model=DetectionResponse)
async def detect_fire(request: FrameRequest):
    """
    Accepts a base64-encoded image frame.
    Returns fire detection result with confidence score.
    """
    try:
        frame = decode_base64_frame(request.frame)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image frame: {str(e)}")

    try:
        if USE_YOLO:
            result = detect_fire_yolo(frame)
        else:
            result = detect_fire_hsv(frame)
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    message = "[!] FIRE DETECTED" if result["fire_detected"] else "[OK] No fire detected"
    logger.info(f"Detection: {message} | Confidence: {result['confidence']:.2%}")

    return DetectionResponse(
        fire_detected=result["fire_detected"],
        confidence=result["confidence"],
        bbox=result["bbox"],
        message=message
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)