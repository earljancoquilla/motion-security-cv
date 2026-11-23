"""
motion_security.py
Motion-Triggered Security Application (Python + OpenCV)

Techniques used (from the laboratory topics):
  a. Image Operations ............ resize, grayscale, Gaussian blur, drawing/overlays
  b. Image Segmentation .......... background subtraction (MOG2), thresholding, morphology
  e. Motion and Object Tracking .. contours, bounding boxes, centroid trails
  c. Haar Cascade Classifier ..... (optional) face detection inside motion regions

Usage examples:
  python motion_security.py --source 0                      # live webcam
  python motion_security.py --source samples/intruder.mp4   # recorded video
  python motion_security.py --source samples/intruder.mp4 --headless --output alerts/out.mp4
"""

import argparse
import csv
import os
import sys
from collections import deque
from datetime import datetime

import cv2
import numpy as np

# ----------------------------- CONFIGURATION ------------------------------ #
FRAME_WIDTH = 640          # every frame is resized to this width (faster processing)
MIN_CONTOUR_AREA = 1500    # ignore blobs smaller than this (noise, insects, light flicker)
ALERT_FRAMES = 5           # motion must last this many consecutive frames to trigger alert
COOLDOWN_FRAMES = 60       # wait this many frames before saving another snapshot
WARMUP_FRAMES = 30         # frames used for the background model to "learn" the scene
TRAIL_LENGTH = 30          # number of past centroids drawn as a trail
ALERT_DIR = "alerts"       # snapshots and log are saved here


# ------------------------------ HELPER FUNCTIONS -------------------------- #
def open_source(source):
    """Open a webcam (integer index) or a video file path."""
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        if not os.path.exists(source):
            sys.exit(f"[ERROR] Video file not found: {source}")
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit("[ERROR] Could not open the video source. "
                 "Check the camera index / file path / camera permissions.")
    return cap


def load_face_cascade():
    """Load the Haar cascade for frontal faces that ships with OpenCV."""
    path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        print("[WARN] Haar cascade not loaded - face detection disabled.")
        return None
    return cascade


def preprocess(frame):
    """Image operations: resize -> grayscale -> Gaussian blur."""
    h, w = frame.shape[:2]
    scale = FRAME_WIDTH / float(w)
    frame = cv2.resize(frame, (FRAME_WIDTH, int(h * scale)))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)  # removes sensor noise
    return frame, gray


def segment_motion(gray, subtractor, kernel):
    """Image segmentation: background subtraction + threshold + morphology."""
    fg_mask = subtractor.apply(gray)                          # foreground = moving pixels
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)  # drop shadows (gray=127)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)       # remove tiny specks
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=3)               # fill holes in objects
    return fg_mask


def find_motion_regions(fg_mask):
    """Find contours in the mask and keep only the big ones."""
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]


def detect_faces(cascade, frame, box):
    """Run the Haar cascade only inside the motion bounding box (fast + fewer false hits)."""
    x, y, w, h = box
    roi = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return [(x + fx, y + fy, fw, fh) for (fx, fy, fw, fh) in faces]


def draw_overlay(frame, contours, trail, fg_mask, alert, frame_no, faces):
    """Visualization: boxes, labels, trail, mask thumbnail, status banner."""
    out = frame.copy()

    for i, c in enumerate(contours, start=1):
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(out, [c], -1, (255, 200, 0), 1)
        cv2.putText(out, f"Object {i}", (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(out, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)
        cv2.putText(out, "Face", (fx, fy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # Tracking trail of the largest object's centroid
    pts = [p for p in trail if p is not None]
    for j in range(1, len(pts)):
        thickness = max(1, int(3 * j / len(pts)))
        cv2.line(out, pts[j - 1], pts[j], (0, 255, 255), thickness)

    # Mask thumbnail in the top-right corner (picture-in-picture)
    thumb = cv2.resize(fg_mask, (160, int(160 * fg_mask.shape[0] / fg_mask.shape[1])))
    thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
    th, tw = thumb.shape[:2]
    out[5:5 + th, out.shape[1] - tw - 5:out.shape[1] - 5] = thumb
    cv2.rectangle(out, (out.shape[1] - tw - 5, 5), (out.shape[1] - 5, 5 + th), (255, 255, 255), 1)

    # Status banner
    color, text = ((0, 0, 255), "ALERT: MOTION DETECTED") if alert else ((0, 160, 0), "Monitoring...")
    cv2.rectangle(out, (0, 0), (300, 30), color, -1)
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(out, f"{stamp}  frame {frame_no}", (8, out.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def log_event(writer, frame_no, n_objects, n_faces, snapshot_path):
    """Append one row to the alert log (CSV)."""
    writer.writerow([datetime.now().isoformat(timespec="seconds"), frame_no,
                     n_objects, n_faces, snapshot_path])


# ---------------------------------- MAIN ---------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Motion-triggered security camera (OpenCV)")
    ap.add_argument("--source", default="0", help="0 for webcam, or path to a video file")
    ap.add_argument("--output", help="optional path to save the annotated video (.mp4)")
    ap.add_argument("--headless", action="store_true", help="do not open a display window")
    ap.add_argument("--no-faces", action="store_true", help="disable Haar face detection")
    args = ap.parse_args()

    os.makedirs(ALERT_DIR, exist_ok=True)
    cap = open_source(args.source)
    face_cascade = None if args.no_faces else load_face_cascade()

    # MOG2 builds a statistical model of the background and flags pixels that differ.
    subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    log_file = open(os.path.join(ALERT_DIR, "alert_log.csv"), "a", newline="")
    log = csv.writer(log_file)
    if log_file.tell() == 0:
        log.writerow(["timestamp", "frame", "objects", "faces", "snapshot"])

    writer = None
    trail = deque(maxlen=TRAIL_LENGTH)
    motion_streak, cooldown, frame_no, alerts_saved, paused = 0, 0, 0, 0, False
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    print("[INFO] Running. Press 'q' to quit, 's' to save a screenshot, SPACE to pause.")
    while True:
        if not paused:
            ok, raw = cap.read()
            if not ok:                      # end of video or camera error
                break
            frame_no += 1

            frame, gray = preprocess(raw)
            fg_mask = segment_motion(gray, subtractor, kernel)

            contours, faces = [], []
            if frame_no > WARMUP_FRAMES:    # ignore the first frames while the model learns
                contours = find_motion_regions(fg_mask)

            # Motion must persist for several frames to avoid false alarms
            motion_streak = motion_streak + 1 if contours else 0
            alert = motion_streak >= ALERT_FRAMES

            if contours:
                biggest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(biggest)
                trail.append((x + w // 2, y + h // 2))
                if face_cascade is not None:
                    faces = detect_faces(face_cascade, frame, (x, y, w, h))
            else:
                trail.clear()

            display = draw_overlay(frame, contours, trail, fg_mask, alert, frame_no, faces)

            if cooldown > 0:
                cooldown -= 1
            if alert and cooldown == 0:     # save evidence + log it
                name = datetime.now().strftime("alert_%Y%m%d_%H%M%S_") + f"{frame_no}.jpg"
                path = os.path.join(ALERT_DIR, name)
                cv2.imwrite(path, display)
                log_event(log, frame_no, len(contours), len(faces), path)
                log_file.flush()
                alerts_saved += 1
                cooldown = COOLDOWN_FRAMES
                print(f"[ALERT] Motion at frame {frame_no} -> {path}")

            if args.output:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.output, fourcc, fps,
                                             (display.shape[1], display.shape[0]))
                writer.write(display)

        if not args.headless:
            cv2.imshow("Motion-Triggered Security", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv2.imwrite(os.path.join(ALERT_DIR, f"screenshot_{frame_no}.jpg"), display)
            elif key == ord(" "):
                paused = not paused

    cap.release()
    if writer:
        writer.release()
    log_file.close()
    cv2.destroyAllWindows()
    print(f"[DONE] Frames processed: {frame_no} | Alerts saved: {alerts_saved}")


if __name__ == "__main__":
    main()
