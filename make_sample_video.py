"""
make_sample_video.py
Creates samples/intruder.mp4 - a synthetic 'room' where a person-like figure
walks across the scene. Use it to test the app when you have no webcam
or no recorded video.
"""
import os
import cv2
import numpy as np

os.makedirs("samples", exist_ok=True)
W, H, FPS, TOTAL = 640, 360, 25, 250
out = cv2.VideoWriter("samples/intruder.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
if not out.isOpened():
    out.release()
    raise SystemExit("[ERROR] Could not create samples/intruder.mp4. Check MP4 codec support and folder permissions.")

# Static background: wall, floor, door, window
bg = np.full((H, W, 3), (170, 160, 150), np.uint8)
cv2.rectangle(bg, (0, 260), (W, H), (90, 110, 130), -1)        # floor
cv2.rectangle(bg, (60, 60), (200, 200), (210, 190, 120), -1)    # window
cv2.rectangle(bg, (500, 70), (600, 260), (60, 80, 110), -1)     # door

rng = np.random.default_rng(1)
for f in range(TOTAL):
    frame = bg.copy()
    noise = rng.normal(0, 3, frame.shape).astype(np.int16)       # camera sensor noise
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Intruder walks in from frame 60 to frame 200
    if 60 <= f <= 200:
        x = int(-60 + (f - 60) * 4.6)
        cv2.rectangle(frame, (x, 150), (x + 50, 270), (40, 40, 160), -1)     # body
        cv2.circle(frame, (x + 25, 130), 22, (150, 190, 230), -1)           # head
    out.write(frame)
out.release()
print("Created samples/intruder.mp4")
