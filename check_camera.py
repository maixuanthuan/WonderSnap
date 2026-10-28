"""Find which /dev/video index is the real webcam.
Usage: python check_camera.py
Saves cam_<index>.jpg for every camera that opens; open the images and
pick the one that shows your face, then set camera.index in config.yaml.
"""
import cv2

for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        continue
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frame = None
    for _ in range(30):          # let auto-exposure settle
        ok, f = cap.read()
        if ok:
            frame = f
    if frame is None:
        print(f"index {i}: opens but returns no frames (metadata node?)")
    else:
        mean = float(frame.mean())
        cv2.imwrite(f"cam_{i}.jpg", frame)
        tag = "OK" if mean >= 8 else "BLACK (IR camera?)"
        print(f"index {i}: {frame.shape[1]}x{frame.shape[0]}  brightness={mean:.1f}  {tag}  -> cam_{i}.jpg")
    cap.release()
