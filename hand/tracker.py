from dataclasses import dataclass
import cv2
import mediapipe as mp
import numpy as np


@dataclass
class HandResult:
    landmarks: object | None
    anchor: tuple[float, float] | None
    hand_size: float
    handedness: str


class HandTracker:
    def __init__(self, model_complexity=0):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=model_complexity,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            return HandResult(None, None, 0.0, "")

        lm = result.multi_hand_landmarks[0]
        pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)

        # Palm center from wrist + MCP joints.
        palm_ids = [0, 5, 9, 13, 17]
        anchor_xy = pts[palm_ids, :2].mean(axis=0)

        # Palm/hand size as a scale proxy.
        hand_size = float(np.linalg.norm(pts[5, :2] - pts[17, :2]))
        handedness = ""
        if result.multi_handedness:
            handedness = result.multi_handedness[0].classification[0].label

        return HandResult(lm, tuple(anchor_xy.tolist()), hand_size, handedness)

    def close(self):
        self.hands.close()
