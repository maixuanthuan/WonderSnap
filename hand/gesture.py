import numpy as np


def _dist(a, b):
    return float(np.linalg.norm(a - b))


class GestureState:
    def __init__(self, label="NONE", event=None, ext=0.0, pinch=0.0):
        self.label = label
        self.event = event      # "FIST" | "OPEN" | "SNAP" | None  (fires once per gesture)
        self.ext = ext          # finger extension ratio (for HUD / tuning)
        self.pinch = pinch      # thumb-index distance / palm width (for HUD / tuning)


class GestureEngine:
    """
    Rule-based gesture classifier.

    ext   = mean over middle/ring/pinky of  dist(tip, wrist) / dist(mcp, wrist)
            open hand ~ 1.8-2.0, fist ~ 0.8-1.2
    FIST / OPEN fire an EVENT once, when the label changes and stays stable.
    SNAP: thumb-index pinch closes, then releases quickly.
    """

    def __init__(self, cfg):
        self.open_ratio = float(cfg["open_ratio"])
        self.fist_ratio = float(cfg["fist_ratio"])
        self.stable_frames = int(cfg.get("stable_frames", 3))
        self.snap_close = float(cfg["snap_close_distance"])
        self.snap_open = float(cfg["snap_open_distance"])
        self.snap_velocity = float(cfg["snap_velocity"])
        self.cooldown = float(cfg["cooldown_seconds"])

        self.prev_pinch = None
        self.prev_time = None
        self.pinch_closed_at = None
        self.last_snap = -999.0

        self.stable_label = "NONE"
        self.cand_label = "NONE"
        self.cand_count = 0

    def _extension(self, pts):
        wrist = pts[0]
        ratios = []
        for mcp, tip in ((9, 12), (13, 16), (17, 20)):
            ratios.append(_dist(pts[tip], wrist) / max(_dist(pts[mcp], wrist), 1e-5))
        return float(np.mean(ratios))

    def update(self, landmarks, now, aspect=1.0):
        if landmarks is None:
            self.prev_pinch = None
            self.prev_time = now
            self.pinch_closed_at = None
            self.stable_label = "NONE"
            self.cand_label, self.cand_count = "NONE", 0
            return GestureState("NONE", None)

        pts = np.array([[p.x * aspect, p.y, p.z] for p in landmarks.landmark], dtype=np.float32)
        pts[:, 2] = 0.0  # 2D is more stable than MediaPipe's relative z

        ext = self._extension(pts)
        palm = max(_dist(pts[5], pts[17]), 1e-5)
        pinch = _dist(pts[4], pts[8]) / palm

        velocity = 0.0
        if self.prev_pinch is not None and self.prev_time is not None:
            velocity = (pinch - self.prev_pinch) / max(now - self.prev_time, 1e-4)
        self.prev_pinch, self.prev_time = pinch, now

        event = None

        # --- SNAP (pinch -> fast release). Ignore when the hand is a fist. ---
        if pinch < self.snap_close and ext > self.fist_ratio + 0.15:
            self.pinch_closed_at = now
        if (
            self.pinch_closed_at is not None
            and now - self.pinch_closed_at < 1.5
            and pinch > self.snap_open
            and velocity > self.snap_velocity
            and now - self.last_snap > self.cooldown
        ):
            event = "SNAP"
            self.last_snap = now
            self.pinch_closed_at = None

        # --- FIST / OPEN label with debounce ---
        if ext >= self.open_ratio:
            raw = "OPEN"
        elif ext <= self.fist_ratio:
            raw = "FIST"
        else:
            raw = "NEUTRAL"

        if raw == self.cand_label:
            self.cand_count += 1
        else:
            self.cand_label, self.cand_count = raw, 1

        if self.cand_count >= self.stable_frames and raw != self.stable_label:
            self.stable_label = raw
            if event is None and raw in ("FIST", "OPEN") and now - self.last_snap > 0.4:
                event = raw

        label = "SNAP" if event == "SNAP" else self.stable_label
        return GestureState(label, event, ext, pinch)
