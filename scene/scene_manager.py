import numpy as np


class SceneManager:
    OBJECTS = ["bigben", "pyramids", "turtle_tower", "liberty"]
    MODE_VALUE = {"build": 1.0, "idle": 0.85, "explode": 0.0, "snap": 0.85}

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.size_mult = float(self.cfg["render"].get("object_scale", 1.6))
        self.follow = bool(self.cfg["render"].get("follow_hand", False))
        self.fixed_size = bool(self.cfg["render"].get("fixed_size", True))
        self.palette_index = int(getattr(self, "palette_index", 0))
        self.object_index = 0
        self.current_object = self.OBJECTS[0]
        self.anchor = np.array([0.5, 0.5], dtype=np.float32)
        self.target_anchor = self.anchor.copy()
        self.hand_scale = 0.35
        self.target_scale = 0.35
        self.state = "build"
        self.state_time = 0.0
        self.transition = 0.0
        self.shockwave = 0.0
        self.particle_mode = "build"
        self.mode_value = 1.0
        self.rotation = 0.0

    def handle_gesture(self, event):
        if event == "FIST":
            if self.state == "explode":
                # open -> fist cycle: particles re-assemble into the NEXT wonder
                self.set_object(self.object_index + 1)
            else:
                self.state, self.state_time, self.particle_mode = "build", 0.0, "build"
        elif event == "OPEN":
            self.state, self.state_time, self.particle_mode = "explode", 0.0, "explode"
        elif event == "SNAP":
            self.state, self.state_time = "snap", 0.0
            self.shockwave = 1.0
            self.transition = 1.0
            self.particle_mode = "snap"
            self.set_object(self.object_index + 1, keep_state=True)

    def change_size(self, factor):
        self.size_mult = float(np.clip(self.size_mult * factor, 0.5, 3.5))

    def set_object(self, index, keep_state=False):
        self.object_index = index % len(self.OBJECTS)
        self.current_object = self.OBJECTS[self.object_index]
        if not keep_state:
            self.state, self.state_time, self.particle_mode = "build", 0.0, "build"
            self.transition = 1.0

    def update(self, dt, anchor, hand_size):
        self.state_time += dt
        self.rotation += dt * 0.55

        center = np.array([0.5, 0.5], dtype=np.float32)
        if anchor is not None:
            a = np.array(anchor, dtype=np.float32)
            gain = float(self.cfg["render"].get("follow_gain", 0.6))
            if self.fixed_size:
                self.target_scale = float(np.clip(0.42 * self.size_mult, 0.25, 0.95))
            else:
                self.target_scale = float(np.clip(hand_size * 3.0 * self.size_mult, 0.25, 0.95))
            # follow_gain < 1 keeps the object inside the result pane even if the hand is at the edge.
            self.target_anchor = center + (a - center) * gain if self.follow else center.copy()
        else:
            if not self.follow:
                self.target_anchor = center.copy()
            if self.fixed_size:
                self.target_scale = float(np.clip(0.42 * self.size_mult, 0.25, 0.95))

        k = min(1.0, float(self.cfg["tracking"]["smoothing"]) * dt * 60.0)
        self.anchor += (self.target_anchor - self.anchor) * k
        self.hand_scale += (self.target_scale - self.hand_scale) * k

        self.transition = max(0.0, self.transition - dt / self.cfg["render"]["transition_duration"])
        self.shockwave = max(0.0, self.shockwave - dt / self.cfg["render"]["shockwave_duration"])

        if self.state == "explode" and self.state_time > 1.2:
            self.particle_mode = "idle"
        elif self.state == "snap" and self.state_time > 0.9:
            self.particle_mode = "build"

        target = self.MODE_VALUE[self.particle_mode]
        rate = 14.0 if self.particle_mode == "explode" else 5.0
        self.mode_value += (target - self.mode_value) * min(1.0, dt * rate)

    @property
    def shockwave_strength(self):
        return self.shockwave

    @property
    def effect_time(self):
        return self.state_time
