import numpy as np


class ParticleSystem:
    """
    CPU simulation (numpy) -> uploaded to a GPU vertex buffer.
    - Changing target = vortex morph: particles bloom out, swirl around the vertical axis
      and fold into the new shape, bottom first (the new wonder "grows" upward).
    - Otherwise a spring keeps them on the target.
    Each particle also has a fixed random seed (separate static buffer) so the shader
    effects don't flicker while particles move.
    """

    def __init__(self, ctx, target, cfg):
        self.ctx = ctx
        self.count = len(target)
        self.spring = float(cfg["spring"])
        self.damping = float(cfg["damping"])
        self.morph_duration = float(cfg.get("morph_duration", 1.8))
        self.swirl = float(cfg.get("morph_swirl", 2.6))

        rng = np.random.default_rng(1234)
        self.target = target.astype(np.float32, copy=True)
        self.pos = (self.target + rng.normal(0, 0.35, target.shape)).astype(np.float32)
        self.vel = np.zeros_like(self.pos)
        self.seed = rng.random(self.count).astype(np.float32)
        self.morph = None

        self.buffer = ctx.buffer(self.pos.tobytes(), dynamic=True)
        self.seed_buffer = ctx.buffer(self.seed.tobytes())

    def set_target(self, target):
        if target.shape != self.target.shape:
            raise ValueError("Particle count cannot change at runtime.")
        self.target = target.astype(np.float32, copy=False)
        z = self.target[:, 2]
        height = (z - z.min()) / max(float(np.ptp(z)), 1e-5)
        self.morph = {
            "src": self.pos.copy(),                      # start from wherever particles are now
            "delay": height * 0.40 + self.seed * 0.10,   # 0..0.5 of the duration, bottom first
            "t": 0.0,
        }
        self.vel[:] = 0.0

    def _update_morph(self, dt):
        m = self.morph
        m["t"] += dt / self.morph_duration
        p = np.clip((m["t"] - m["delay"]) / 0.5, 0.0, 1.0)      # each particle travels for half the duration
        e = p * p * p * (p * (p * 6.0 - 15.0) + 10.0)           # smootherstep
        w = np.sin(np.pi * e)                                    # 0 at both ends -> exact start/end

        pos = m["src"] + (self.target - m["src"]) * e[:, None]
        pos[:, :2] *= (1.0 + 0.45 * w)[:, None]                  # bloom outward
        pos[:, 2] += 0.12 * w * (self.seed - 0.5)                # float a little up/down
        ang = self.swirl * w * (0.6 + 0.8 * self.seed)           # swirl around the vertical axis
        c, s = np.cos(ang), np.sin(ang)
        x, y = pos[:, 0].copy(), pos[:, 1].copy()
        pos[:, 0] = x * c - y * s
        pos[:, 1] = x * s + y * c
        self.pos = pos.astype(np.float32)

        if p.min() >= 1.0:
            self.pos = self.target.copy()
            self.morph = None

    def update(self, dt):
        if self.morph is not None:
            self._update_morph(dt)
        else:
            # Normalise to a 60 fps step so behaviour does not depend on frame rate.
            step = np.float32(min(dt * 60.0, 3.0))
            self.vel += (self.target - self.pos) * np.float32(self.spring) * step
            self.vel *= np.float32(self.damping ** float(step))
            self.pos += self.vel * step
        self.buffer.write(self.pos.tobytes())

    def release(self):
        for b in (self.buffer, self.seed_buffer):
            try:
                b.release()
            except Exception:
                pass