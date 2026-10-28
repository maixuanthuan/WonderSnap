import ctypes
import numpy as np
import moderngl
import cv2
from pathlib import Path


VERT = r"""
#version 330

in vec3 in_pos;
in float in_seed;   // fixed per particle -> no flicker while morphing

uniform float u_time;
uniform float u_mode;
uniform float u_shockwave;
uniform float u_transition;
uniform float u_rotation;
uniform vec2 u_anchor;
uniform float u_scale;
uniform float u_point_size;
uniform float u_aspect;

out float v_energy;
out float v_depth;
out float v_height;

mat2 rot(float a) {
    float c = cos(a);
    float s = sin(a);
    return mat2(c, -s, s, c);
}

void main() {
    // Point clouds are built z-up; the screen is y-up (image y grows downward),
    // so map model (x, y, z) -> view (x, -z, y). "y" of the model becomes depth.
    vec3 m = vec3(in_pos.x, -in_pos.z, in_pos.y);
    vec3 p = m;

    // Spin around the vertical axis.
    p.xz = rot(u_rotation) * p.xz;
    // Slight downward tilt so rings/domes read as 3D instead of edge-on.
    p.yz = rot(-0.45) * p.yz;

    float h = in_seed;
    vec3 dir = normalize(p + vec3(0.001, 0.003, 0.002));

    // u_mode: 1.0 = built, ~0.85 = idle shimmer, 0.0 = exploded.
    float build_amt = clamp(u_mode, 0.0, 1.0);
    p += dir * (1.0 - build_amt) * (0.15 + h * 0.55);

    // SNAP shockwave.
    p += dir * u_shockwave * (0.10 + h * 0.35);

    // Transition turbulence.
    p += vec3(
        sin(u_time * 8.0 + h * 30.0),
        cos(u_time * 7.0 + h * 27.0),
        sin(u_time * 9.0 + h * 21.0)
    ) * u_transition * 0.018;

    float z = 1.0 + p.z * 0.35;
    // u_scale is relative to image HEIGHT -> divide x by aspect to keep it round.
    vec2 screen = vec2(p.x / u_aspect, p.y) * u_scale / z;

    // Image coords (0..1, y down) -> clip space of the pane viewport.
    float x = (u_anchor.x + screen.x - 0.5) * 2.0;
    float y = (0.5 - (u_anchor.y + screen.y)) * 2.0;

    gl_Position = vec4(x, y, 0.0, 1.0);
    gl_PointSize = u_point_size * (1.0 + (1.0 - z) * 0.8);

    v_energy = 0.45 + 0.55 * h;
    // Far side of the object is dimmer -> readable 3D shape.
    v_depth = 1.0 - 0.60 * clamp(p.z + 0.5, 0.0, 1.0);
    // Colour gradient bottom -> top of the model (model z is "up").
    v_height = clamp(in_pos.z + 0.5, 0.0, 1.0);
}
"""

FRAG = r"""
#version 330

in float v_energy;
in float v_depth;
in float v_height;
uniform float u_alpha;
uniform float u_glow;
uniform vec3 u_col_a;
uniform vec3 u_col_b;
uniform float u_rainbow;
uniform float u_hue;
uniform float u_transition;
out vec4 fragColor;

vec3 hsv2rgb(vec3 c) {
    vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

void main() {
    float d = length(gl_PointCoord - vec2(0.5));
    if (d > 0.5)
        discard;

    float a = smoothstep(0.5, 0.12, d) * u_alpha;
    vec3 base = mix(u_col_a, u_col_b, v_height);
    if (u_rainbow > 0.5)
        base = hsv2rgb(vec3(fract(v_height * 0.75 + u_hue), 0.75, 1.0));
    vec3 c = base * (0.60 + 0.40 * v_energy) * v_depth;
    // Small bright core -> sparkle without washing everything to white.
    c += vec3(0.30) * smoothstep(0.22, 0.0, d) * u_glow * v_depth;
    // Energy flash while morphing between wonders.
    c *= 1.0 + 0.7 * u_transition;
    fragColor = vec4(c, a);
}
"""

QUAD_VERT = r"""
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 uv;
void main() {
    uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

QUAD_FRAG = r"""
#version 330
uniform sampler2D tex;
in vec2 uv;
out vec4 color;
void main() {
    color = texture(tex, uv);
}
"""

PALETTES = {
    "bigben":       ((1.00, 0.55, 0.15), (1.00, 0.92, 0.55)),
    "pyramids":     ((0.95, 0.40, 0.08), (1.00, 0.85, 0.45)),
    "turtle_tower": ((0.15, 0.75, 0.55), (1.00, 0.85, 0.45)),
    "liberty":      ((0.10, 0.70, 0.60), (1.00, 0.85, 0.30)),
}

# Selectable with key P. "auto" = each object keeps its own colours.
PALETTE_LIST = [
    ("auto", None),
    ("ocean",   ((0.10, 0.60, 1.00), (0.55, 1.00, 0.95))),
    ("fire",    ((1.00, 0.15, 0.05), (1.00, 0.85, 0.20))),
    ("emerald", ((0.05, 0.75, 0.35), (0.70, 1.00, 0.45))),
    ("sunset",  ((0.95, 0.20, 0.55), (1.00, 0.70, 0.25))),
    ("violet",  ((0.45, 0.20, 1.00), (0.95, 0.55, 1.00))),
    ("gold",    ((0.85, 0.50, 0.05), (1.00, 0.95, 0.55))),
    ("ice",     ((0.55, 0.80, 1.00), (1.00, 1.00, 1.00))),
    ("neon",    ((0.10, 1.00, 0.85), (1.00, 0.20, 0.90))),
    ("aurora",  ((0.10, 1.00, 0.50), (0.55, 0.30, 1.00))),
    ("candy",   ((1.00, 0.45, 0.75), (0.45, 0.95, 1.00))),
    ("lava",    ((0.80, 0.05, 0.00), (1.00, 0.95, 0.60))),
    ("sakura",  ((1.00, 0.55, 0.70), (1.00, 0.92, 0.95))),
    ("mint",    ((0.10, 0.70, 0.65), (0.85, 1.00, 0.95))),
    ("galaxy",  ((0.10, 0.15, 0.85), (1.00, 0.30, 0.85))),
    ("lime",    ((0.40, 0.90, 0.05), (1.00, 1.00, 0.30))),
    ("blood",   ((0.45, 0.00, 0.05), (1.00, 0.15, 0.20))),
    ("silver",  ((0.55, 0.58, 0.65), (1.00, 1.00, 1.00))),
    ("custom",  None),   # your own two colours: render.custom_colors in config.yaml
    ("rainbow", None),
]
PALETTE_COUNT = len(PALETTE_LIST)


def palette_name(index):
    return PALETTE_LIST[index % PALETTE_COUNT][0]


def palette_index_by_name(name):
    for i, (n, _) in enumerate(PALETTE_LIST):
        if n == str(name).lower():
            return i
    return 0


BG_COLOR = (0.015, 0.02, 0.045, 1.0)


class Renderer:
    def __init__(self, ctx, width, height, cfg):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.cfg = cfg

        self.particle_prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.bg_prog = ctx.program(vertex_shader=QUAD_VERT, fragment_shader=QUAD_FRAG)

        quad = np.array([
            -1, -1, 0, 0,
             1, -1, 1, 0,
            -1,  1, 0, 1,
             1,  1, 1, 1,
        ], dtype=np.float32)
        self.quad_vbo = ctx.buffer(quad.tobytes())
        self.quad_vao = ctx.vertex_array(
            self.bg_prog, [(self.quad_vbo, "2f 2f", "in_pos", "in_uv")]
        )

        # Two textures: left = camera + HUD/landmarks, right = clean camera for the AR view.
        self.tex_left = self._make_tex()
        self.tex_right = self._make_tex()
        self.bg_prog["tex"] = 0

        self.particle_vao = None
        self.last_buffer = None

        def _hex(h):
            h = str(h).lstrip("#")
            return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        try:
            ca, cb = cfg["render"].get("custom_colors", ["#ff6a00", "#00d4ff"])
            self.custom = (_hex(ca), _hex(cb))
        except Exception:
            self.custom = ((1.0, 0.42, 0.0), (0.0, 0.83, 1.0))

    def _make_tex(self):
        t = self.ctx.texture((self.width, self.height), 3, dtype="f1")  # "u1" = INTEGER texture -> sampler2D reads 0 (black)
        t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return t

    def _upload(self, tex, frame):
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(np.flipud(rgb))  # GL origin is bottom-left
        tex.write(rgb.tobytes())

    def _ensure_particle_vao(self, particles):
        if self.last_buffer is particles.buffer:
            return
        self.particle_vao = self.ctx.vertex_array(
            self.particle_prog,
            [(particles.buffer, "3f", "in_pos"), (particles.seed_buffer, "1f", "in_seed")],
        )
        self.last_buffer = particles.buffer

    def _fit(self, x, y, w, h):
        """Letterbox the camera aspect ratio into a pane so nothing is stretched."""
        cam = self.width / self.height
        if w / max(h, 1) > cam:
            vh, vw = h, int(h * cam)
        else:
            vw, vh = w, int(w / cam)
        return (x + (w - vw) // 2, y + (h - vh) // 2, max(vw, 1), max(vh, 1))

    def render(self, frame_left, frame_right, particles, scene, fb_size):
        """frame_right=None -> result pane shows ONLY the AR object on a dark background."""
        fbw, fbh = fb_size
        if fbw < 2 or fbh < 2:
            return
        self._ensure_particle_vao(particles)
        self._upload(self.tex_left, frame_left)
        if frame_right is not None:
            self._upload(self.tex_right, frame_right)

        ctx = self.ctx
        ctx.screen.use()
        ctx.viewport = (0, 0, fbw, fbh)
        ctx.clear(*BG_COLOR)

        half = fbw // 2
        left_vp = self._fit(0, 0, half, fbh)
        right_vp = self._fit(half, 0, fbw - half, fbh)

        ctx.disable(moderngl.BLEND)
        ctx.viewport = left_vp
        self.tex_left.use(0)
        self.quad_vao.render(moderngl.TRIANGLE_STRIP)

        ctx.viewport = right_vp
        if frame_right is not None:
            self.tex_right.use(0)
            self.quad_vao.render(moderngl.TRIANGLE_STRIP)

        # Particles: same viewport as the right camera pane, so coordinates line up.
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        p = self.particle_prog
        p["u_time"] = float(scene.effect_time)
        p["u_mode"] = float(scene.mode_value)
        p["u_shockwave"] = float(scene.shockwave_strength)
        p["u_transition"] = float(scene.transition)
        p["u_rotation"] = float(scene.rotation)
        p["u_anchor"] = tuple(float(v) for v in scene.anchor)
        p["u_scale"] = float(scene.hand_scale)
        p["u_aspect"] = float(self.width / self.height)
        p["u_point_size"] = float(self.cfg["particles"]["point_size"])
        p["u_alpha"] = float(self.cfg["render"]["particle_alpha"])
        p["u_glow"] = float(self.cfg["render"]["glow_strength"])
        pname, pcols = PALETTE_LIST[scene.palette_index % PALETTE_COUNT]
        if pname == "auto":
            pcols = PALETTES.get(scene.current_object, PALETTES["bigben"])
        elif pname == "custom":
            pcols = self.custom
        rainbow = pname == "rainbow"
        col_a, col_b = pcols if pcols else ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
        p["u_col_a"] = col_a
        p["u_col_b"] = col_b
        p["u_rainbow"] = 1.0 if rainbow else 0.0
        p["u_hue"] = float(scene.rotation * 0.08)   # rainbow slowly drifts

        self.particle_vao.render(mode=moderngl.POINTS)

        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        ctx.disable(moderngl.BLEND)

    def release(self):
        for obj in [self.particle_vao, self.quad_vao, self.quad_vbo,
                    self.tex_left, self.tex_right, self.particle_prog, self.bg_prog]:
            try:
                if obj is not None:
                    obj.release()
            except Exception:
                pass