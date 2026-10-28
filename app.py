import sys
import time
import yaml
import cv2
import numpy as np
import moderngl
import glfw
import mediapipe as mp

from hand.tracker import HandTracker
from hand.gesture import GestureEngine
from particles.point_cloud import build_point_cloud
from particles.system import ParticleSystem
from scene.scene_manager import SceneManager
from renderer.renderer import Renderer, PALETTE_COUNT, palette_name, palette_index_by_name


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------- camera ---
def _open(index, cam_cfg):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    fourcc = cam_cfg.get("fourcc")
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
    cap.set(cv2.CAP_PROP_FPS, cam_cfg["fps"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _brightness(cap, n=25):
    """Read a few frames (auto-exposure needs time) and return the mean pixel value."""
    mean = -1.0
    for _ in range(n):
        ok, f = cap.read()
        if ok and f is not None:
            mean = float(f[::8, ::8].mean())
    return mean


def open_camera(cfg):
    cam = cfg["camera"]
    thr = float(cam.get("min_brightness", 8))
    first = int(cam["index"])

    cap = _open(first, cam)
    if cap is not None:
        b = _brightness(cap)
        print(f"[camera] index {first}: {int(cap.get(3))}x{int(cap.get(4))}, brightness={b:.1f}")
        if b >= thr or not cam.get("auto_find", True):
            return cap
        print(f"[camera] index {first} gives a BLACK image (IR/metadata node?). Scanning other indices...")
        cap.release()
    else:
        print(f"[camera] cannot open index {first}. Scanning other indices...")

    for i in range(0, 10):
        if i == first:
            continue
        c = _open(i, cam)
        if c is None:
            continue
        b = _brightness(c)
        print(f"[camera] index {i}: {int(c.get(3))}x{int(c.get(4))}, brightness={b:.1f}")
        if b >= thr:
            print(f"[camera] using index {i}  (set camera.index: {i} in config.yaml to skip the scan)")
            return c
        c.release()

    print("[camera] no bright camera found - falling back to the configured index.")
    cap = _open(first, cam)
    if cap is None:
        raise RuntimeError("Cannot open any webcam. Check permissions: ls -l /dev/video*; groups")
    return cap


# -------------------------------------------------------------------- HUD ---
_mp_draw = mp.solutions.drawing_utils
_mp_hands = mp.solutions.hands


def draw_hud(frame, hand, gstate, scene, fps, brightness, debug):
    """Left pane stays clean (just you). Keypoints/text only when debug (key K) is on,
    except the black-camera warning which is always shown."""
    if not debug and brightness >= 8:
        return frame
    img = frame.copy()
    h, w = img.shape[:2]
    s = max(0.5, h / 720.0)
    th = max(1, int(2 * s))

    def txt(text, y, color=(255, 255, 255)):
        cv2.putText(img, text, (12, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.putText(img, text, (12, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, color, th, cv2.LINE_AA)

    if debug:
        if hand.landmarks is not None:
            _mp_draw.draw_landmarks(img, hand.landmarks, _mp_hands.HAND_CONNECTIONS)
            txt(f"GESTURE: {gstate.label}   ext={gstate.ext:.2f}  pinch={gstate.pinch:.2f}", 30 * s, (0, 255, 0))
        else:
            txt("NO HAND DETECTED", 30 * s, (0, 0, 255))
        txt(f"object={scene.current_object}  mode={scene.particle_mode}  size={scene.size_mult:.2f}  fps={fps:.0f}", 60 * s, (255, 220, 120))

    if brightness < 8:
        txt("CAMERA IMAGE IS BLACK - wrong camera index?", h / 2, (0, 0, 255))
        txt("run: python check_camera.py", h / 2 + 40 * s, (0, 0, 255))
    return img


# ------------------------------------------------------------------- main ---
def main():
    cfg = load_config()
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        cfg["camera"]["index"] = int(sys.argv[1])  # python app.py 2

    cap = open_camera(cfg)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or cfg["camera"]["width"]
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or cfg["camera"]["height"]
    aspect = width / height

    if not glfw.init():
        raise RuntimeError("GLFW initialization failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

    # Two panes side by side; shrink the window if it would not fit on the monitor.
    win_w, win_h = width * 2, height
    mode = glfw.get_video_mode(glfw.get_primary_monitor())
    if mode is not None:
        k = min(1.0, mode.size.width * 0.92 / win_w, mode.size.height * 0.85 / win_h)
        win_w, win_h = int(win_w * k), int(win_h * k)

    window = glfw.create_window(win_w, win_h, "WonderSnap | Camera (left) + AR (right)", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Cannot create OpenGL window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    ctx = moderngl.create_context()
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)

    renderer = Renderer(ctx, width, height, cfg)
    tracker = HandTracker(cfg["tracking"].get("model_complexity", 0))
    track_w = int(cfg["tracking"].get("track_width", 480))
    gesture = GestureEngine(cfg["gesture"])
    scene = SceneManager(cfg)
    scene.palette_index = palette_index_by_name(cfg["render"].get("palette", "auto"))

    n = int(cfg["particles"]["count"])
    targets = {name: build_point_cloud(name, n, seed=17 + i) for i, name in enumerate(SceneManager.OBJECTS)}
    particles = ParticleSystem(ctx, targets[scene.current_object], cfg["particles"])
    shown_object = scene.current_object

    pending = []
    ui = {"debug": False, "cam_behind": bool(cfg["render"].get("camera_behind_result", False))}

    def on_key(win, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_SPACE:
            pending.append("SNAP")
        elif key == glfw.KEY_F:
            pending.append("FIST")
        elif key == glfw.KEY_O:
            pending.append("OPEN")
        elif key == glfw.KEY_R:
            scene.reset()
        elif key == glfw.KEY_K:
            ui["debug"] = not ui["debug"]          # keypoints + debug text on the left pane
        elif key == glfw.KEY_B:
            ui["cam_behind"] = not ui["cam_behind"]  # camera image behind the AR object
        elif key == glfw.KEY_C:
            scene.follow = not scene.follow          # follow hand <-> centered
        elif key == glfw.KEY_S:
            scene.fixed_size = not scene.fixed_size  # fixed size <-> size from hand
        elif key == glfw.KEY_P:
            step = -1 if (mods & glfw.MOD_SHIFT) else 1
            scene.palette_index = (scene.palette_index + step) % PALETTE_COUNT   # next/prev colour
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            scene.change_size(1.15)
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            scene.change_size(1 / 1.15)
        elif glfw.KEY_1 <= key < glfw.KEY_1 + len(SceneManager.OBJECTS):
            scene.set_object(key - glfw.KEY_1)

    glfw.set_key_callback(window, on_key)

    last_time = time.perf_counter()
    fps_clock, fps_frames, fps = last_time, 0, 0.0
    brightness, bad_reads = 100.0, 0
    mirror = cfg["render"].get("background_mirror", True)

    print("WonderSnap started.")
    print("OPEN=explode | FIST=build next wonder | SNAP=switch | Q/ESC=quit")
    print("Keys: 1-4 object, SPACE snap, F fist, O open, R reset | P color (Shift+P back), K keypoints, B camera behind result, C center/follow, S fixed/hand size, +/- size")

    try:
        while not glfw.window_should_close(window):
            glfw.poll_events()

            ok, frame = cap.read()
            if not ok or frame is None:
                bad_reads += 1
                if bad_reads > 300:
                    raise RuntimeError("Camera stopped delivering frames.")
                time.sleep(0.005)
                continue
            bad_reads = 0

            now = time.perf_counter()
            dt = min(now - last_time, 0.05)
            last_time = now

            if mirror:
                frame = cv2.flip(frame, 1)
            brightness = 0.9 * brightness + 0.1 * float(frame[::8, ::8].mean())

            small = frame
            if track_w and frame.shape[1] > track_w:
                small = cv2.resize(frame, (track_w, int(track_w / aspect)), interpolation=cv2.INTER_AREA)
            result = tracker.process(small)   # landmarks are normalised -> valid for the full frame too
            gstate = gesture.update(result.landmarks, now, aspect)

            events = list(pending)
            pending.clear()
            if gstate.event:
                events.append(gstate.event)
            for ev in events:
                scene.handle_gesture(ev)

            if result.landmarks is not None:
                scene.update(dt, result.anchor, result.hand_size)
            else:
                scene.update(dt, None, 0.0)

            if scene.current_object != shown_object:
                particles.set_target(targets[scene.current_object])
                shown_object = scene.current_object
            particles.update(dt)

            left = draw_hud(frame, result, gstate, scene, fps, brightness, ui["debug"])
            right = frame if ui["cam_behind"] else None
            renderer.render(left, right, particles, scene, glfw.get_framebuffer_size(window))
            glfw.swap_buffers(window)

            fps_frames += 1
            if now - fps_clock >= 0.5:
                fps = fps_frames / (now - fps_clock)
                fps_frames, fps_clock = 0, now
                hand_txt = gstate.label if result.landmarks is not None else "no hand"
                glfw.set_window_title(
                    window,
                    f"WonderSnap | {scene.current_object} | {hand_txt} | {palette_name(scene.palette_index)} | {fps:.0f} fps | P color  C center/follow  S fixed size  +/- size  K keypoints",
                )
    finally:
        tracker.close()
        cap.release()
        particles.release()
        renderer.release()
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    main()
