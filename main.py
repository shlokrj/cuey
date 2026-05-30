import json
import os
from pathlib import Path
from statistics import median
import time

os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gesture import (
    GESTURE_FIST,
    GESTURE_NONE,
    GESTURE_OPEN_PALM,
    GESTURE_PEACE,
    GESTURE_POINT,
    GESTURE_SWIPE_LEFT,
    GESTURE_SWIPE_RIGHT,
    GESTURE_VOLUME_DOWN,
    GESTURE_VOLUME_TOGGLE,
    GESTURE_VOLUME_UP,
    HandProximityGate,
    PointerSwipeDetector,
    StaticGestureGate,
    VolumeHoldDetector,
    classify_static,
    hand_size,
)
from spotify import next_track, pause, play, previous_track, volume_down, volume_up

CALIBRATION_PATH = Path(__file__).resolve().parent / ".cuey_calibration.json"
CALIBRATION_SECONDS = 1.6
MIN_CALIBRATION_SAMPLES = 10

ACTIONS = {
    GESTURE_OPEN_PALM:   ("Play",       play),
    GESTURE_FIST:        ("Pause",      pause),
    GESTURE_SWIPE_RIGHT: ("Next Track", next_track),
    GESTURE_SWIPE_LEFT:  ("Prev Track", previous_track),
}

VOLUME_ACTIONS = {
    GESTURE_VOLUME_UP:   ("Volume Up",   volume_up),
    GESTURE_VOLUME_DOWN: ("Volume Down", volume_down),
}

LABELS = {
    GESTURE_NONE:          "None",
    GESTURE_OPEN_PALM:     "Open Palm",
    GESTURE_FIST:          "Fist",
    GESTURE_PEACE:         "Peace",
    GESTURE_POINT:         "Point",
    GESTURE_SWIPE_RIGHT:   "Swipe Right",
    GESTURE_SWIPE_LEFT:    "Swipe Left",
    GESTURE_VOLUME_TOGGLE: "Volume Sign",
    GESTURE_VOLUME_UP:     "Volume Up",
    GESTURE_VOLUME_DOWN:   "Volume Down",
}

# ── palette — all blues & purples, stored as RGB for PIL ─────────────────
# cv2 calls receive these reversed (BGR) via the _bgr() helper below
C_LAVENDER   = (210, 185, 255)   # soft lavender
C_PERIWINKLE = (165, 160, 255)   # blue-purple periwinkle
C_LILAC      = (230, 215, 255)   # pale lilac
C_VIOLET     = (190, 150, 240)   # deeper violet
C_BLUE_PURP  = (195, 210, 255)   # ice blue-lavender
C_DIM        = (175, 165, 210)   # muted purple-grey
C_WHITE      = (248, 244, 255)   # purple-tinted white
C_PANEL_BG   = (22,  12,  40)    # deep purple-black (RGB)
C_BORDER     = (140, 115, 215)   # medium purple

_FONT_PATH = "/System/Library/Fonts/SFNS.ttf"
_font_cache: dict = {}


def _font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_cache:
        try:
            _font_cache[size] = ImageFont.truetype(_FONT_PATH, size)
        except OSError:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def _tw(text: str, size: int) -> int:
    bb = _font(size).getbbox(text)
    return bb[2] - bb[0]


def _bgr(rgb: tuple) -> tuple:
    return (rgb[2], rgb[1], rgb[0])


class _TextLayer:
    """Collects text draws and flushes all at once via PIL for clean font rendering."""

    def __init__(self):
        self._q: list = []

    def put(self, text: str, xy: tuple, color_rgb: tuple, size: int):
        self._q.append((text, xy, color_rgb, size))

    def flush(self, frame):
        if not self._q:
            return
        pil = Image.fromarray(frame[:, :, ::-1])   # BGR -> RGB
        draw = ImageDraw.Draw(pil)
        for text, xy, color, size in self._q:
            draw.text(xy, text, font=_font(size), fill=color)
        frame[:] = np.array(pil)[:, :, ::-1]       # RGB -> BGR
        self._q.clear()


_tl = _TextLayer()


def draw_watermark(frame):
    fh, fw = frame.shape[:2]
    text = "CUEY"
    sz = 26
    w = _tw(text, sz)
    _tl.put(text, (fw - w - 14, fh - sz - 12), C_DIM, sz)


def draw_help_panel(frame):
    PAD      = 14
    SZ_TITLE = 22
    SZ_SUB   = 13
    SZ_SECT  = 13
    SZ_ROW   = 14
    SZ_KEY   = 13

    gesture_rows = [
        ("Open Palm",    "Play"),
        ("Fist",         "Pause"),
        ("Swipe Right",  "Next"),
        ("Swipe Left",   "Prev"),
        ("Hold Edge",    "Repeat"),
        ("Index+Pinky",  "Vol mode"),
        ("Thumbs Up/Dn", "Vol +/-"),
        ("Peace",        "Listen"),
    ]
    key_rows = [
        "C calibrate   H hide",
        "L listen      Q quit",
        "V volume mode",
    ]

    max_row_w = max(_tw(g + "     ->  " + a, SZ_ROW) for g, a in gesture_rows)
    pw = max(
        _tw("CUEY", SZ_TITLE),
        _tw("gesture controller", SZ_SUB),
        _tw("GESTURES", SZ_SECT),
        _tw("KEYS", SZ_SECT),
        max_row_w,
        max(_tw(r, SZ_KEY) for r in key_rows),
    ) + PAD * 2

    ph = (
        PAD + 26 + 6 + 15 + 10
        + 1 + 10
        + 16 + 6
        + len(gesture_rows) * 21
        + 8 + 1 + 10
        + 16 + 6
        + len(key_rows) * 19
        + PAD
    )

    fh, fw = frame.shape[:2]
    x1 = fw - pw - 10
    y1 = 10
    x2 = fw - 10
    y2 = min(fh - 10, y1 + ph)

    ov = frame.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), _bgr(C_PANEL_BG), -1)
    cv2.addWeighted(ov, 0.84, frame, 0.16, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), _bgr(C_BORDER), 1)

    tx = x1 + PAD
    y = y1 + PAD

    _tl.put("CUEY", (tx, y), C_LAVENDER, SZ_TITLE)
    y += 26 + 6
    _tl.put("gesture controller", (tx, y), C_DIM, SZ_SUB)
    y += 15 + 10
    cv2.line(frame, (x1 + PAD, y), (x2 - PAD, y), _bgr(C_BORDER), 1)
    y += 1 + 10

    _tl.put("GESTURES", (tx, y), C_PERIWINKLE, SZ_SECT)
    y += 16 + 6

    for gesture, action in gesture_rows:
        gw = _tw(gesture, SZ_ROW)
        gap = _tw("  ", SZ_ROW)
        _tl.put(gesture, (tx, y), C_LILAC, SZ_ROW)
        _tl.put("->  " + action, (tx + gw + gap, y), C_PERIWINKLE, SZ_ROW)
        y += 21

    y += 8
    cv2.line(frame, (x1 + PAD, y), (x2 - PAD, y), _bgr(C_BORDER), 1)
    y += 1 + 10

    _tl.put("KEYS", (tx, y), C_PERIWINKLE, SZ_SECT)
    y += 16 + 6
    for row in key_rows:
        _tl.put(row, (tx, y), C_DIM, SZ_KEY)
        y += 19


def load_calibration():
    try:
        data = json.loads(CALIBRATION_PATH.read_text())
        min_size = float(data["min_size"])
        release_size = float(data["release_size"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None

    if release_size <= 0 or min_size <= 0 or release_size > min_size:
        return None
    return min_size, release_size


def save_calibration(min_size, release_size, calibrated_size):
    data = {
        "min_size": round(min_size, 4),
        "release_size": round(release_size, 4),
        "calibrated_size": round(calibrated_size, 4),
    }
    try:
        CALIBRATION_PATH.write_text(json.dumps(data, indent=2) + "\n")
    except OSError as exc:
        print(f"[calibration] save failed: {exc}")


def apply_distance_calibration(proximity_gate, calibrated_size):
    min_size = max(0.12, calibrated_size * 0.82)
    release_size = max(0.10, calibrated_size * 0.70)
    proximity_gate.set_thresholds(min_size, release_size)
    save_calibration(min_size, release_size, calibrated_size)
    return min_size, release_size


def get_action(detected, listening, volume_mode):
    if detected == GESTURE_NONE:
        return None
    if detected == GESTURE_PEACE:
        return GESTURE_PEACE
    if not listening:
        return None
    if detected == GESTURE_VOLUME_TOGGLE:
        return GESTURE_VOLUME_TOGGLE
    if volume_mode:
        return None
    return ACTIONS.get(detected)


def draw_ui(frame, detected, last_action_label, last_action_time, now, listening, volume_mode, hand_present, hand_close):
    PAD   = 12
    SZ    = 15
    ROW_H = SZ + 9

    rows = []
    if listening:
        rows.append(("Listening  ON", C_LAVENDER))
    else:
        rows.append(("Listening  OFF", C_VIOLET))

    mode_col = C_BLUE_PURP if volume_mode else C_LILAC
    rows.append((f"Mode  {'Volume' if volume_mode else 'Playback'}", mode_col))
    rows.append((LABELS.get(detected, detected), C_WHITE))

    if hand_present:
        if hand_close:
            rows.append(("Distance  OK", C_PERIWINKLE))
        else:
            rows.append(("Move closer", C_DIM))

    if last_action_label and now - last_action_time < 2.0:
        rows.append((f"-> {last_action_label}", C_LAVENDER))

    pw = max(_tw(t, SZ) for t, _ in rows) + PAD * 2 if rows else 180
    ph = len(rows) * ROW_H + PAD * 2

    x1, y1 = 10, 10
    x2, y2 = x1 + pw, y1 + ph

    ov = frame.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), _bgr(C_PANEL_BG), -1)
    cv2.addWeighted(ov, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), _bgr(C_BORDER), 1)

    for i, (text, color) in enumerate(rows):
        _tl.put(text, (x1 + PAD, y1 + PAD + i * ROW_H), color, SZ)


def draw_calibration_ui(frame, calibration_active, calibration_progress, calibration_message, calibration_message_time, now):
    fh, _ = frame.shape[:2]
    if calibration_active:
        _tl.put("Calibrating: hold hand at desired distance", (10, fh - 58), C_LAVENDER, 14)
        bx, by = 10, fh - 34
        bw, bh = 260, 10
        filled = int(bw * max(0.0, min(calibration_progress, 1.0)))
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), _bgr(C_PANEL_BG), -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), _bgr(C_BORDER), 1)
        if filled:
            cv2.rectangle(frame, (bx, by), (bx + filled, by + bh), _bgr(C_LAVENDER), -1)
    elif calibration_message and now - calibration_message_time < 3.0:
        _tl.put(calibration_message, (10, fh - 32), C_PERIWINKLE, 14)


def main():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(0)

    last_action_time = 0.0
    last_action_label = ""
    listening = True
    volume_mode = False
    proximity_gate = HandProximityGate()
    loaded_calibration = load_calibration()
    if loaded_calibration:
        proximity_gate.set_thresholds(*loaded_calibration)
    swipe_detector = PointerSwipeDetector()
    volume_detector = VolumeHoldDetector()
    static_gate = StaticGestureGate()
    show_help = True
    calibration_active = False
    calibration_start = 0.0
    calibration_samples = []
    calibration_message = "Loaded saved distance calibration" if loaded_calibration else ""
    calibration_message_time = time.monotonic() if loaded_calibration else 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.monotonic()
        frame = cv2.flip(frame, 1)
        small = cv2.resize(frame, (320, 240))
        results = hands.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        swipe = GESTURE_NONE
        stable_static = GESTURE_NONE
        hand_present = bool(results.multi_hand_landmarks)
        hand_close = False
        hand_size_value = 0.0

        if hand_present:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = hand_landmarks.landmark
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detected = classify_static(landmarks)
            hand_size_value = hand_size(landmarks)

            if calibration_active:
                calibration_samples.append(hand_size_value)
                hand_close = True
                swipe_detector.reset()
                volume_detector.reset()
                static_gate.reset()
            else:
                hand_close, hand_size_value = proximity_gate.update(landmarks)

            if not calibration_active:
                if not hand_close:
                    swipe_detector.reset()
                    volume_detector.reset()
                    static_gate.reset()
                elif listening:
                    if volume_mode:
                        swipe = volume_detector.update(detected, now)
                        swipe_detector.reset()
                        motion_blocked = swipe != GESTURE_NONE
                    else:
                        swipe = swipe_detector.update(landmarks, now, hand_size_value)
                        volume_detector.reset()
                        motion_blocked = swipe != GESTURE_NONE or swipe_detector.is_motion_active(now)
                    stable_static = static_gate.update(detected, now, blocked=motion_blocked)
                else:
                    swipe_detector.reset()
                    volume_detector.reset()
                    wake_gesture = detected if detected == GESTURE_PEACE else GESTURE_NONE
                    stable_static = static_gate.update(wake_gesture, now)
        else:
            detected = GESTURE_NONE
            proximity_gate.reset()
            swipe_detector.reset()
            volume_detector.reset()
            static_gate.reset()

        if calibration_active and now - calibration_start >= CALIBRATION_SECONDS:
            if len(calibration_samples) >= MIN_CALIBRATION_SAMPLES:
                calibrated_size = median(calibration_samples)
                apply_distance_calibration(proximity_gate, calibrated_size)
                calibration_message = "Distance calibrated"
                print(f"[calibration] hand_size={calibrated_size:.3f}")
            else:
                calibration_message = "Calibration failed: show your hand"
                print("[calibration] failed: not enough hand samples")
            calibration_active = False
            calibration_message_time = now
            calibration_samples.clear()
            swipe_detector.reset()
            volume_detector.reset()
            static_gate.reset()

        if calibration_active:
            action = None
        else:
            if listening and swipe != GESTURE_NONE:
                action = VOLUME_ACTIONS.get(swipe) if volume_mode else ACTIONS.get(swipe)
            else:
                action = None
            if action is None:
                action = get_action(stable_static, listening, volume_mode)

        if action == GESTURE_PEACE:
            listening = not listening
            if not listening:
                volume_mode = False
            swipe_detector.reset()
            volume_detector.reset()
            static_gate.reset()
            last_action_label = f"Listening {'On' if listening else 'Off'}"
            last_action_time = now
            print(f"[toggle] listening={listening}")
        elif action == GESTURE_VOLUME_TOGGLE:
            volume_mode = not volume_mode
            swipe_detector.reset()
            volume_detector.reset()
            static_gate.reset()
            last_action_label = f"Volume Mode {'On' if volume_mode else 'Off'}"
            last_action_time = now
            print(f"[toggle] volume_mode={volume_mode}")
        elif action:
            label, fn = action
            print(f"[action] {label}")
            fn()
            last_action_time = now
            last_action_label = label

        visible_gesture = swipe if swipe != GESTURE_NONE else detected
        calibration_progress = (now - calibration_start) / CALIBRATION_SECONDS if calibration_active else 0.0
        draw_ui(frame, visible_gesture, last_action_label, last_action_time, now, listening, volume_mode, hand_present, hand_close)
        draw_calibration_ui(frame, calibration_active, calibration_progress, calibration_message, calibration_message_time, now)
        if show_help:
            draw_help_panel(frame)
        draw_watermark(frame)
        _tl.flush(frame)
        cv2.imshow("Cuey", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            calibration_active = True
            calibration_start = now
            calibration_samples.clear()
            calibration_message = ""
            proximity_gate.reset()
            swipe_detector.reset()
            volume_detector.reset()
            static_gate.reset()
            print("[calibration] started")
        if key == ord("l"):
            listening = not listening
            if not listening:
                volume_mode = False
            swipe_detector.reset()
            volume_detector.reset()
            static_gate.reset()
            last_action_label = f"Listening {'On' if listening else 'Off'}"
            last_action_time = now
            print(f"[toggle] listening={listening}")
        if key == ord("v") and listening:
            volume_mode = not volume_mode
            swipe_detector.reset()
            volume_detector.reset()
            static_gate.reset()
            last_action_label = f"Volume Mode {'On' if volume_mode else 'Off'}"
            last_action_time = now
            print(f"[toggle] volume_mode={volume_mode}")
        if key == ord("h"):
            show_help = not show_help
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
