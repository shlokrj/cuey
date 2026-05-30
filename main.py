import json
import os
from pathlib import Path
from statistics import median
import time

# Must be set before importing mediapipe to suppress absl/glog noise
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import mediapipe as mp

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
    GESTURE_NONE:        "None",
    GESTURE_OPEN_PALM:   "Open Palm",
    GESTURE_FIST:        "Fist",
    GESTURE_PEACE:       "Peace",
    GESTURE_POINT:       "Point",
    GESTURE_SWIPE_RIGHT: "Swipe Right",
    GESTURE_SWIPE_LEFT:  "Swipe Left",
    GESTURE_VOLUME_TOGGLE: "Volume Sign",
    GESTURE_VOLUME_UP:   "Volume Up",
    GESTURE_VOLUME_DOWN: "Volume Down",
}

# ── UI palette (BGR) ──────────────────────────────────────────────────────
C_LAVENDER   = (255, 185, 225)
C_PERIWINKLE = (255, 180, 180)
C_MINT       = (205, 255, 185)
C_PINK       = (210, 182, 255)
C_ICE        = (255, 225, 190)
C_WHITE      = (245, 245, 255)
C_DIM        = (200, 190, 215)
C_PANEL_BG   = (48,  28,  58)
C_BORDER     = (195, 140, 180)
_FONT        = cv2.FONT_HERSHEY_DUPLEX


def draw_text(frame, text, pos, color=C_WHITE, scale=0.46, thickness=1):
    cv2.putText(frame, text, pos, _FONT, scale, color, thickness)


def draw_help_panel(frame):
    PAD = 14
    TH = 1
    SZ_TITLE = 0.54
    SZ_SUB   = 0.35
    SZ_SECT  = 0.39
    SZ_ROW   = 0.40
    SZ_KEY   = 0.37
    ROW_H    = 22

    gesture_rows = [
        ("Open Palm",    "Play"),
        ("Fist",         "Pause"),
        ("Swipe Right",  "Next"),
        ("Swipe Left",   "Prev"),
        ("Hold Edge",    "Repeat skip"),
        ("Index+Pinky",  "Vol mode"),
        ("Thumbs Up/Dn", "Vol +/-"),
        ("Peace",        "Listen toggle"),
    ]
    key_rows = [
        "C calibrate   H hide",
        "L listen      Q quit",
        "V volume mode",
    ]

    all_sized = (
        [("CUEY", SZ_TITLE), ("gesture controller", SZ_SUB),
         ("GESTURES", SZ_SECT), ("KEYS", SZ_SECT)]
        + [(f"{g}    {a}", SZ_ROW) for g, a in gesture_rows]
        + [(r, SZ_KEY) for r in key_rows]
    )
    pw = max(cv2.getTextSize(t, _FONT, s, TH)[0][0] for t, s in all_sized) + PAD * 2

    ph = (
        PAD + 24 + 16 + 10
        + 2 + 10
        + 18 + 6
        + len(gesture_rows) * ROW_H
        + 10 + 2 + 10
        + 18 + 6
        + len(key_rows) * 18
        + PAD
    )

    fh, fw = frame.shape[:2]
    x1 = fw - pw - 10
    y1 = 10
    x2 = fw - 10
    y2 = min(fh - 10, y1 + ph)

    ov = frame.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), C_PANEL_BG, -1)
    cv2.addWeighted(ov, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), C_BORDER, 1)

    def put(text, y, color, sz):
        cv2.putText(frame, text, (x1 + PAD, y), _FONT, sz, color, TH)

    def divider(y):
        cv2.line(frame, (x1 + PAD, y), (x2 - PAD, y), C_BORDER, 1)

    y = y1 + PAD + 18
    put("CUEY", y, C_LAVENDER, SZ_TITLE)
    y += 16
    put("gesture controller", y, C_DIM, SZ_SUB)
    y += 12
    divider(y); y += 12

    put("GESTURES", y, C_MINT, SZ_SECT)
    y += 22
    for gesture, action in gesture_rows:
        lw = cv2.getTextSize(gesture + "  ", _FONT, SZ_ROW, TH)[0][0]
        cv2.putText(frame, gesture, (x1 + PAD, y), _FONT, SZ_ROW, C_ICE, TH)
        cv2.putText(frame, "->  " + action, (x1 + PAD + lw, y), _FONT, SZ_ROW, C_PERIWINKLE, TH)
        y += ROW_H

    y += 6
    divider(y); y += 12

    put("KEYS", y, C_MINT, SZ_SECT)
    y += 22
    for row in key_rows:
        put(row, y, C_DIM, SZ_KEY)
        y += 18


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
        return GESTURE_PEACE  # sentinel: toggle listening
    if not listening:
        return None
    if detected == GESTURE_VOLUME_TOGGLE:
        return GESTURE_VOLUME_TOGGLE  # sentinel: toggle volume mode
    if volume_mode:
        return None
    return ACTIONS.get(detected)


def draw_ui(frame, detected, last_action_label, last_action_time, now, listening, volume_mode, hand_present, hand_close):
    PAD = 10
    TH = 1
    SZ = 0.44
    ROW_H = 22

    rows = []
    if listening:
        rows.append(("  Listening  ON ", C_MINT))
    else:
        rows.append(("  Listening  OFF", C_PINK))

    mode_col = C_PERIWINKLE if volume_mode else C_LAVENDER
    rows.append((f"  Mode  {'Volume' if volume_mode else 'Playback'}", mode_col))

    rows.append((f"  {LABELS.get(detected, detected)}", C_WHITE))

    if hand_present:
        if hand_close:
            rows.append(("  Distance  OK", C_MINT))
        else:
            rows.append(("  Move closer", C_ICE))

    if last_action_label and now - last_action_time < 2.0:
        rows.append((f"  -> {last_action_label}", C_PERIWINKLE))

    widths = [cv2.getTextSize(t, _FONT, SZ, TH)[0][0] for t, _ in rows]
    pw = max(widths) + PAD * 2 if widths else 180
    ph = len(rows) * ROW_H + PAD * 2

    x1, y1 = 10, 10
    x2, y2 = x1 + pw, y1 + ph

    ov = frame.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), C_PANEL_BG, -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), C_BORDER, 1)

    for i, (text, color) in enumerate(rows):
        y = y1 + PAD + 16 + i * ROW_H
        cv2.putText(frame, text, (x1, y), _FONT, SZ, color, TH)


def draw_calibration_ui(frame, calibration_active, calibration_progress, calibration_message, calibration_message_time, now):
    fh, _ = frame.shape[:2]
    if calibration_active:
        cv2.putText(frame, "Calibrating: hold hand at desired distance", (10, fh - 58), _FONT, 0.44, C_LAVENDER, 1)
        bx, by = 10, fh - 34
        bw, bh = 260, 10
        filled = int(bw * max(0.0, min(calibration_progress, 1.0)))
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), C_PANEL_BG, -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), C_BORDER, 1)
        if filled:
            cv2.rectangle(frame, (bx, by), (bx + filled, by + bh), C_LAVENDER, -1)
    elif calibration_message and now - calibration_message_time < 3.0:
        cv2.putText(frame, calibration_message, (10, fh - 28), _FONT, 0.44, C_MINT, 1)


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
