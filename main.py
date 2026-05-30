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
    HandProximityGate,
    PointerSwipeDetector,
    StaticGestureGate,
    classify_static,
    hand_size,
)
from spotify import next_track, pause, play, previous_track

CALIBRATION_PATH = Path(__file__).resolve().parent / ".cuey_calibration.json"
CALIBRATION_SECONDS = 1.6
MIN_CALIBRATION_SAMPLES = 10

ACTIONS = {
    GESTURE_OPEN_PALM:   ("Play",       play),
    GESTURE_FIST:        ("Pause",      pause),
    GESTURE_SWIPE_RIGHT: ("Next Track", next_track),
    GESTURE_SWIPE_LEFT:  ("Prev Track", previous_track),
}

LABELS = {
    GESTURE_NONE:        "None",
    GESTURE_OPEN_PALM:   "Open Palm",
    GESTURE_FIST:        "Fist",
    GESTURE_PEACE:       "Peace",
    GESTURE_POINT:       "Point",
    GESTURE_SWIPE_RIGHT: "Swipe Right",
    GESTURE_SWIPE_LEFT:  "Swipe Left",
}

HELP_LINES = [
    "Commands",
    "Open palm: Play",
    "Fist: Pause",
    "Swipe right: Next",
    "Swipe left: Previous",
    "Hold edge: repeat skip",
    "Peace: Listening on/off",
    "When OFF: peace only",
    "Keep hand close",
    "C: calibrate distance",
    "H: hide help    Q: quit",
]


def draw_text(frame, text, pos, color=(0, 255, 0), scale=0.75, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_help_panel(frame):
    padding = 12
    line_height = 22
    scale = 0.48
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    text_widths = [
        cv2.getTextSize(line, font, scale, thickness)[0][0]
        for line in HELP_LINES
    ]
    panel_width = max(text_widths) + padding * 2
    panel_height = line_height * len(HELP_LINES) + padding
    frame_height, frame_width = frame.shape[:2]
    x1 = max(10, frame_width - panel_width - 10)
    y1 = 10
    x2 = min(frame_width - 10, x1 + panel_width)
    y2 = min(frame_height - 10, y1 + panel_height)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)

    for i, line in enumerate(HELP_LINES):
        y = y1 + padding + 16 + i * line_height
        color = (255, 255, 255)
        if i == 0:
            color = (0, 255, 255)
        elif line.startswith("When OFF") or line.startswith("Keep"):
            color = (180, 230, 255)
        elif line.startswith("H:"):
            color = (180, 180, 180)
        draw_text(frame, line, (x1 + padding, y), color=color, scale=scale, thickness=thickness)


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


def get_action(detected, listening):
    if detected == GESTURE_NONE:
        return None
    if detected == GESTURE_PEACE:
        return GESTURE_PEACE  # sentinel: toggle listening
    if not listening:
        return None
    return ACTIONS.get(detected)


def draw_ui(frame, detected, last_action_label, last_action_time, now, listening, hand_present, hand_close):
    status = "ON" if listening else "OFF"
    status_color = (0, 255, 0) if listening else (0, 0, 255)
    draw_text(frame, f"Listening: {status}", (10, 35), color=status_color)
    draw_text(frame, f"Gesture: {LABELS.get(detected, detected)}", (10, 70))
    action_y = 105
    if hand_present:
        if hand_close:
            draw_text(frame, "Distance: OK", (10, 105), color=(0, 220, 0))
        else:
            draw_text(frame, "Move hand closer", (10, 105), color=(0, 255, 255))
        action_y = 140
    if last_action_label and now - last_action_time < 2.0:
        draw_text(frame, f"-> {last_action_label}", (10, action_y), color=(0, 200, 255))


def draw_calibration_ui(frame, calibration_active, calibration_progress, calibration_message, calibration_message_time, now):
    frame_height, _ = frame.shape[:2]
    if calibration_active:
        y = frame_height - 58
        draw_text(frame, "Calibrating distance: hold hand where you want to control Cuey", (10, y), color=(0, 255, 255), scale=0.55, thickness=1)
        bar_x, bar_y = 10, frame_height - 34
        bar_width, bar_height = 260, 12
        filled = int(bar_width * max(0.0, min(calibration_progress, 1.0)))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (80, 80, 80), 1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_height), (0, 255, 255), -1)
    elif calibration_message and now - calibration_message_time < 3.0:
        draw_text(frame, calibration_message, (10, frame_height - 28), color=(0, 255, 255), scale=0.55, thickness=1)


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
    proximity_gate = HandProximityGate()
    loaded_calibration = load_calibration()
    if loaded_calibration:
        proximity_gate.set_thresholds(*loaded_calibration)
    swipe_detector = PointerSwipeDetector()
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
                static_gate.reset()
            else:
                hand_close, hand_size_value = proximity_gate.update(landmarks)

            if not calibration_active:
                if not hand_close:
                    swipe_detector.reset()
                    static_gate.reset()
                elif listening:
                    swipe = swipe_detector.update(landmarks, now, hand_size_value)
                    motion_blocked = swipe != GESTURE_NONE or swipe_detector.is_motion_active(now)
                    stable_static = static_gate.update(detected, now, blocked=motion_blocked)
                else:
                    swipe_detector.reset()
                    wake_gesture = detected if detected == GESTURE_PEACE else GESTURE_NONE
                    stable_static = static_gate.update(wake_gesture, now)
        else:
            detected = GESTURE_NONE
            proximity_gate.reset()
            swipe_detector.reset()
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
            static_gate.reset()

        if calibration_active:
            action = None
        else:
            action = ACTIONS.get(swipe) if listening and swipe != GESTURE_NONE else None
            if action is None:
                action = get_action(stable_static, listening)

        if action == GESTURE_PEACE:
            listening = not listening
            swipe_detector.reset()
            print(f"[toggle] listening={listening}")
        elif action:
            label, fn = action
            print(f"[action] {label}")
            fn()
            last_action_time = now
            last_action_label = label

        visible_gesture = swipe if swipe != GESTURE_NONE else detected
        calibration_progress = (now - calibration_start) / CALIBRATION_SECONDS if calibration_active else 0.0
        draw_ui(frame, visible_gesture, last_action_label, last_action_time, now, listening, hand_present, hand_close)
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
            static_gate.reset()
            print("[calibration] started")
        if key == ord("h"):
            show_help = not show_help
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
