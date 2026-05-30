import os
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
    PointerSwipeDetector,
    StaticGestureGate,
    classify_static,
)
from spotify import next_track, pause, play, previous_track

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


def draw_text(frame, text, pos, color=(0, 255, 0), scale=0.75, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def get_action(detected, listening):
    if detected == GESTURE_NONE:
        return None
    if detected == GESTURE_PEACE:
        return GESTURE_PEACE  # sentinel: toggle listening
    if not listening:
        return None
    return ACTIONS.get(detected)


def draw_ui(frame, detected, last_action_label, last_action_time, now, listening):
    status = "ON" if listening else "OFF"
    status_color = (0, 255, 0) if listening else (0, 0, 255)
    draw_text(frame, f"Listening: {status}", (10, 35), color=status_color)
    draw_text(frame, f"Gesture: {LABELS.get(detected, detected)}", (10, 70))
    if last_action_label and now - last_action_time < 2.0:
        draw_text(frame, f"-> {last_action_label}", (10, 105), color=(0, 200, 255))


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
    swipe_detector = PointerSwipeDetector()
    static_gate = StaticGestureGate()

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

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detected = classify_static(hand_landmarks.landmark)
            if listening:
                swipe = swipe_detector.update(hand_landmarks.landmark, now)
            else:
                swipe_detector.reset()
            motion_blocked = swipe != GESTURE_NONE or swipe_detector.is_motion_active(now)
            stable_static = static_gate.update(detected, now, blocked=motion_blocked)
        else:
            detected = GESTURE_NONE
            swipe_detector.reset()
            static_gate.reset()

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
        draw_ui(frame, visible_gesture, last_action_label, last_action_time, now, listening)
        cv2.imshow("Cuey", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
