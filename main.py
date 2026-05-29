import time

import cv2
import mediapipe as mp

from gesture import (
    GESTURE_FIST,
    GESTURE_NONE,
    GESTURE_OPEN_PALM,
    GESTURE_THUMB_LEFT,
    GESTURE_THUMB_RIGHT,
    classify_static,
)
from spotify import next_track, pause, play, previous_track

ACTIONS = {
    GESTURE_OPEN_PALM:   ("Play",       play),
    GESTURE_FIST:        ("Pause",      pause),
    GESTURE_THUMB_RIGHT: ("Next Track", next_track),
    GESTURE_THUMB_LEFT:  ("Prev Track", previous_track),
}

LABELS = {
    GESTURE_NONE:        "None",
    GESTURE_OPEN_PALM:   "Open Palm",
    GESTURE_FIST:        "Fist",
    GESTURE_THUMB_RIGHT: "Thumb Right",
    GESTURE_THUMB_LEFT:  "Thumb Left",
}


def draw_text(frame, text, pos, color=(0, 255, 0), scale=0.75, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def get_action(detected, prev_static):
    if detected == GESTURE_NONE or detected == prev_static:
        return None
    return ACTIONS.get(detected)


def draw_ui(frame, detected, last_action_label, last_action_time, now):
    draw_text(frame, f"Gesture: {LABELS.get(detected, detected)}", (10, 35))
    if last_action_label and now - last_action_time < 2.0:
        draw_text(frame, f"-> {last_action_label}", (10, 70), color=(0, 200, 255))


def main():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(0)

    last_action_time = 0.0
    last_action_label = ""
    prev_static = GESTURE_NONE

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            detected = classify_static(results.multi_hand_landmarks[0].landmark)
        else:
            detected = GESTURE_NONE
            prev_static = GESTURE_NONE

        now = time.monotonic()

        action = get_action(detected, prev_static)
        if action:
            label, fn = action
            print(f"[action] {label}")
            fn()
            last_action_time = now
            last_action_label = label

        prev_static = detected

        draw_ui(frame, detected, last_action_label, last_action_time, now)
        cv2.imshow("Cuey", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
