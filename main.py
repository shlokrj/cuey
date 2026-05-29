import time

import cv2
import mediapipe as mp

from gesture import (
    GESTURE_FIST,
    GESTURE_NONE,
    GESTURE_OPEN_PALM,
    GESTURE_SWIPE_LEFT,
    GESTURE_SWIPE_RIGHT,
    SwipeDetector,
    classify_static,
)
from spotify import next_track, pause, play, previous_track

COOLDOWN = 1.5  # seconds between any two actions

SWIPE_GESTURES = {GESTURE_SWIPE_LEFT, GESTURE_SWIPE_RIGHT}

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
    GESTURE_SWIPE_RIGHT: "Swipe Right",
    GESTURE_SWIPE_LEFT:  "Swipe Left",
}


def draw_text(frame, text, pos, color=(0, 255, 0), scale=0.75, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


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
    swipe = SwipeDetector()

    last_action_time = 0.0
    last_action_label = ""
    prev_static = GESTURE_NONE

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        detected = GESTURE_NONE

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            lm = hand.landmark
            swipe.update(lm[0].x)

            swipe_result = swipe.detect()
            if swipe_result != GESTURE_NONE:
                detected = swipe_result
            else:
                detected = classify_static(lm)
        else:
            prev_static = GESTURE_NONE

        now = time.monotonic()
        cooled = now - last_action_time >= COOLDOWN

        should_fire = (
            detected != GESTURE_NONE
            and cooled
            and (detected in SWIPE_GESTURES or detected != prev_static)
        )

        if should_fire:
            action = ACTIONS.get(detected)
            if action:
                label, fn = action
                fn()
                last_action_time = now
                last_action_label = label
                swipe.reset()

        if detected not in SWIPE_GESTURES:
            prev_static = detected

        # --- UI ---
        h = frame.shape[0]
        draw_text(frame, f"Gesture: {LABELS.get(detected, detected)}", (10, 35))

        elapsed = now - last_action_time
        if last_action_label and elapsed < 2.0:
            draw_text(frame, f"-> {last_action_label}", (10, 70), color=(0, 200, 255))

        cooldown_left = COOLDOWN - (now - last_action_time)
        if cooldown_left > 0:
            draw_text(
                frame,
                f"Cooldown: {cooldown_left:.1f}s",
                (10, h - 15),
                color=(120, 120, 255),
                scale=0.55,
            )

        cv2.imshow("Cuey", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
