from collections import deque
from math import hypot


GESTURE_NONE = "none"
GESTURE_OPEN_PALM = "open_palm"
GESTURE_FIST = "fist"
GESTURE_PEACE = "peace"
GESTURE_POINT = "point"
GESTURE_SWIPE_RIGHT = "swipe_right"
GESTURE_SWIPE_LEFT = "swipe_left"

INDEX_TIP = 8


def fingers_up(lm):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return [lm[tip].y < lm[pip].y for tip, pip in zip(tips, pips)]


def is_pointing(lm):
    index, middle, ring, pinky = fingers_up(lm)
    return index and not middle and not ring and not pinky


def classify_static(lm):
    index, middle, ring, pinky = fingers_up(lm)

    if index and middle and not ring and not pinky:
        return GESTURE_PEACE

    if index and middle and ring and pinky:
        return GESTURE_OPEN_PALM

    if is_pointing(lm):
        return GESTURE_POINT

    if not index and not middle and not ring and not pinky:
        return GESTURE_FIST

    return GESTURE_NONE


class PointerSwipeDetector:
    def __init__(
        self,
        history_seconds=0.50,
        min_distance=0.22,
        min_speed=1.0,
        max_vertical_ratio=0.75,
        min_direction_ratio=0.70,
        cooldown_seconds=0.90,
        motion_speed=0.55,
        motion_distance=0.08,
        motion_grace_seconds=0.25,
    ):
        self.history_seconds = history_seconds
        self.min_distance = min_distance
        self.min_speed = min_speed
        self.max_vertical_ratio = max_vertical_ratio
        self.min_direction_ratio = min_direction_ratio
        self.cooldown_seconds = cooldown_seconds
        self.motion_speed = motion_speed
        self.motion_distance = motion_distance
        self.motion_grace_seconds = motion_grace_seconds
        self.positions = deque()
        self.last_swipe_time = -cooldown_seconds
        self.last_motion_time = -motion_grace_seconds

    def reset(self):
        self.positions.clear()

    def is_motion_active(self, now):
        return now - self.last_motion_time < self.motion_grace_seconds

    def update(self, lm, now):
        if now - self.last_swipe_time < self.cooldown_seconds:
            self.reset()
            return GESTURE_NONE

        tip = lm[INDEX_TIP]
        self.positions.append((now, tip.x, tip.y))

        while self.positions and now - self.positions[0][0] > self.history_seconds:
            self.positions.popleft()

        if len(self.positions) < 2:
            return GESTURE_NONE

        start_time, start_x, start_y = self.positions[0]
        elapsed = now - start_time
        if elapsed <= 0.05:
            return GESTURE_NONE

        dx = tip.x - start_x
        dy = tip.y - start_y
        distance = hypot(dx, dy)
        speed = abs(dx) / elapsed
        total_speed = distance / elapsed

        if total_speed >= self.motion_speed or distance >= self.motion_distance:
            self.last_motion_time = now

        if abs(dx) < self.min_distance:
            return GESTURE_NONE
        if speed < self.min_speed:
            return GESTURE_NONE
        if abs(dy) > abs(dx) * self.max_vertical_ratio:
            return GESTURE_NONE

        horizontal_path = 0.0
        previous_x = self.positions[0][1]
        for _, x, _ in list(self.positions)[1:]:
            horizontal_path += abs(x - previous_x)
            previous_x = x

        if horizontal_path == 0.0:
            return GESTURE_NONE
        if abs(dx) / horizontal_path < self.min_direction_ratio:
            return GESTURE_NONE

        self.last_swipe_time = now
        self.reset()
        return GESTURE_SWIPE_RIGHT if dx > 0 else GESTURE_SWIPE_LEFT


class StaticGestureGate:
    def __init__(
        self,
        hold_seconds=None,
        cooldown_seconds=None,
    ):
        self.hold_seconds = hold_seconds or {
            GESTURE_OPEN_PALM: 0.30,
            GESTURE_FIST: 0.40,
            GESTURE_PEACE: 0.65,
        }
        self.cooldown_seconds = cooldown_seconds or {
            GESTURE_OPEN_PALM: 0.70,
            GESTURE_FIST: 0.70,
            GESTURE_PEACE: 1.20,
        }
        self.candidate = GESTURE_NONE
        self.candidate_since = 0.0
        self.fired_candidate = False
        self.last_fired = {}

    def reset(self):
        self.candidate = GESTURE_NONE
        self.candidate_since = 0.0
        self.fired_candidate = False

    def update(self, gesture, now, blocked=False):
        if blocked or gesture not in self.hold_seconds:
            self.reset()
            return GESTURE_NONE

        if gesture != self.candidate:
            self.candidate = gesture
            self.candidate_since = now
            self.fired_candidate = False
            return GESTURE_NONE

        if self.fired_candidate:
            return GESTURE_NONE

        if now - self.candidate_since < self.hold_seconds[gesture]:
            return GESTURE_NONE

        last_fired = self.last_fired.get(gesture, -self.cooldown_seconds[gesture])
        if now - last_fired < self.cooldown_seconds[gesture]:
            return GESTURE_NONE

        self.last_fired[gesture] = now
        self.fired_candidate = True
        return gesture
