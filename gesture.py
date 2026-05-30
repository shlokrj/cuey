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
WRIST = 0
THUMB = (1, 2, 3, 4)
INDEX = (5, 6, 7, 8)
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)
FINGERS = (INDEX, MIDDLE, RING, PINKY)
MIN_ACTIVE_HAND_SIZE = 0.24
RELEASE_ACTIVE_HAND_SIZE = 0.21


def _dist(lm, a, b):
    return hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)


def _dist_to_point(lm, idx, point):
    return hypot(lm[idx].x - point[0], lm[idx].y - point[1])


def _palm_center(lm):
    points = (WRIST, INDEX[0], MIDDLE[0], RING[0], PINKY[0])
    return (
        sum(lm[idx].x for idx in points) / len(points),
        sum(lm[idx].y for idx in points) / len(points),
    )


def _palm_scale(lm):
    palm_length = _dist(lm, WRIST, MIDDLE[0])
    palm_width = _dist(lm, INDEX[0], PINKY[0])
    return max(palm_length, palm_width, 0.05)


def hand_size(lm):
    min_x = min(point.x for point in lm)
    max_x = max(point.x for point in lm)
    min_y = min(point.y for point in lm)
    max_y = max(point.y for point in lm)
    bbox_size = max(max_x - min_x, max_y - min_y)
    return max(bbox_size, _palm_scale(lm) * 1.8)


def _finger_straightness(lm, joints):
    mcp, pip, dip, tip = joints
    path = _dist(lm, mcp, pip) + _dist(lm, pip, dip) + _dist(lm, dip, tip)
    if path == 0:
        return 0.0
    return _dist(lm, mcp, tip) / path


def _finger_extended(lm, joints):
    mcp, pip, dip, tip = joints
    scale = _palm_scale(lm)
    tip_from_wrist = _dist(lm, WRIST, tip)
    return (
        tip_from_wrist > _dist(lm, WRIST, pip) + 0.10 * scale
        and tip_from_wrist > _dist(lm, WRIST, mcp) + 0.30 * scale
        and tip_from_wrist > _dist(lm, WRIST, dip) + 0.02 * scale
        and _finger_straightness(lm, joints) > 0.72
    )


def _finger_folded(lm, joints):
    _, pip, _, tip = joints
    scale = _palm_scale(lm)
    palm_center = _palm_center(lm)
    tip_near_palm = _dist_to_point(lm, tip, palm_center) < 0.95 * scale
    tip_not_past_pip = _dist(lm, WRIST, tip) < _dist(lm, WRIST, pip) + 0.08 * scale
    return not _finger_extended(lm, joints) and (tip_near_palm or tip_not_past_pip)


def _thumb_extended(lm):
    scale = _palm_scale(lm)
    palm_center = _palm_center(lm)
    thumb_path = _dist(lm, THUMB[0], THUMB[1]) + _dist(lm, THUMB[1], THUMB[2]) + _dist(lm, THUMB[2], THUMB[3])
    if thumb_path == 0:
        return False

    thumb_straightness = _dist(lm, THUMB[0], THUMB[3]) / thumb_path
    return (
        thumb_straightness > 0.68
        and _dist_to_point(lm, THUMB[3], palm_center) > 0.70 * scale
        and _dist(lm, THUMB[3], INDEX[0]) > 0.25 * scale
    )


def _finger_tip_spread(lm, left, right):
    return _dist(lm, left[3], right[3]) / _palm_scale(lm)


def fingers_up(lm):
    return [_finger_extended(lm, finger) for finger in FINGERS]


def is_pointing(lm):
    return _finger_extended(lm, INDEX) and all(_finger_folded(lm, finger) for finger in (MIDDLE, RING, PINKY))


def is_open_palm(lm):
    scale = _palm_scale(lm)
    palm_center = _palm_center(lm)
    extended = [_finger_extended(lm, finger) for finger in FINGERS]
    clear_tips = [
        _dist_to_point(lm, finger[3], palm_center) > 0.72 * scale
        for finger in FINGERS
    ]
    tucked_count = sum(_finger_folded(lm, finger) for finger in FINGERS)
    return sum(extended) >= 3 and sum(clear_tips) >= 3 and tucked_count <= 1


def is_fist(lm):
    palm_center = _palm_center(lm)
    scale = _palm_scale(lm)
    fingertips_tucked = all(_dist_to_point(lm, finger[3], palm_center) < 1.05 * scale for finger in FINGERS)
    return all(_finger_folded(lm, finger) for finger in FINGERS) and fingertips_tucked


def is_peace(lm):
    index_middle_spread = _finger_tip_spread(lm, INDEX, MIDDLE) > 0.18
    folded_ring_pinky = _finger_folded(lm, RING) and _finger_folded(lm, PINKY)
    return _finger_extended(lm, INDEX) and _finger_extended(lm, MIDDLE) and folded_ring_pinky and index_middle_spread


def classify_static(lm):
    if is_peace(lm):
        return GESTURE_PEACE

    if is_open_palm(lm):
        return GESTURE_OPEN_PALM

    if is_pointing(lm):
        return GESTURE_POINT

    if is_fist(lm):
        return GESTURE_FIST

    return GESTURE_NONE


class PointerSwipeDetector:
    def __init__(
        self,
        history_seconds=0.50,
        min_distance=0.18,
        hand_distance_ratio=0.55,
        max_distance=0.28,
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
        self.hand_distance_ratio = hand_distance_ratio
        self.max_distance = max_distance
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

    def update(self, lm, now, hand_size_value=None):
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

        required_distance = self.min_distance
        if hand_size_value is not None:
            required_distance = max(
                required_distance,
                min(hand_size_value * self.hand_distance_ratio, self.max_distance),
            )

        if abs(dx) < required_distance:
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


class HandProximityGate:
    def __init__(
        self,
        min_size=MIN_ACTIVE_HAND_SIZE,
        release_size=RELEASE_ACTIVE_HAND_SIZE,
    ):
        self.min_size = min_size
        self.release_size = release_size
        self.close = False

    def reset(self):
        self.close = False

    def update(self, lm):
        size = hand_size(lm)
        if self.close:
            self.close = size >= self.release_size
        else:
            self.close = size >= self.min_size
        return self.close, size


class StaticGestureGate:
    def __init__(
        self,
        hold_seconds=None,
        cooldown_seconds=None,
    ):
        self.hold_seconds = hold_seconds or {
            GESTURE_OPEN_PALM: 0.25,
            GESTURE_FIST: 0.50,
            GESTURE_PEACE: 0.75,
        }
        self.cooldown_seconds = cooldown_seconds or {
            GESTURE_OPEN_PALM: 0.70,
            GESTURE_FIST: 0.70,
            GESTURE_PEACE: 1.30,
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
