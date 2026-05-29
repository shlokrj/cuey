import time
from collections import deque

GESTURE_NONE = "none"
GESTURE_OPEN_PALM = "open_palm"
GESTURE_FIST = "fist"
GESTURE_SWIPE_RIGHT = "swipe_right"
GESTURE_SWIPE_LEFT = "swipe_left"


def fingers_up(lm):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return [lm[tip].y < lm[pip].y for tip, pip in zip(tips, pips)]


def classify_static(lm):
    index, middle, ring, pinky = fingers_up(lm)

    if index and middle and ring and pinky:
        return GESTURE_OPEN_PALM

    if not index and not middle and not ring and not pinky:
        return GESTURE_FIST

    return GESTURE_NONE


class SwipeDetector:
    """Detects left/right wrist motion over a rolling time window."""

    def __init__(self, window: float = 0.45, threshold: float = 0.18):
        self._pts: deque = deque()
        self._window = window
        self._threshold = threshold

    def update(self, x: float):
        now = time.monotonic()
        self._pts.append((now, x))
        cutoff = now - self._window
        while self._pts and self._pts[0][0] < cutoff:
            self._pts.popleft()

    def detect(self) -> str:
        if len(self._pts) < 4:
            return GESTURE_NONE
        xs = [p[1] for p in self._pts]
        delta = xs[-1] - xs[0]
        if delta > self._threshold:
            return GESTURE_SWIPE_RIGHT
        if delta < -self._threshold:
            return GESTURE_SWIPE_LEFT
        return GESTURE_NONE

    def reset(self):
        self._pts.clear()
