GESTURE_NONE = "none"
GESTURE_OPEN_PALM = "open_palm"
GESTURE_FIST = "fist"
GESTURE_THUMB_RIGHT = "thumb_right"
GESTURE_THUMB_LEFT = "thumb_left"


def fingers_up(lm):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return [lm[tip].y < lm[pip].y for tip, pip in zip(tips, pips)]


def classify_static(lm):
    index, middle, ring, pinky = fingers_up(lm)

    if index and middle and ring and pinky:
        return GESTURE_OPEN_PALM

    if not index and not middle and not ring and not pinky:
        # Thumb direction: compare tip (4) to MCP (2)
        dx = lm[4].x - lm[2].x
        dy = lm[4].y - lm[2].y
        if abs(dx) > abs(dy) and abs(dx) > 0.06:
            return GESTURE_THUMB_RIGHT if dx > 0 else GESTURE_THUMB_LEFT
        return GESTURE_FIST

    return GESTURE_NONE
