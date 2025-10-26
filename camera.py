import cv2
import numpy as np
import mediapipe as mp
import time

# -------------------- Settings --------------------
CAM_INDEX = 0
MAX_HANDS = 1
MIN_DET_CONF = 0.7
MIN_TRACK_CONF = 0.7

# Colors: BGR tuples
PALETTE = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 255, 255) # White (optional)
]
PALETTE_NAMES = ["Red", "Green", "Blue", "Yellow", "White"]
BRUSH_THICKNESS = 8
ERASER_THICKNESS = 60
SMOOTHING = 0.7  # 0..1 -> higher = smoother but more lag

# -------------------- MediaPipe init --------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=MAX_HANDS,
    min_detection_confidence=MIN_DET_CONF,
    min_tracking_confidence=MIN_TRACK_CONF
)

# -------------------- App state --------------------
cap = cv2.VideoCapture(CAM_INDEX)
canvas = None

color_index = 0
draw_color = PALETTE[color_index]
mode = "Idle"
prev_mode = None
prev_x, prev_y = 0, 0
color_changed = False  # prevents continuous cycling while hand stays open

# -------------------- Helpers --------------------
def fingers_up(hand_landmarks, hand_label):
    """
    Returns list of 5 ints (1 = finger up, 0 = down) for
    [thumb, index, middle, ring, pinky]
    Uses hand_label ("Right"/"Left") to handle mirrored thumb logic.
    """
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb: compare tip (4) with ip (3) on x axis; direction depends on hand label
    if hand_label == "Right":
        fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x else 0)
    else:
        fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 1].x else 0)

    # Other fingers: tip.y < pip.y => finger up (remember image is flipped when showing)
    for i in range(1, 5):
        tip_y = hand_landmarks.landmark[tip_ids[i]].y
        pip_y = hand_landmarks.landmark[tip_ids[i] - 2].y
        fingers.append(1 if tip_y < pip_y else 0)

    return fingers

# -------------------- Main loop --------------------
try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Camera not found or can't read frame.")
            break

        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        h, w, _ = frame.shape

        if canvas is None:
            canvas = np.zeros_like(frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # reset some state when mode changes
        if prev_mode != mode:
            prev_x, prev_y = 0, 0
            prev_mode = mode

        if results.multi_hand_landmarks:
            # Use multi_handedness to get "Right"/"Left" label safely
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = "Right"
                if results.multi_handedness and len(results.multi_handedness) > idx:
                    hand_label = results.multi_handedness[idx].classification[0].label

                lm = hand_landmarks.landmark
                fingers = fingers_up(hand_landmarks, hand_label)
                total_fingers = fingers.count(1)

                # index finger tip coords (landmark 8)
                ix = int(lm[8].x * w)
                iy = int(lm[8].y * h)

                # --- Mode mapping according to user's requested mapping ---
                # 1 -> Pointer, 2 -> Draw, 3 -> Erase, 4 -> Clear, 5 -> Change color
                if total_fingers == 1:
                    mode = "Pointer"
                    color_changed = False
                elif total_fingers == 2:
                    mode = "Draw"
                    color_changed = False
                elif total_fingers == 3:
                    mode = "Erase"
                    color_changed = False
                elif total_fingers == 4:
                    mode = "Clear"
                    color_changed = False
                elif total_fingers == 5:
                    mode = "Color"
                    # only cycle color once per full-open gesture
                    if not color_changed:
                        color_index = (color_index + 1) % len(PALETTE)
                        draw_color = PALETTE[color_index]
                        color_changed = True
                else:
                    mode = "Idle"
                    color_changed = False

                # Smoothing pointer to reduce jitter
                if prev_x == 0 and prev_y == 0:
                    smooth_x, smooth_y = ix, iy
                else:
                    smooth_x = int(prev_x * SMOOTHING + ix * (1 - SMOOTHING))
                    smooth_y = int(prev_y * SMOOTHING + iy * (1 - SMOOTHING))

                # --- Actions per mode ---
                if mode == "Draw":
                    # draw circle for feedback
                    cv2.circle(frame, (ix, iy), 8, draw_color, -1)
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = smooth_x, smooth_y
                    cv2.line(canvas, (prev_x, prev_y), (smooth_x, smooth_y), draw_color, BRUSH_THICKNESS)
                    prev_x, prev_y = smooth_x, smooth_y

                elif mode == "Erase":
                    cv2.circle(frame, (ix, iy), 30, (0, 0, 0), -1)
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = smooth_x, smooth_y
                    cv2.line(canvas, (prev_x, prev_y), (smooth_x, smooth_y), (0, 0, 0), ERASER_THICKNESS)
                    prev_x, prev_y = smooth_x, smooth_y

                elif mode == "Pointer":
                    cv2.circle(frame, (ix, iy), 15, (0, 255, 0), -1)
                    prev_x, prev_y = 0, 0

                elif mode == "Clear":
                    # Clear the canvas immediately
                    canvas = np.zeros_like(frame)
                    prev_x, prev_y = 0, 0

                elif mode == "Color":
                    # show the color name on screen while in Color mode
                    cv2.putText(frame, f"Color: {PALETTE_NAMES[color_index]}", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, draw_color, 2)
                    prev_x, prev_y = 0, 0

                else:
                    prev_x, prev_y = 0, 0

                # draw hand landmarks for feedback
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Overlay canvas: non-black pixels from canvas are applied to frame
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
        out = cv2.add(frame_bg, canvas_fg)

        # HUD: Mode and Color
        cv2.putText(out, f"Mode: {mode}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(out, f"Color: {PALETTE_NAMES[color_index]}", (300, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, draw_color, 2)

        cv2.imshow("Gesture Paint", out)

        key = cv2.waitKey(1) & 0xFF
        # Press ESC to exit
        if key == 27:
            break
        # Optional: press 's' to save the canvas (press if you want to keep a copy)
        if key == ord('s'):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"gesture_paint_{timestamp}.png"
            cv2.imwrite(filename, canvas)
            print(f"Saved {filename}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
