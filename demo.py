import cv2
import numpy as np
import time
import serial
from collections import deque
from picamera2 import Picamera2
from scipy.signal import savgol_filter

# -----------------------------------------
# USER SETTINGS
# -----------------------------------------
SMOOTHING_WINDOW = 7   # More smoothing
OUTLIER_THRESHOLD = 20 # deg jump allowed
PENDULUM_LENGTH_M = 1.70

GREEN_LOWER = np.array([40, 40, 40], dtype=np.uint8)
GREEN_UPPER = np.array([85, 255, 255], dtype=np.uint8)

last_angle = None
last_time = time.time()

angle_buffer = deque(maxlen=SMOOTHING_WINDOW)

# -----------------------------------------
# SERIAL SETUP
# -----------------------------------------
SERIAL_PORT = "/dev/ttyACM0"
BAUD = 115200

try:
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
    print(f"[OK] Serial connected: {SERIAL_PORT}")
except:
    print("[WARNING] Serial not available")
    ser = None


# -----------------------------------------
# CAMERA INITIALISATION
# -----------------------------------------
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (320, 240)}))
picam2.start()
time.sleep(1)

print("Camera started. Detecting TWO green blobs...\n")

# -----------------------------------------
# FUNCTION: Detect two blobs
# -----------------------------------------
def detect_two_green_blobs(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    mask = cv2.GaussianBlur(mask, (7,7), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        return None, None, mask

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    centers = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))

    if len(centers) < 2:
        return None, None, mask

    # Pivot = top one
    pivot = min(centers, key=lambda p: p[1])
    bob   = max(centers, key=lambda p: p[1])

    return pivot, bob, mask


# -----------------------------------------
# STAGE 1: 3-POINT CALIBRATION
# -----------------------------------------
calib_angles = []
calib_speeds = []

print("=== CALIBRATION ===")
print("At each of 3 angles: hold pendulum still, press ENTER, then enter TRUE speed.\n")

for i in range(3):
    print(f"Calibration point {i+1}/3 — hold position and press ENTER.")
    while True:
        frame = picam2.capture_array()
        pivot, bob, mask = detect_two_green_blobs(frame)
        cv2.imshow("Mask", mask)

        if pivot and bob:
            cv2.circle(frame, pivot, 6, (255,255,0), -1)
            cv2.circle(frame, bob, 6, (0,255,0), -1)
            cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 13:  # ENTER = confirm angle
            break

    if pivot is None or bob is None:
        print("Could not detect two blobs — try again.")
        continue

    dx = bob[0] - pivot[0]
    dy = bob[1] - pivot[1]
    angle = np.degrees(np.arctan2(dx, dy))
    print(f"Detected angle = {angle:.2f}°")

    true_speed = float(input("Enter true speed at this angle (m/s): "))
    calib_angles.append(angle)
    calib_speeds.append(true_speed)

# ✔ Fit 2nd-order polynomial (non-linear mapping)
calib_angles = np.array(calib_angles)
calib_speeds = np.array(calib_speeds)
a, b, c = np.polyfit(calib_angles, calib_speeds, 2)

print("\n=== CALIBRATION COMPLETE ===")
print(f"Model: speed = {a:.5f}*θ² + {b:.5f}*θ + {c:.5f}\n")
time.sleep(1)


# -----------------------------------------
# REAL-TIME LOOP
# -----------------------------------------
print("=== REAL-TIME TRACKING STARTED ===\n")

while True:
    frame = picam2.capture_array()
    pivot, bob, mask = detect_two_green_blobs(frame)
    display = frame.copy()

    if pivot and bob:

        cv2.circle(display, pivot, 6, (255,255,0), -1)
        cv2.circle(display, bob, 6, (0,255,0), -1)
        cv2.line(display, pivot, bob, (255,0,0), 2)

        dx = bob[0] - pivot[0]
        dy = bob[1] - pivot[1]
        angle = np.degrees(np.arctan2(dx, dy))

        # Noise rejection — remove spikes
        if last_angle is not None:
            if abs(angle - last_angle) > OUTLIER_THRESHOLD:
                angle = last_angle

        last_angle = angle
        angle_buffer.append(angle)

        # Smooth angle using Savitzky–Golay
        if len(angle_buffer) >= 5:
            smooth_angle = savgol_filter(list(angle_buffer), 5, 2)[-1]
        else:
            smooth_angle = np.mean(angle_buffer)

        # Apply NON-LINEAR model
        speed = a * smooth_angle**2 + b * smooth_angle + c

        # Never negative
        speed = max(speed, 0)

        # OUTPUT
        print(f"{speed:.3f} m/s")

        if ser:
            try:
                ser.write(f"{speed:.3f}\n".encode())
            except:
                pass

        cv2.putText(display, f"angle={smooth_angle:.1f} deg", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),2)
        cv2.putText(display, f"v={speed:.2f} m/s", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

    cv2.imshow("Mask", mask)
    cv2.imshow("Tracking", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.close()
cv2.destroyAllWindows()
