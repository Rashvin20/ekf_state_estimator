import cv2
import numpy as np
import time
import serial
from collections import deque

# ==========================================
# USER SETTINGS
# ==========================================
CAMERA_INDEX = 0
pivot = (320, 120)             # Adjust to match your setup
SMOOTHING_WINDOW = 5
SERIAL_PORT = "/dev/ttyACM0"   # Change based on your device
BAUD_RATE = 115200

# ---- GREEN COLOR RANGE (HSV) ----
GREEN_LOWER = np.array([40, 40, 40], dtype=np.uint8)
GREEN_UPPER = np.array([85, 255, 255], dtype=np.uint8)

# ==========================================
# BUFFERS FOR SMOOTHING
# ==========================================
angle_buffer = deque(maxlen=SMOOTHING_WINDOW)
velocity_buffer = deque(maxlen=SMOOTHING_WINDOW)

last_angle = None
last_time = time.time()

# ==========================================
# INITIALISE SERIAL PORT
# ==========================================
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Serial connected: {SERIAL_PORT}")
except Exception as e:
    print(f"WARNING: Serial not available ({e})")
    ser = None

# ==========================================
# INITIALISE CAMERA
# ==========================================
cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(0.5)

if not cap.isOpened():
    print("Camera not found!")
    exit()

print("Camera initialized. Tracking GREEN ball...")

# ==========================================
# MAIN LOOP
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame capture failed")
        break

    display = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask for GREEN ball
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cx = cy = None

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area > 100:
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Draw tracking
            cv2.circle(display, (cx, cy), 6, (0, 255, 0), -1)
            cv2.circle(display, pivot, 6, (255, 255, 0), -1)
            cv2.line(display, pivot, (cx, cy), (255, 255, 0), 2)

            # ==========================================
            # ANGLE COMPUTATION
            # ==========================================
            dx = cx - pivot[0]
            dy = cy - pivot[1]

            angle = np.degrees(np.arctan2(dx, dy))
            angle_buffer.append(angle)
            smooth_angle = np.mean(angle_buffer)

            # ==========================================
            # ANGULAR VELOCITY
            # ==========================================
            now = time.time()
            dt = now - last_time
            last_time = now

            if last_angle is None:
                last_angle = smooth_angle
                velocity = 0
            else:
                dtheta = smooth_angle - last_angle

                # Fix wrap-around discontinuity
                if dtheta > 180: dtheta -= 360
                if dtheta < -180: dtheta += 360

                velocity = dtheta / dt
                last_angle = smooth_angle

            # smooth the velocity
            velocity_buffer.append(velocity)
            smooth_velocity = np.mean(velocity_buffer)

            # ==========================================
            # SEND ONLY VELOCITY OVER SERIAL
            # ==========================================
            if ser is not None:
                msg = f"{smooth_velocity:.3f}\n"
                ser.write(msg.encode("utf-8"))

            # ==========================================
            # DISPLAY ON SCREEN
            # ==========================================
            cv2.putText(display, f"Velocity: {smooth_velocity:.2f} deg/s",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Mask", mask)
    cv2.imshow("Tracking", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# CLEANUP
# ==========================================
cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()
