import cv2
import numpy as np
import time
import serial
from collections import deque
from picamera2 import Picamera2

# -----------------------------------------
# USER SETTINGS
# -----------------------------------------
pivot = (120, 50)     # pivot in cropped frame
SMOOTHING_WINDOW = 5
PENDULUM_LENGTH_M = 1.70   # length in meters

# Your green colour range
GREEN_LOWER = np.array([40, 200, 110], dtype=np.uint8)
GREEN_UPPER = np.array([50, 220, 130], dtype=np.uint8)

# Buffers
angle_buffer = deque(maxlen=SMOOTHING_WINDOW)
velocity_buffer = deque(maxlen=SMOOTHING_WINDOW)

last_angle = None
last_time = time.time()

# -----------------------------------------
# SERIAL SETUP
# -----------------------------------------
SERIAL_PORT = "/dev/ttyAMA0"
BAUD = 115200

try:
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
    print(f"[OK] Serial connected: {SERIAL_PORT}")
except Exception as e:
    print(f"[WARNING] Serial not available ({e})")
    ser = None


# -----------------------------------------
# INITIALISE PICAMERA2
# -----------------------------------------
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (320, 240)}))

try:
    picam2.start()
    time.sleep(1)
except Exception as e:
    print("Camera failed to start:", e)
    exit()

print("Camera initialized. Tracking GREEN marker...")

# -----------------------------------------
# MAIN LOOP (200 Hz)
# -----------------------------------------
while True:

    loop_start = time.time()

    frame = picam2.capture_array()

    # Crop to reduce noise and speed up processing
    frame = frame[20:220, 40:280]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # mask for GREEN
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area > 80:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # -----------------------------------------
                # PROPER ANGLE COMPUTATION (0–90 degrees)
                # -----------------------------------------
                dx = cx - pivot[0]
                dy = cy - pivot[1]

                # Angle between vertical and rod
                angle = abs(np.degrees(np.arctan2(abs(dx), dy)))

                # Clamp to 0–90
                angle = max(0, min(angle, 90))

                # Smooth the angle
                angle_buffer.append(angle)
                smooth_angle = np.mean(angle_buffer)

                # -----------------------------------------
                # ANGULAR VELOCITY
                # -----------------------------------------
                now = time.time()
                dt = now - last_time
                last_time = now

                if last_angle is None:
                    last_angle = smooth_angle
                    ang_vel_deg = 0
                else:
                    ang_vel_deg = (smooth_angle - last_angle) / dt
                    last_angle = smooth_angle

                velocity_buffer.append(ang_vel_deg)
                ang_vel_deg_smooth = np.mean(velocity_buffer)

                # Convert to linear velocity
                ang_vel_rad = ang_vel_deg_smooth * (np.pi / 180)
                linear_velocity = ang_vel_rad * PENDULUM_LENGTH_M

                # If moving backwards → clamp
                if linear_velocity < 0:
                    linear_velocity = 0

                # -----------------------------------------
                # OUTPUT (angle only)
                # -----------------------------------------
                print(f"{smooth_angle:.3f} deg")

                if ser:
                    try:
                        ser.write(f"{smooth_angle:.3f}\n".encode())
                    except:
                        pass

    # --------------------------
    # Maintain 200 Hz loop
    # --------------------------
    elapsed = time.time() - loop_start
    delay = max(0, 0.005 - elapsed)
    time.sleep(delay)

# Cleanup
picam2.close()
if ser:
    ser.close()
cv2.destroyAllWindows()
