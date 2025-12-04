import cv2
import numpy as np
import time
import serial
from collections import deque
from picamera2 import Picamera2

# -----------------------------------------
# USER SETTINGS
# -----------------------------------------
pivot = (160, 60)          # pivot location in cropped frame
SMOOTHING_WINDOW = 5
PENDULUM_LENGTH_M = 1.70   # 170 cm = 1.70 m

# Green HSV range
GREEN_LOWER = np.array([40, 40, 40], dtype=np.uint8)
GREEN_UPPER = np.array([85, 255, 255], dtype=np.uint8)

# Buffers
angle_buffer = deque(maxlen=SMOOTHING_WINDOW)
velocity_buffer = deque(maxlen=SMOOTHING_WINDOW)

last_angle = None
last_time = time.time()

# -----------------------------------------
# SERIAL SETUP
# -----------------------------------------
SERIAL_PORT = "/dev/ttyACM0"  # change if needed
BAUD = 115200

try:
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
    print(f"[OK] Serial connected: {SERIAL_PORT}")
except Exception as e:
    print(f"[WARNING] Serial not available ({e})")
    ser = None


# -----------------------------------------
# INITIALISE PI CAMERA
# -----------------------------------------
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (320, 240)}))

try:
    picam2.start()
    time.sleep(1)
except Exception as e:
    print("Camera failed to start:", e)
    exit()

print("Camera initialized. Tracking GREEN ball...")


# -----------------------------------------
# MAIN LOOP
# -----------------------------------------
while True:
    frame = picam2.capture_array()

    # crop to reduce processing (adjust if needed)
    frame = frame[20:220, 40:280]

    display = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # mask for GREEN
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

            # Draw
            cv2.circle(display, (cx, cy), 6, (0, 255, 0), -1)
            cv2.circle(display, pivot, 6, (255, 255, 0), -1)
            cv2.line(display, pivot, (cx, cy), (255, 255, 0), 2)

            # -----------------------------------------
            # ANGLE (degrees)
            # -----------------------------------------
            dx = cx - pivot[0]
            dy = cy - pivot[1]
            angle = np.degrees(np.arctan2(dx, dy))

            angle_buffer.append(angle)
            smooth_angle = np.mean(angle_buffer)

            # -----------------------------------------
            # ANGULAR VELOCITY (deg/s)
            # -----------------------------------------
            now = time.time()
            dt = now - last_time
            last_time = now

            if last_angle is None:
                last_angle = smooth_angle
                ang_vel_deg = 0
            else:
                dtheta = smooth_angle - last_angle

                # wrap-around fix
                if dtheta > 180: dtheta -= 360
                if dtheta < -180: dtheta += 360

                ang_vel_deg = dtheta / dt
                last_angle = smooth_angle

            velocity_buffer.append(ang_vel_deg)
            ang_vel_deg_smooth = np.mean(velocity_buffer)

            # -----------------------------------------
            # CONVERT TO LINEAR VELOCITY (m/s)
            # -----------------------------------------
            ang_vel_rad = ang_vel_deg_smooth * (np.pi / 180)
            linear_velocity = ang_vel_rad * PENDULUM_LENGTH_M

            # -----------------------------------------
            # PRINT + SERIAL OUTPUT
            # -----------------------------------------
            print(f"{linear_velocity:.3f} m/s")

            if ser is not None:
                try:
                    ser.write(f"{linear_velocity:.3f}\n".encode())
                except:
                    pass

            # -----------------------------------------
            # SHOW ON SCREEN
            # -----------------------------------------
            cv2.putText(display, f"v = {linear_velocity:.2f} m/s",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Mask", mask)
    cv2.imshow("Tracking", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.close()
cv2.destroyAllWindows()

if ser:
    ser.close()
