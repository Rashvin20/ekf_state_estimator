import cv2
import numpy as np
import time
import serial
from collections import deque
from picamera2 import Picamera2

# ============================================================
# LUT CALIBRATOR (loads .npz)
# ============================================================
class LUTCalibrator:
    def __init__(self, weights_path):
        w = np.load(weights_path)
        self.xk = w["xk"].astype(np.float32)   # student knots (angle/10)
        self.yk = w["yk"].astype(np.float32)   # speed knots (m/s)

    def predict(self, x):
        x = np.float32(x)
        # clip to range to avoid weird extrapolation
        x = np.clip(x, self.xk[0], self.xk[-1])
        return float(np.interp(x, self.xk, self.yk))

# === Load LUT weights (YOUR PATH) ===
LUT_PATH = "lut64_calib_fast_v1.npz"
calib = LUTCalibrator(LUT_PATH)
print("[OK] LUT loaded:", LUT_PATH)

# ============================================================
# CAMERA / SERIAL CONFIG
# ============================================================
pivot = (120, 10)
SMOOTHING_WINDOW = 5

GREEN_LOWER = np.array([40, 150, 100], dtype=np.uint8)
GREEN_UPPER = np.array([60, 250, 150], dtype=np.uint8)

angle_buffer = deque(maxlen=SMOOTHING_WINDOW)

SERIAL_PORT = "/dev/ttyAMA0"
BAUD = 115200

try:
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0)  # non-blocking
    print("[OK] Serial connected")
except Exception as e:
    ser = None
    print("[WARN] Serial offline:", e)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (320, 240)}))
picam2.start()
time.sleep(1)

print("Camera initialized. Running LUT speed inference...\n")

# ============================================================
# MAIN LOOP
# ============================================================
while True:
    frame = picam2.capture_array()
    frame = frame[20:220, 40:280]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area > 80:
            M = cv2.moments(c)
            if M["m00"] > 1e-6:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Compute angle (0–90 deg)
                dx = cx - pivot[0]
                dy = cy - pivot[1]
                angle_deg = abs(np.degrees(np.arctan2(abs(dx), dy)))
                angle_deg = max(0.0, min(float(angle_deg), 90.0))

                # Smooth
                angle_buffer.append(angle_deg)
                smooth_angle_deg = float(np.mean(angle_buffer))

                # IMPORTANT: training used angle/10
                student_x = smooth_angle_deg / 10.0

                # LUT prediction
                speed_mps = calib.predict(student_x)

                # Print + serial
                print(f"{speed_mps:.3f} m/s")
                if ser:
                    try:
                        ser.write(f"{speed_mps:.3f}\n".encode())
                    except:
                        pass

                # Optional preview (comment out for max Hz)
                display = frame.copy()
                cv2.circle(display, (cx, cy), 5, (0, 255, 0), -1)
                cv2.circle(display, pivot, 5, (255, 255, 0), -1)
                cv2.line(display, pivot, (cx, cy), (255, 255, 0), 2)
                cv2.putText(display, f"{speed_mps:.2f} m/s", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Tracking", display)
                cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.close()
cv2.destroyAllWindows()
if ser:
    ser.close()
