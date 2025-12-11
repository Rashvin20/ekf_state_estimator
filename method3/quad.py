import cv2
import numpy as np
import time
import serial
from collections import deque
from picamera2 import Picamera2

# ============================================================
# QUADRATIC MODEL: y = a*x^2 + b*x + c  (x in DEGREES)
# ============================================================
class QuadraticDegInfer:
    def __init__(self, weights_path="quad_deg_v1.npz"):
        z = np.load(weights_path)
        self.a = float(z["a"])
        self.b = float(z["b"])
        self.c = float(z["c"])
        print("[OK] Loaded quadratic weights:", weights_path)
        print(f"[INFO] a={self.a:.10e}, b={self.b:.10e}, c={self.c:.10e}")

    def predict_speed_mps(self, angle_deg: float) -> float:
        x = float(angle_deg)
        return self.a * x * x + self.b * x + self.c

# ============================================================
# CONFIG
# ============================================================
WEIGHTS_PATH = "quad_deg_v1.npz"

pivot = (120, 10)
SMOOTHING_WINDOW = 5

GREEN_LOWER = np.array([40, 150, 100], dtype=np.uint8)
GREEN_UPPER = np.array([60, 250, 150], dtype=np.uint8)

SERIAL_PORT = "/dev/ttyAMA0"
BAUD = 115200

SHOW_WINDOWS = True   # set False for higher FPS

# ============================================================
# INIT
# ============================================================
infer = QuadraticDegInfer(WEIGHTS_PATH)

angle_buffer = deque(maxlen=SMOOTHING_WINDOW)

try:
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0)  # non-blocking
    print("[OK] Serial connected:", SERIAL_PORT)
except Exception as e:
    ser = None
    print("[WARN] Serial offline:", e)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (320, 240)}))
picam2.start()
time.sleep(1)

print("\nCamera initialized. Running quadratic speed inference...")
print("Press 'q' to quit.\n")

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

                # Angle in degrees (0..90)
                dx = cx - pivot[0]
                dy = cy - pivot[1]
                angle_deg = abs(np.degrees(np.arctan2(abs(dx), dy)))
                angle_deg = max(0.0, min(float(angle_deg), 90.0))

                # Smooth angle
                angle_buffer.append(angle_deg)
                smooth_angle_deg = float(np.mean(angle_buffer))

                # Predict speed
                speed_mps = infer.predict_speed_mps(smooth_angle_deg)

                # Optional: clip negative predictions
                if speed_mps < 0:
                    speed_mps = 0.0

                # Print + serial
                print(f"{speed_mps:.3f} m/s")
                if ser:
                    try:
                        ser.write(f"{speed_mps:.3f}\n".encode())
                    except:
                        pass

                # Preview windows (optional)
                if SHOW_WINDOWS:
                    display = frame.copy()
                    cv2.circle(display, (cx, cy), 5, (0, 255, 0), -1)
                    cv2.circle(display, pivot, 5, (255, 255, 0), -1)
                    cv2.line(display, pivot, (cx, cy), (255, 255, 0), 2)
                    cv2.putText(display, f"{speed_mps:.2f} m/s", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.imshow("Tracking", display)
                    cv2.imshow("Mask", mask)

    # Quit
    if SHOW_WINDOWS:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # No GUI mode: allow Ctrl+C to stop
        pass

# ============================================================
# CLEANUP
# ============================================================
picam2.close()
if SHOW_WINDOWS:
    cv2.destroyAllWindows()
if ser:
    ser.close()

print("Exited cleanly.")
