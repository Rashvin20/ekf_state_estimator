import cv2
import numpy as np
import time
import serial
from collections import deque
from picamera2 import Picamera2

import torch
from torch import nn

# ============================================================
# FAST SETTINGS (optional)
# ============================================================
torch.set_num_threads(1)  # helps on Raspberry Pi to reduce overhead

# ============================================================
# SPEEDNET64 MODEL (MUST MATCH TRAINING)
# ============================================================
class SpeedNet64(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)

class SpeedNet64DegInfer:
    """
    Inference wrapper for model trained on ANGLE IN DEGREES.
    Loads:
      - speednet64_deg_v1.pth
      - speednet64_deg_scalers_v1.npz
    """
    def __init__(self,
                 weights_path="speednet64_deg_v1.pth",
                 scaler_path="speednet64_deg_scalers_v1.npz"):
        # Load model
        self.model = SpeedNet64()
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()

        # Load scaler params (saved by your training script)
        s = np.load(scaler_path)
        self.x_mean = float(s["x_mean"][0])
        self.x_std  = float(s["x_std"][0])
        self.y_mean = float(s["y_mean"][0])
        self.y_std  = float(s["y_std"][0])

        # Pre-allocate tensor (avoid allocations per frame)
        self._x = torch.zeros((1, 1), dtype=torch.float32)

        print("[OK] Loaded NN weights:", weights_path)
        print("[OK] Loaded scalers   :", scaler_path)
        print(f"[INFO] x_mean={self.x_mean:.6f}, x_std={self.x_std:.6f}")
        print(f"[INFO] y_mean={self.y_mean:.6f}, y_std={self.y_std:.6f}")

    def predict_speed_mps(self, angle_deg: float) -> float:
        # Normalize input (angle in degrees!)
        x_n = (float(angle_deg) - self.x_mean) / self.x_std
        self._x[0, 0] = x_n

        with torch.no_grad():
            y_n = self.model(self._x).item()

        # Un-normalize output to m/s
        speed = y_n * self.y_std + self.y_mean
        return float(speed)

# ============================================================
# CONFIG
# ============================================================
WEIGHTS_PATH = "speednet64_deg_v1.pth"
SCALER_PATH  = "speednet64_deg_scalers_v1.npz"

# Camera/vision
pivot = (120, 50)
SMOOTHING_WINDOW = 5

GREEN_LOWER = np.array([40, 150, 100], dtype=np.uint8)
GREEN_UPPER = np.array([60, 250, 150], dtype=np.uint8)

angle_buffer = deque(maxlen=SMOOTHING_WINDOW)

# Serial
SERIAL_PORT = "/dev/ttyAMA0"
BAUD = 115200

# ============================================================
# INIT
# ============================================================
infer = SpeedNet64DegInfer(WEIGHTS_PATH, SCALER_PATH)

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

print("Camera initialized. Running NN speed inference...\n")
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

                # Compute angle in degrees (0–90)
                dx = cx - pivot[0]
                dy = cy - pivot[1]
                angle_deg = abs(np.degrees(np.arctan2(abs(dx), dy)))
                angle_deg = max(0.0, min(float(angle_deg), 90.0))

                # Smooth angle
                angle_buffer.append(angle_deg)
                smooth_angle_deg = float(np.mean(angle_buffer))

                # Predict speed (m/s) from angle in degrees (NO /10 HERE)
                speed_mps = infer.predict_speed_mps(smooth_angle_deg)

                # Print + serial
                print(f"{speed_mps:.3f} m/s")
                if ser:
                    try:
                        ser.write(f"{speed_mps:.3f}\n".encode())
                    except:
                        pass

                # Preview (comment out for higher FPS)
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

# ============================================================
# CLEANUP
# ============================================================
picam2.close()
cv2.destroyAllWindows()
if ser:
    ser.close()
print("Exited cleanly.")
