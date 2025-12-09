import numpy as np
import serial
import time

# -------------------------------------------
# LOAD MODEL
# -------------------------------------------
data = np.load("model.npz")

W1 = data["W1"]
b1 = data["b1"]
W2 = data["W2"]
b2 = data["b2"]
W3 = data["W3"]
b3 = data["b3"]
angle_mean = float(data["angle_mean"])
angle_std = float(data["angle_std"])

def predict_speed(angle):
    """Compute speed from angle using the trained neural network."""
    x = (angle - angle_mean) / angle_std
    x = np.array([[x]])

    h1 = np.maximum(0, x @ W1 + b1)
    h2 = np.maximum(0, h1 @ W2 + b2)
    y = h2 @ W3 + b3

    speed = float(y[0,0])
    return max(speed, 0.0)  # Never negative


# -------------------------------------------
# SERIAL READER
# -------------------------------------------
ser = serial.Serial("/dev/ttyACM0", 115200, timeout=1)
print("Connected. Reading angles...")

while True:
    try:
        line = ser.readline().decode().strip()
        if not line:
            continue

        angle = float(line)  # Expect pure angle in degrees

        speed = predict_speed(angle)

        print(f"Angle={angle:6.2f}° → Speed={speed:6.3f} m/s")

    except Exception as e:
        print("Error:", e)
        time.sleep(0.2)
