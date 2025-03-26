# Sub-Terranean Navigation Challenge – State Estimator

## Overview

This project addresses the **Sub-Terranean Navigation Challenge** where a robot navigates a confined indoor space **without GPS**. The objective is to estimate the robot's 2D position `(x, y)` and orientation `(yaw)` by fusing data from:
- **IMU sensors** (accelerometer, gyroscope, magnetometer)
- **Time-of-Flight (ToF) distance sensors**

The file `state_estimator.m` contains the MATLAB implementation of the **sensor fusion algorithm** used to estimate the robot's state in real-time.

---

## File Structure

- `state_estimator.m`: Main code that performs state estimation using sensor fusion techniques (e.g., Extended Kalman Filter).


---

## Sensor Fusion Method

The algorithm fuses noisy IMU and ToF data using a **state estimator**. Specifically:

- **Prediction Step**: Uses IMU data (acceleration, angular velocity) to predict the next state using a motion model.
- **Update Step**: Refines predictions using ToF sensor measurements and (optionally) magnetometer data.
- **Noise Handling**: Accounts for process and measurement noise using covariance matrices.

Implemented using an **Extended Kalman Filter (EKF)** framework.

---

## Usage

1. Open `state_estimator.m` in MATLAB.
