# -----------------------------------------
# ANGLE (degrees)
# -----------------------------------------
dx = cx - pivot[0]
dy = cy - pivot[1]

# Correct pendulum angle: 0° (rest) → 90° (horizontal)
angle = abs(np.degrees(np.arctan2(abs(dx), dy)))
angle = max(0, min(angle, 90))   # clamp to 0–90°

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
    last_angle = smooth_angle
    ang_vel_deg = dtheta / dt  # deg/s

velocity_buffer.append(ang_vel_deg)
ang_vel_deg_smooth = np.mean(velocity_buffer)

# -----------------------------------------
# CONVERT TO LINEAR VELOCITY (m/s)
# -----------------------------------------
ang_vel_rad = ang_vel_deg_smooth * (np.pi / 180)
linear_velocity = ang_vel_rad * PENDULUM_LENGTH_M

if linear_velocity < 0:
    linear_velocity = 0  # no negative physical speed
