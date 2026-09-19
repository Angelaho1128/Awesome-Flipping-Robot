import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "sim/assets/arm.xml"

print("Loading MuJoCo model...")

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print("MuJoCo loaded!")
print("Joints:", model.njnt)
print("Actuators:", model.nu)


# ==================================================
# PARAMETERS
# ==================================================

# Arm geometry
ELBOW_TO_WRIST = 0.40
WRIST_TO_PAN = 0.40

# Pancake
PANCAKE_MASS = 0.05
PANCAKE_RADIUS = 0.095
PANCAKE_THICKNESS = 0.012

# Pan
PAN_MASS = 0.40

# 57STH56 wrist motor
WRIST_MOTOR_MASS = 0.70


# --------------------------------------------------
# Find joints
# --------------------------------------------------

elbow_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "elbow"
)

wrist_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "wrist"
)

elbow_qpos = model.jnt_qposadr[elbow_id]
wrist_qpos = model.jnt_qposadr[wrist_id]

elbow_qvel = model.jnt_dofadr[elbow_id]
wrist_qvel = model.jnt_dofadr[wrist_id]


# --------------------------------------------------
# Initial robot configuration
# --------------------------------------------------

data.qpos[elbow_qpos] = np.deg2rad(35)
data.qpos[wrist_qpos] = np.deg2rad(-65)

mujoco.mj_forward(model, data)


# --------------------------------------------------
# Find pancake and pan
# --------------------------------------------------

pancake_body = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "pancake"
)

pan_body = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "pan"
)

pancake_joint = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "pancake_free"
)

pancake_qpos = model.jnt_qposadr[pancake_joint]


# --------------------------------------------------
# Place pancake ON the tilted pan
# --------------------------------------------------

# Get the pan's exact position
pan_position = data.xpos[pan_body].copy()

# Get the pan's exact orientation
pan_rotation = data.xmat[pan_body].reshape(3, 3).copy()

# The pan's local Z axis is perpendicular
# to the cooking surface.
pan_normal = pan_rotation[:, 2]

# Place pancake slightly above the pan.
#
# Pan surface is approximately 0.015 m above
# the pan body's center.
#
# Pancake half-thickness is 0.012 m.
#
# Total offset = 0.027 m.
pancake_position = pan_position + pan_normal * 0.027

# Set pancake position
data.qpos[
    pancake_qpos:pancake_qpos + 3
] = pancake_position


# --------------------------------------------------
# Make pancake parallel to the pan
# --------------------------------------------------

pancake_quat = np.zeros(4)

mujoco.mju_mat2Quat(
    pancake_quat,
    pan_rotation.flatten()
)

data.qpos[
    pancake_qpos + 3:pancake_qpos + 7
] = pancake_quat

mujoco.mj_forward(model, data)


# --------------------------------------------------
# Flip controller
# --------------------------------------------------

def flip_controller(t):

    # Starting configuration
    start_elbow = np.deg2rad(35)
    start_wrist = np.deg2rad(-65)

    # ==========================================
    # PHASE 1: LOAD
    # ==========================================
    if t < 0.30:

        p = t / 0.30

        # Elbow moves forward/up slightly.
        elbow = np.deg2rad(35) + np.deg2rad(20) * p

        # Keep pan facing generally forward.
        wrist = np.deg2rad(-65)


    # ==========================================
    # PHASE 2: ARC / LAUNCH
    # ==========================================
    elif t < 0.65:

        p = (t - 0.30) / 0.35

        # Elbow sweeps through the launch.
        #
        # 55° -> 0°
        #
        # This creates the large arm movement.
        elbow = np.deg2rad(55) - np.deg2rad(55) * p

        # Wrist stays mostly controlled during
        # the initial arc.
        wrist = np.deg2rad(-65) + np.deg2rad(15) * p


    # ==========================================
    # PHASE 3: WRIST SNAP AT TOP OF ARC
    # ==========================================
    elif t < 0.83:

        p = (t - 0.65) / 0.18

        # Elbow continues slightly through the arc.
        elbow = np.deg2rad(0) + np.deg2rad(-10) * p

        # Now rotate the pan/pancake.
        wrist = np.deg2rad(-50) + np.deg2rad(180) * p


    # ==========================================
    # PHASE 4: FOLLOW THROUGH / CATCH
    # ==========================================
    elif t < 1.20:

        p = (t - 0.83) / 0.37

        # Bring elbow underneath the pancake.
        elbow = np.deg2rad(-10) + np.deg2rad(45) * p

        # Bring wrist back underneath the pancake.
        wrist = np.deg2rad(130) - np.deg2rad(195) * p


    # ==========================================
    # PHASE 5: RETURN TO START
    # ==========================================
    elif t < 1.55:

        p = (t - 1.15) / 0.40

        # Lower the pan slightly for the catch
        elbow = np.deg2rad(35 - 65 * (1 - p))
        wrist = np.deg2rad(-60)


    # ==========================================
    # CATCH
    # ==========================================
    else:

        elbow = start_elbow
        wrist = start_wrist

    return elbow, wrist


# --------------------------------------------------
# PD controller
# --------------------------------------------------

def pd_control(target, position, velocity):

    kp = 100
    kd = 8

    torque = kp * (target - position) - kd * velocity

    return np.clip(torque, -30, 30)


# --------------------------------------------------
# Reset simulation
# --------------------------------------------------

def reset_simulation():

    mujoco.mj_resetData(model, data)

    # Restore initial robot configuration
    data.qpos[elbow_qpos] = np.deg2rad(35)
    data.qpos[wrist_qpos] = np.deg2rad(-65)

    mujoco.mj_forward(model, data)

    # Recalculate the pan position/orientation
    pan_position = data.xpos[pan_body].copy()
    pan_rotation = data.xmat[pan_body].reshape(3, 3).copy()

    pan_normal = pan_rotation[:, 2]

    # Put pancake back ON the tilted pan
    pancake_position = pan_position + pan_normal * 0.027

    data.qpos[
        pancake_qpos:pancake_qpos + 3
    ] = pancake_position

    # Match pancake orientation to pan
    pancake_quat = np.zeros(4)

    mujoco.mju_mat2Quat(
        pancake_quat,
        pan_rotation.flatten()
    )

    data.qpos[
        pancake_qpos + 3:pancake_qpos + 7
    ] = pancake_quat

    # Remove all initial velocity
    data.qvel[:] = 0

    mujoco.mj_forward(model, data)


# --------------------------------------------------
# Viewer
# --------------------------------------------------

print("Opening viewer...")

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    start_time = time.time()

    while viewer.is_running():

        t = time.time() - start_time


        # ------------------------------------------
        # Automatically restart after each attempt
        # ------------------------------------------

        if t >= 3.0:

            reset_simulation()

            start_time = time.time()

            continue


        # ------------------------------------------
        # Get desired joint positions
        # ------------------------------------------

        elbow_target, wrist_target = flip_controller(t)


        # ------------------------------------------
        # Current positions
        # ------------------------------------------

        elbow_position = data.qpos[elbow_qpos]
        wrist_position = data.qpos[wrist_qpos]


        # ------------------------------------------
        # Current velocities
        # ------------------------------------------

        elbow_velocity = data.qvel[elbow_qvel]
        wrist_velocity = data.qvel[wrist_qvel]


        # ------------------------------------------
        # Calculate motor torques
        # ------------------------------------------

        elbow_torque = pd_control(
            elbow_target,
            elbow_position,
            elbow_velocity
        )

        wrist_torque = pd_control(
            wrist_target,
            wrist_position,
            wrist_velocity
        )


        # ------------------------------------------
        # Apply torques
        # ------------------------------------------

        data.ctrl[0] = elbow_torque
        data.ctrl[1] = wrist_torque


        # ------------------------------------------
        # Step simulation
        # ------------------------------------------

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(0.002)