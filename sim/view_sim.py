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

data.qpos[elbow_qpos] = np.deg2rad(45)
data.qpos[wrist_qpos] = np.deg2rad(-65)

mujoco.mj_forward(model, data)

# --------------------------------------------------
# Find pancake
# --------------------------------------------------

pancake_body = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "pancake"
)

# --------------------------------------------------
# Flip controller
# --------------------------------------------------

def flip_controller(t):

    # ----------------------------------------------
    # Phase 1:
    # Move the arm upward
    # ----------------------------------------------

    if t < 0.7:

        progress = t / 0.7

        elbow_target = np.deg2rad(
            45 + 30 * progress
        )

        wrist_target = np.deg2rad(
            -65 + 20 * progress
        )

    # ----------------------------------------------
    # Phase 2:
    # Rapid wrist acceleration
    # This launches the pancake
    # ----------------------------------------------

    elif t < 1.05:

        progress = (t - 0.7) / 0.35

        elbow_target = np.deg2rad(
            75 - 35 * progress
        )

        wrist_target = np.deg2rad(
            -45 + 150 * progress
        )

    # ----------------------------------------------
    # Phase 3:
    # Follow through
    # ----------------------------------------------

    elif t < 1.5:

        progress = (t - 1.05) / 0.45

        elbow_target = np.deg2rad(
            40 + 20 * progress
        )

        wrist_target = np.deg2rad(
            105 - 60 * progress
        )

    else:

        elbow_target = np.deg2rad(60)
        wrist_target = np.deg2rad(45)

    return elbow_target, wrist_target


# --------------------------------------------------
# PD controller
# --------------------------------------------------

def pd_control(target, position, velocity):

    kp = 80
    kd = 8

    return kp * (target - position) - kd * velocity


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

        # Get desired joint positions

        elbow_target, wrist_target = flip_controller(t)

        # Current positions

        elbow_position = data.qpos[elbow_qpos]
        wrist_position = data.qpos[wrist_qpos]

        # Current velocities

        elbow_velocity = data.qvel[elbow_qvel]
        wrist_velocity = data.qvel[wrist_qvel]

        # Calculate motor torques

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

        # Apply torques

        data.ctrl[0] = elbow_torque
        data.ctrl[1] = wrist_torque

        # Step simulation

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(0.002)