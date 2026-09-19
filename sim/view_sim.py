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

data.qpos[elbow_qpos] = np.deg2rad(35)
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

    # ==========================================
    # STARTING CONFIGURATION
    # ==========================================

    start_elbow = np.deg2rad(35)
    start_wrist = np.deg2rad(-60)

    # ==========================================
    # PHASE 1: LOAD
    # ==========================================

    if t < 0.35:

        p = t / 0.35

        # Move backward/down.
        elbow = start_elbow + np.deg2rad(20) * p

        # Keep the pan slightly downward.
        wrist = start_wrist - np.deg2rad(5) * p


    # ==========================================
    # PHASE 2: LAUNCH
    # ==========================================

    elif t < 0.75:

        p = (t - 0.35) / 0.40

        # Rapid elbow extension.
        elbow = np.deg2rad(55) - np.deg2rad(90) * p

        # Wrist follows but does not snap yet.
        wrist = np.deg2rad(-65) + np.deg2rad(30) * p


    # ==========================================
    # PHASE 3: WRIST SNAP
    # ==========================================

    elif t < 0.95:

        p = (t - 0.75) / 0.20

        # Elbow reaches the top of the arc.
        elbow = np.deg2rad(-35) + np.deg2rad(10) * p

        # Very fast wrist snap.
        wrist = np.deg2rad(-35) + np.deg2rad(170) * p


    # ==========================================
    # PHASE 4: FOLLOW THROUGH
    # ==========================================

    elif t < 1.35:

        p = (t - 0.95) / 0.40

        # Bring the arm around the arc.
        elbow = np.deg2rad(-25) + np.deg2rad(60) * p

        wrist = np.deg2rad(135) - np.deg2rad(195) * p


    # ==========================================
    # PHASE 5: RETURN TO CATCH
    # ==========================================

    elif t < 1.75:

        p = (t - 1.35) / 0.40

        elbow = np.deg2rad(-25) + (
            np.deg2rad(35) - np.deg2rad(-25)
        ) * p

        wrist = np.deg2rad(-60)

    # ==========================================
    # PHASE 6: CATCH
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