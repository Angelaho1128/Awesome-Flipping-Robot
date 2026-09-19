import mujoco
import numpy as np


class PancakeEnv:

    def __init__(self, model_path="sim/assets/arm.xml"):

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # -------------------------
        # Joint IDs
        # -------------------------

        self.elbow_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "elbow"
        )

        self.wrist_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "wrist"
        )

        # -------------------------
        # Pancake body
        # -------------------------

        self.pancake_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "pancake"
        )

        # -------------------------
        # Joint addresses
        # -------------------------

        self.elbow_qpos = self.model.jnt_qposadr[
            self.elbow_id
        ]

        self.wrist_qpos = self.model.jnt_qposadr[
            self.wrist_id
        ]

        self.elbow_qvel = self.model.jnt_dofadr[
            self.elbow_id
        ]

        self.wrist_qvel = self.model.jnt_dofadr[
            self.wrist_id
        ]

        # -------------------------
        # Pancake free joint
        # -------------------------

        pancake_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "pancake_free"
        )

        self.pancake_qpos = self.model.jnt_qposadr[
            pancake_joint_id
        ]

        self.time = 0.0


    # =====================================================
    # RESET
    # =====================================================

    def reset(self):

        mujoco.mj_resetData(
            self.model,
            self.data
        )

        # Starting arm configuration

        self.data.qpos[self.elbow_qpos] = np.deg2rad(35)
        self.data.qpos[self.wrist_qpos] = np.deg2rad(-35)

        # Pancake starts above the pan

        self.data.qpos[
            self.pancake_qpos:
            self.pancake_qpos + 3
        ] = [
            0.80,
            0.0,
            0.205
        ]

        # Pancake orientation

        self.data.qpos[
            self.pancake_qpos + 3:
            self.pancake_qpos + 7
        ] = [
            1,
            0,
            0,
            0
        ]

        # Pancake velocity

        self.data.qvel[:] = 0

        mujoco.mj_forward(
            self.model,
            self.data
        )

        self.time = 0.0

        return self.get_observation()


    # =====================================================
    # STEP
    # =====================================================

    def step(self, action):

        action = np.asarray(
            action,
            dtype=np.float32
        )

        # Elbow motor

        self.data.ctrl[0] = action[0]

        # Wrist motor

        self.data.ctrl[1] = action[1]

        mujoco.mj_step(
            self.model,
            self.data
        )

        self.time = self.data.time

        observation = self.get_observation()

        reward = self.get_reward()

        done = self.is_done()

        return observation, reward, done


    # =====================================================
    # OBSERVATION
    # =====================================================

    def get_observation(self):

        pancake_pos = self.data.xpos[
            self.pancake_body_id
        ].copy()

        pancake_vel = self.data.cvel[
            self.pancake_body_id
        ].copy()

        return np.array([

            # Pancake position
            pancake_pos[0],
            pancake_pos[1],
            pancake_pos[2],

            # Pancake velocity
            pancake_vel[3],
            pancake_vel[4],
            pancake_vel[5],

            # Joint positions
            self.data.qpos[self.elbow_qpos],
            self.data.qpos[self.wrist_qpos],

            # Joint velocities
            self.data.qvel[self.elbow_qvel],
            self.data.qvel[self.wrist_qvel],

            # Time
            self.time

        ], dtype=np.float32)


    # =====================================================
    # REWARD
    # =====================================================

    def get_reward(self):

        pancake_pos = self.data.xpos[
            self.pancake_body_id
        ]

        # Reward keeping pancake near the robot/pan area

        distance = np.linalg.norm(
            pancake_pos[:2]
        )

        return -distance


    # =====================================================
    # TERMINATION
    # =====================================================

    def is_done(self):

        if self.time >= 3.0:
            return True

        pancake_pos = self.data.xpos[
            self.pancake_body_id
        ]

        # Pancake fell below the floor

        if pancake_pos[2] < 0.02:
            return True

        return False