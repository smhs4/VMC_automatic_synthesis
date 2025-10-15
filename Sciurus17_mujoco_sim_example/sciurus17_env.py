import os
from typing import Optional, Sequence, Tuple
import gymnasium as gym
import numpy as np
import mujoco, mujoco.viewer

class Sciurus17Env(gym.Env):
    """
    Minimal Gymnasium wrapper around a MuJoCo model.

    Action modes:
      - "torque": action = torques for the selected actuators
      - "position": action = desired joint positions (PD inside step)
    Options:
      - gravity_comp: if True, adds qfrc_bias (gravity + Coriolis) to control
    """
    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        xml_path: str,
        actuator_names: Optional[Sequence[str]] = None,
        render_mode: str = "human",
        action_mode: str = "torque",
        gravity_comp: bool = False,
        Kp: float = 50.0,
        Kd: float = 2.0,
        frame_skip: int = 5,
        torque_limit: float = 4.0,   # matches your ctrlrange in XML
        qpos_limit: float = np.pi,   # clip for position targets in "position" mode
        seed: Optional[int] = None,
    ):
        super().__init__()
        assert action_mode in ("torque", "position")
        assert render_mode in ("human", "none", "rgb_array")
        self.render_mode = render_mode
        self.action_mode = action_mode
        self.gravity_comp = gravity_comp
        self.Kp = Kp
        self.Kd = Kd
        self.frame_skip = frame_skip
        self.torque_limit = float(torque_limit)
        self.qpos_limit = float(qpos_limit)

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Resolve actuators to control
        if actuator_names is None:
            # control all actuators defined in XML
            self.actuator_ids = np.arange(self.model.nu, dtype=int)
            self.actuator_names = [
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                for i in self.actuator_ids
            ]
        else:
            self.actuator_names = list(actuator_names)
            self.actuator_ids = np.array([
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                for n in self.actuator_names
            ], dtype=int)

        self.n_act = len(self.actuator_ids)
        self.nq = self.model.nq
        self.nv = self.model.nv

        # Action space
        if self.action_mode == "torque":
            self.action_space = gym.spaces.Box(
                low=-self.torque_limit, high=self.torque_limit, shape=(self.n_act,), dtype=np.float32
            )
        else:  # "position" desired joint angles
            self.action_space = gym.spaces.Box(
                low=-self.qpos_limit, high=self.qpos_limit, shape=(self.n_act,), dtype=np.float32
            )

        # Observation: [qpos, qvel]
        high = np.inf * np.ones(self.nq + self.nv, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # Optional viewer
        self._viewer = None

        if seed is not None:
            self.reset(seed=seed)

        self.viewer = None
        self.renderer = None
        if render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        elif render_mode == "rgb_array":
            # create an offscreen renderer
            self.renderer = mujoco.Renderer(self.model, width=1280, height=720)

    # ---------- Gym API ----------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        # Set a neutral pose (freejoint first, then joints in order of qpos)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Small randomization (optional)
        rng = np.random.default_rng(seed)
        jitter = 0.02 * rng.standard_normal(self.nq)
        self.data.qpos[:] += jitter
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        assert action.shape == (self.n_act,)

        # Clear control
        self.data.ctrl[:] = 0.0

        # Base torque from action
        if self.action_mode == "torque":
            tau_cmd = np.clip(action, -self.torque_limit, self.torque_limit)
        else:
            # "position" mode: PD on actuator-related joints
            # Map actuator -> joint dof address
            tau_cmd = np.zeros(self.n_act, dtype=np.float32)
            for i, act_id in enumerate(self.actuator_ids):
                # Each actuator is tied to exactly one joint dof (in your XML)
                jnt_id = int(self.model.actuator_trnid[act_id][0])  # joint id
                dof_adr = int(self.model.jnt_dofadr[jnt_id])        # velocity index
                qpos_adr = int(self.model.jnt_qposadr[jnt_id])      # position index

                q_des = float(np.clip(action[i], -self.qpos_limit, self.qpos_limit))
                q = float(self.data.qpos[qpos_adr])
                dq = float(self.data.qvel[dof_adr])
                tau_cmd[i] = self.Kp * (q_des - q) - self.Kd * dq

            tau_cmd = np.clip(tau_cmd, -self.torque_limit, self.torque_limit)

        # Optional gravity compensation (additive in generalized coordinates)
        if self.gravity_comp:
            mujoco.mj_rne(self.model, self.data, 0, self.data.qacc)  # updates qfrc_bias
            # qfrc_bias is per-DOF; we need to pick components that each actuator addresses
            tau_gc = np.zeros(self.n_act, dtype=np.float32)
            for i, act_id in enumerate(self.actuator_ids):
                jnt_id = int(self.model.actuator_trnid[act_id][0])
                dof_adr = int(self.model.jnt_dofadr[jnt_id])
                tau_gc[i] = float(self.data.qfrc_bias[dof_adr])
            u = tau_cmd + tau_gc
        else:
            u = tau_cmd

        # Write controls into the correct ctrl slots
        self.data.ctrl[self.actuator_ids] = u
        print("ctrl:", self.data.ctrl[self.actuator_ids])

        # Integrate
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        # Placeholder reward/termination: always continue
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # Render if wanted
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is not None:
                self.viewer.sync()
        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                # lazy init if needed
                self.renderer = mujoco.Renderer(self.model, width=1280, height=720)
            self.renderer.update_scene(self.data, camera="fixed")
            return self.renderer.render()
        elif self.render_mode == "none":
            return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ---------- helpers ----------
    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)
