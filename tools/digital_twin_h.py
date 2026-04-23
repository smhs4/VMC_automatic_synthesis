#!/usr/bin/env python3

import rospy
import mujoco
import mujoco.viewer
import numpy as np
import os
# CHANGE 1: Import PoseStamped instead of Pose
from geometry_msgs.msg import PoseStamped 
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import rospkg

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import json
from pathlib import Path



class MujocoVizNode:
    def __init__(self):
        rospy.init_node('mujoco_viz')

        # Configuration
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('vmc_reaching')

        # Main phase assets
        self.main_robot_urdf_path = os.path.join(pkg_path, 'urdf', 'best_model.xml')
        self.main_vm_config_path = os.path.join(pkg_path, 'config', 'best_model_tendons.json')
        # Initial pose phase assets
        self.init_robot_urdf_path = os.path.join(pkg_path, 'urdf', 'initial_pose_model.xml')
        self.init_vm_config_path = os.path.join(pkg_path, 'config', 'initial_pose_tendons.json')

        self.cup_mesh_path = os.path.join(pkg_path, 'urdf', 'cup.stl')

        self.tendon_k = [100]
        self.tendon_l0 = [0.15]

        self.real_time = 0

        self.latest_object_pose = None

        has_init_assets = Path(self.init_robot_urdf_path).exists() and Path(self.init_vm_config_path).exists()
        if has_init_assets:
            self.control_phase = "init_pose"
            active_xml = self.init_robot_urdf_path
            active_json = self.init_vm_config_path
        else:
            self.control_phase = "main"
            active_xml = self.main_robot_urdf_path
            active_json = self.main_vm_config_path
            rospy.logwarn(
                "Initial pose assets missing. Starting directly in main phase. "
                f"Expected: {self.init_robot_urdf_path} and {self.init_vm_config_path}"
            )

        try:
            self.load_active_model(active_xml, active_json, preserve_state=False)
        except Exception as e:
            rospy.logerr(f"Failed to load active model/descriptor: {e}")
            return

        # Two-stage VMC: initial pose alignment, then main task VMC.
        self.pose_joint_names = [
            "waist_yaw_joint", "neck_yaw_joint", "neck_pitch_joint",
            "r_arm_joint1", "r_arm_joint2", "r_arm_joint3", "r_arm_joint4",
            "r_arm_joint5", "r_arm_joint6", "r_arm_joint7", "r_hand_mimic_joint",
            "l_arm_joint1", "l_arm_joint2", "l_arm_joint3", "l_arm_joint4",
            "l_arm_joint5", "l_arm_joint6", "l_arm_joint7", "l_hand_mimic_joint"
        ]
        self.pose_joint_targets = np.array([
            0.05062136600022616, -0.3436116964863836, -1.3959225169759335,
            0.10584467436410924, -1.5539225381281545, 0.09817477042468103, 2.741223667951641,
            0.02761165418194154, -1.7916895602504288, -1.6122138080678088, 0.3,
            -0.0798, 1.53, 0.145, -2.26,
            0.0372, 1.07, -2.0, -0.3
        ], dtype=np.float64)
        self.pose_error_threshold = 0.08  # rad RMS across tracked joints
        self.pose_settle_time = 0.25      # seconds under threshold before switching
        self.pose_max_time = 8.0          # safety timeout
        self.pose_error_below_since = None
        rospy.loginfo(f"Control phase initialized to: {self.control_phase}")

        # Subscribers

        # Object tracking subscriber 
        rospy.Subscriber('/object_pose', PoseStamped, self.pose_callback)

        # Joint positions subscribers
        rospy.Subscriber('/sciurus17/controller2/joint_states', JointState, self.joint_callback) #
        rospy.Subscriber('/sciurus17/controller1/joint_states', JointState, self.joint_callback) #
        rospy.Subscriber('/sciurus17/controller3/joint_states', JointState, self.joint_callback) #

        # Torque publisher setup
        self.pub_left  = rospy.Publisher("/sciurus17/controller2/vmc_command", Float64MultiArray, queue_size=1)
        self.pub_right = rospy.Publisher("/sciurus17/controller1/vmc_command", Float64MultiArray, queue_size=1)
        self.pub_neck  = rospy.Publisher("/sciurus17/controller3/neck/vmc_command", Float64MultiArray, queue_size=1)
        self.pub_waist = rospy.Publisher("/sciurus17/controller3/waist/vmc_command", Float64MultiArray, queue_size=1)

        self.left_indices = [12, 13, 14, 15, 16, 17, 18]
        self.right_indices = [3, 4, 5, 6, 7, 8, 9]
        self.waist_indices = [0]
        self.neck_indices = [1, 2]
        self.left_gripper_indices = [19, 20]
        self.right_gripper_indices = [10, 11]

        ########## Plotting params ##########
        self.time = []
        self.total_torque = []
        self.velocity_plot = []
        self.last_time = rospy.Time.now()
        self.loop_time = 0
        ########## Velocity calc params #########
        self.prev_tendon_length = {}
        self.prev_time = None
        self.filtered_velocity = {}
        self.control_rate = 500  # Hz

        rospy.loginfo("Digital Twin Ready - Starting Virtual Model Control.")
        self.control_timer = rospy.Timer(
            rospy.Duration(1.0 / self.control_rate),
            self.control_loop
        ) 
        # self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        # self.viewer_timer = rospy.Timer(
        # rospy.Duration(1.0 / 30.0), # 30 Hz.
        # self.viewer_loop
        # )
        # self.run()
    
    def tendon_id_map(self, Virtual_Descriptor):
        self.tendon_name_to_id = {}

        for spring in Virtual_Descriptor:
            tendon_name = spring["name"]
            tendon_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_TENDON,
                tendon_name
            )

            if tendon_id == -1:
                raise ValueError(f"Tendon '{tendon_name}' not found in model.")

            self.tendon_name_to_id[tendon_name] = tendon_id

    def load_virtual_descriptor(self, filepath):

        with open(filepath, 'r') as f:
            descriptor = json.load(f)

        return descriptor

    def load_virtual_descriptor_optional(self, filepath):
        """Load descriptor if present; otherwise return None."""
        if not Path(filepath).exists():
            rospy.logwarn(f"Initial pose VMC descriptor not found: {filepath}. Starting directly in main phase.")
            return None
        return self.load_virtual_descriptor(filepath)

    def compute_pose_error_rms(self):
        """RMS joint error vs desired initial pose for phase switching."""
        errs = []
        for jname, target in zip(self.pose_joint_names, self.pose_joint_targets):
            if jname not in self.joint_map:
                continue
            qaddr = self.joint_map[jname]
            errs.append(float(self.data.qpos[qaddr]) - float(target))
        if len(errs) == 0:
            return np.inf
        errs = np.array(errs, dtype=np.float64)
        return float(np.sqrt(np.mean(errs ** 2)))

    def compute_tendon_velocity(self, tendon_id, current_length):

        now = self.data.time  

        rospy.loginfo_throttle(0.5, f"MuJoCo time, {self.data.time}")
        if self.prev_time is None:
            self.prev_time = now
            self.prev_tendon_length[tendon_id] = current_length
            self.filtered_velocity[tendon_id] = 0.0
            return 0.0

        dt = now - self.prev_time
        if dt <= 1e-6:
            return self.filtered_velocity.get(tendon_id, 0.0)

        prev_length = self.prev_tendon_length.get(tendon_id, current_length)

        raw_velocity = (current_length - prev_length) / dt
        
        # ---- Clamp raw velocity ----
        max_raw = 5.0   # m/s (choose safe value)
        raw_velocity = np.clip(raw_velocity, -max_raw, max_raw)

        # ---- Low-pass filter ----
        alpha = 0.2
        prev_filtered = self.filtered_velocity.get(tendon_id, 0.0)

        filtered = alpha * raw_velocity + (1 - alpha) * prev_filtered

        # ---- Clamp final velocity ----
        max_vel = 3.0
        filtered = np.clip(filtered, -max_vel, max_vel)

        # ---- Save state ----
        self.prev_tendon_length[tendon_id] = current_length
        self.filtered_velocity[tendon_id] = filtered
        self.prev_time = now

        return filtered


    def add_cup_to_spec(self):
        rospy.loginfo("Cup Added.")
        mesh = self.spec.add_mesh()
        mesh.name = "cup_mesh_asset"
        mesh.file = self.cup_mesh_path
            
        cup_body = self.spec.worldbody.add_body()
        cup_body.name = "cup_body"
        cup_body.pos = [0.5, 0.0, 0.5]

        cup_geom = cup_body.add_geom()
        cup_geom.name = "cup_geom"
        cup_geom.type = mujoco.mjtGeom.mjGEOM_MESH
        cup_geom.meshname = "cup_mesh_asset"
        cup_geom.rgba = [0, 1, 0, 1]

        joint = cup_body.add_joint()
        joint.name = "cup_joint"
        joint.type = mujoco.mjtJoint.mjJNT_FREE

    def save_plot(self, x, torque, velocity):

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        # ---- Torque plot ----
        ax1.plot(x, torque)
        ax1.set_ylabel('Torque (Nm)')
        ax1.set_title('Elbow Joint Torque')
        ax1.grid()

        # ---- Velocity plot ----
        ax2.plot(x, velocity)
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Tendon Velocity (m/s)')
        ax2.set_title('Tendon Velocity')
        ax2.grid()

        fig.tight_layout()
        fig.savefig("torque_velocity_b_1_a_0.01.png")

        rospy.loginfo("PLOT SAVED")

    def setup_mappings(self):
        self.joint_map = {}
        # Map robot joints
        for i in range(self.model.njnt): 
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and name != "target_freejoint":
                self.joint_map[name] = self.model.jnt_qposadr[i]

        # Find the cup joint address
        # 'cup_joint' is the name we gave it in the XML <freejoint> tag
        cup_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_freejoint")
        if cup_jnt_id != -1:
            self.cup_qpos_adr = self.model.jnt_qposadr[cup_jnt_id]
        else:
            rospy.logwarn("Could not find 'cup_joint' in XML!")
            self.cup_qpos_adr = -1

    def _capture_joint_state(self):
        state = {}
        if not hasattr(self, "joint_map"):
            return state
        for name, addr in self.joint_map.items():
            if 0 <= addr < len(self.data.qpos):
                q = float(self.data.qpos[addr])
                v = float(self.data.qvel[addr]) if 0 <= addr < len(self.data.qvel) else 0.0
                state[name] = (q, v)
        return state

    def _restore_joint_state(self, state):
        for name, (q, v) in state.items():
            if name in self.joint_map:
                addr = self.joint_map[name]
                if 0 <= addr < len(self.data.qpos):
                    self.data.qpos[addr] = q
                if 0 <= addr < len(self.data.qvel):
                    self.data.qvel[addr] = v
        self._apply_latest_object_pose()
        mujoco.mj_forward(self.model, self.data)

    def _apply_latest_object_pose(self):
        if self.latest_object_pose is None or self.cup_qpos_adr == -1:
            return
        start = self.cup_qpos_adr
        p = self.latest_object_pose
        self.data.qpos[start+0] = p[0]
        self.data.qpos[start+1] = p[1]
        self.data.qpos[start+2] = p[2]
        self.data.qpos[start+3] = p[3]
        self.data.qpos[start+4] = p[4]
        self.data.qpos[start+5] = p[5]
        self.data.qpos[start+6] = p[6]

    def load_active_model(self, xml_path, descriptor_path, preserve_state=True):
        prev_state = self._capture_joint_state() if (preserve_state and hasattr(self, "data")) else {}

        rospy.loginfo(f"Loading active model: {xml_path}")
        spec = mujoco.MjSpec()
        spec.from_file(xml_path)
        model = spec.compile()
        data = mujoco.MjData(model)

        self.spec = spec
        self.model = model
        self.data = data
        self.Virtual_Descriptor = self.load_virtual_descriptor(filepath=descriptor_path)

        self.setup_mappings()
        self.setup_sensor_mappings()
        self.tendon_id_map(Virtual_Descriptor=self.Virtual_Descriptor)

        if preserve_state and len(prev_state) > 0:
            self._restore_joint_state(prev_state)
        else:
            self._apply_latest_object_pose()
            mujoco.mj_forward(self.model, self.data)

        rospy.loginfo(f"Loaded VMC descriptor: {descriptor_path}")

    def switch_to_main_phase(self, reason):
        if self.control_phase == "main":
            return
        self.load_active_model(self.main_robot_urdf_path, self.main_vm_config_path, preserve_state=True)
        self.control_phase = "main"
        self.pose_error_below_since = None
        rospy.loginfo(f"Switched to main phase ({reason})")

    def get_VMC_torque(self, descriptor=None, enable_lifter_schedule=True): # 
        tau_total = np.zeros(self.model.nv)
        if descriptor is None:
            descriptor = self.Virtual_Descriptor

        for spring in descriptor:

            if spring["name"] == "left_lifter" or spring["name"] == "right_lifter":
                if enable_lifter_schedule and self.real_time < 5:
                    continue

            ID = self.tendon_name_to_id[spring["name"]]
            # rospy.loginfo(f"spring name {spring['name']}, ID: {ID}")
            J = self.get_tendon_jacobian(self.data)[ID]
            l = self.data.ten_length[ID]
            v = self.data.ten_velocity[ID]
            # v = self.compute_tendon_velocity(tendon_id= ID, current_length= l)
            rospy.loginfo_throttle(0.5, f"{spring['name']} length: {l}, Velocity: {v} ")
            k = spring.get("constant", spring.get("cosntant", 10))
            b = spring.get("damping", 1)
            l_0 = spring.get("rest_length", 0)
            force = self.get_tendon_force(l, l_0, k, v, b) 
            force = force * 0.2
            if enable_lifter_schedule and (spring["name"] == "left_lifter" or spring["name"] == "right_lifter"):
                force = force * min( (self.real_time - 5 ) / 5 * 20, 35)
            rospy.loginfo_throttle(0.5, f"{spring['name']} force: {force} ")
            # force = max(min(force, 10), -10)
            tau_total += J * force

        tau_total = np.clip(tau_total, -3.0, 3.0) 

        return tau_total[:21] 


    def get_tendon_jacobian(self, data):
        return data.ten_J.copy()

    def get_tendon_force(self, l, l_0, k , v, b=1):
        force = -k * (l - l_0) - v*b
        return force

    def setup_sensor_mappings(self):
        self.sensor_map = {}
        for i in range(self.model.nsensor):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, i
            )
            self.sensor_map[name] = i
            rospy.loginfo(f"Sensor {i}: {name}")

    def pose_callback(self, msg):
        p = msg.pose
        self.latest_object_pose = (
            p.position.x, p.position.y, p.position.z,
            p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z
        )
        self._apply_latest_object_pose()


    def joint_callback(self, msg):
        for name, pos, vel in zip(msg.name, msg.position, msg.velocity):
            if name in self.joint_map:
                addr = self.joint_map[name]
                self.data.qpos[addr] = pos

                self.data.qvel[addr] = vel
                # self.joint_damping = - vel * 10

    def control_loop(self, event):
        # if self.loop_time == 20000:
        #     self.save_plot(x = self.time, torque = self.total_torque, velocity=self.velocity_plot)
        #     rospy.loginfo("Reached loop limit — shutting down.")
        #     rospy.signal_shutdown("Finished experiment")

        mujoco.mj_forward(self.model, self.data)

        if self.control_phase == "init_pose":
            tau = self.get_VMC_torque(enable_lifter_schedule=False) - self.data.qvel[:21] * 0.2

            pose_err = self.compute_pose_error_rms()
            now_t = self.real_time
            if pose_err < self.pose_error_threshold:
                if self.pose_error_below_since is None:
                    self.pose_error_below_since = now_t
                elif (now_t - self.pose_error_below_since) >= self.pose_settle_time:
                    self.switch_to_main_phase(
                        reason=f"pose reached: err={pose_err:.4f}, t={now_t:.2f}s"
                    )
            else:
                self.pose_error_below_since = None

            if now_t >= self.pose_max_time:
                self.switch_to_main_phase(
                    reason=f"timeout: err={pose_err:.4f}, t={now_t:.2f}s"
                )
        else:
            tau = self.get_VMC_torque() - self.data.qvel[:21] * 0.2 # check spring gain. try low pass filter on joint velocities. 
        
        tau_left = tau[self.left_indices]
        tau_right = tau[self.right_indices]
        tau_neck = tau[self.neck_indices]
        tau_waist = tau[self.waist_indices]
        msg_left = Float64MultiArray()
        msg_left.data = tau_left.tolist()
        msg_right = Float64MultiArray()
        msg_right.data = tau_right.tolist()
        msg_neck = Float64MultiArray()
        msg_neck.data = tau_neck.tolist()
        msg_waist = Float64MultiArray()
        msg_waist.data = tau_waist.tolist()

        self.pub_left.publish(msg_left)
        self.pub_right.publish(msg_right)
        self.pub_neck.publish(msg_neck)
        self.pub_waist.publish(msg_waist)

        # Loop timing diagnostics

        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.real_time += dt
        self.last_time = current_time
        self.time.append(self.loop_time)
        self.total_torque.append(tau_left[3])
        self.loop_time += 1
        rospy.loginfo_throttle(1.0, f"Loop dt: {dt:.6f}  |  Hz: {1.0/dt:.1f}")


        ##################### Developing ###################
    def viewer_loop(self, event):
        if self.viewer.is_running():
            self.viewer.sync()

    def run_viewer(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running() and not rospy.is_shutdown():
                rate = rospy.Rate(500)  

                viewer.sync()

                rate.sleep()


    def run_passive(self): 
        if self.loop_time == 200:
            self.save_plot(x = self.time, torque = self.total_torque, velocity=self.velocity_plot)
            # Running = False
        # rate = rospy.Rate(500)  
        mujoco.mj_forward(self.model, self.data)


        tau = self.get_VMC_torque()
        rounded_tau = np.round(tau, 2)  # 3 decimal places
        # rospy.loginfo_throttle(0.5, f"Left arm torques: {rounded_tau}")
        left_indices = [12, 13, 14, 15, 16, 17, 18]

        tau_left = tau[left_indices]
        # tau_left = np.clip(tau_left, -1.0, 1.0) # clip torques for safety.
        assert len(tau_left) == 7
        
        msg_left = Float64MultiArray()
        msg_left.data = tau_left.tolist()

        msg_right = Float64MultiArray()
        msg_right.data = [0.0]*7

        msg_neck = Float64MultiArray()
        msg_neck.data = [0.0]*2

        msg_waist = Float64MultiArray()
        msg_waist.data = [0.0]

        self.pub_left.publish(msg_left)
        self.pub_right.publish(msg_right)
        self.pub_neck.publish(msg_neck)
        self.pub_waist.publish(msg_waist)
        
        self.time.append(self.loop_time)
        self.total_torque.append(tau_left[3])
        self.loop_time += 1
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        rospy.loginfo_throttle(1.0, f"Loop dt: {dt:.6f}  |  Hz: {1.0/dt:.1f}")
        rospy.loginfo_throttle(0.5, f"in control_loop")
        rospy.loginfo_throttle(0.5, f"{self.loop_time}")


        # rate.sleep()
    
    def run(self): # cant figure out rendering issue. Large fluctations in publish frequency when trying to render. 

        rate = rospy.Rate(500)  # control rate
        render_divider = 0

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        while not rospy.is_shutdown() and self.viewer.is_running():

            # ---- Simulation / Control ----
            mujoco.mj_forward(self.model, self.data)

            mujoco.mj_forward(self.model, self.data)

            tau = self.get_VMC_torque()
            left_indices = [12, 13, 14, 15, 16, 17, 18]
            tau_left = tau[left_indices]

            msg_left = Float64MultiArray()
            msg_left.data = tau_left.tolist()

            self.pub_left.publish(msg_left)
            self.pub_right.publish(Float64MultiArray(data=[0.0]*7))
            self.pub_neck.publish(Float64MultiArray(data=[0.0]*2))
            self.pub_waist.publish(Float64MultiArray(data=[0.0]))

            # Loop timing diagnostics

            current_time = rospy.Time.now()
            dt = (current_time - self.last_time).to_sec()
            self.last_time = current_time
            self.time.append(self.loop_time)
            self.total_torque.append(tau_left[3])
            self.loop_time += 1
            rospy.loginfo_throttle(1.0, f"Loop dt: {dt:.6f}  |  Hz: {1.0/dt:.1f}")

            # ---- Render at lower rate ----
            render_divider += 1
            if render_divider >= 500 // 1:  # ~30 Hz
                # self.viewer.sync()
                render_divider = 0

            rate.sleep()



if __name__ == '__main__':
    try:
        MujocoVizNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
