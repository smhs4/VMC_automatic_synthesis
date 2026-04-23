#!/usr/bin/env python3

import rospy
import mujoco
import mujoco.viewer
import numpy as np
import os
import datetime
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

        # self.robot_urdf_path = os.path.join(pkg_path, 'urdf', 'sciurus17_tendon_test.xml')

        
        self.robot_urdf_path = os.path.join(pkg_path, 'urdf', 'best_model_33.xml')
        self.vm_config_path = os.path.join(pkg_path,'config','best_model_tendons_33.json')

        self.cup_mesh_path = os.path.join(pkg_path, 'urdf', 'cup.stl')

        self.tendon_k = [100]
        self.tendon_l0 = [0.15]

        self.real_time = 0

        # Load Robot
        try:
            rospy.loginfo(f"Loading robot from: {self.robot_urdf_path}")
            self.spec = mujoco.MjSpec()
            self.spec.from_file(self.robot_urdf_path)
        except Exception as e:
            rospy.logerr(f"Failed to load robot URDF: {e}")
            return

        # Add Cup
        # self.add_cup_to_spec()

        # Compile
        try:
            self.model = self.spec.compile()
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            rospy.logerr(f"Compilation Error: {e}")
            return

        # Setup ID Mappings
        self.Virtual_Descriptor = self.load_virtual_descriptor(filepath = self.vm_config_path)
        self.setup_mappings()

        self.setup_sensor_mappings()

        self.tendon_id_map(Virtual_Descriptor = self.Virtual_Descriptor)
        rospy.loginfo(self.Virtual_Descriptor)

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

        self.timer = 0
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
        ########## Logger variable setup ###########
        self.plot_directory = os.path.join(pkg_path,'plots_h')

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.plot_directory = os.path.join(self.plot_directory, f"run_{timestamp}")

        os.makedirs(self.plot_directory, exist_ok=True)
        self.torque_vmc = []
        self.torque_gc = []
        self.joint_positions = []
        self.object_position = []
        self.logger_time = []
        self.real_joint_positions_log = []
        self.real_joint_positions_time = []
        self.body_id = self.model.body("target").id
        self.joint_state_received = False

        # PID-before-VMC stage
        self.vmc_active = False
        self.pid_start_time = None
        self.pid_integral = np.zeros(21, dtype=np.float64)
        self.pid_prev_time = None
        self.pid_activate_timeout = 4.0
        self.pid_position_tol = 0.08
        self.pid_velocity_tol = 0.35
        self.initial_pose_joint_names = [
            "waist_yaw_joint", "neck_yaw_joint", "neck_pitch_joint",
            "r_arm_joint1", "r_arm_joint2", "r_arm_joint3", "r_arm_joint4",
            "r_arm_joint5", "r_arm_joint6", "r_arm_joint7", "r_hand_mimic_joint",
            "l_arm_joint1", "l_arm_joint2", "l_arm_joint3", "l_arm_joint4",
            "l_arm_joint5", "l_arm_joint6", "l_arm_joint7", "l_hand_mimic_joint",
        ]
        self.initial_pose_joint_values = np.array([
            0.05062136600022616, -0.3436116964863836, -1.3959225169759335,
            0.10584467436410924, -1.5539225381281545, 0.09817477042468103, 2.741223667951641,
            0.02761165418194154, -1.7916895602504288, -1.6122138080678088, 0.3,
            -0.0798, 1.53, 0.145, -2.26,
            0.0372, 1.07, -2.0, -0.3
        ], dtype=np.float64)
        self.initial_qpos_target = self._build_initial_qpos_target(
            self.initial_pose_joint_names,
            self.initial_pose_joint_values
        )
        self.controlled_dof_labels = self._build_controlled_dof_labels()
        # Per-DOF PID and limits (by joint name) to account for different inertia/mass.
        # Tune these dictionaries to reduce oscillation on specific joints.
        self.pid_kp = self._build_per_dof_vector(
            default_value=12.0,
            overrides={
                "waist_yaw_joint": 16.0,
                "neck_yaw_joint": 8.0,
                "neck_pitch_joint": 8.0,
                "r_arm_joint1": 18.0, "r_arm_joint2": 18.0, "r_arm_joint3": 16.0, "r_arm_joint4": 14.0,
                "r_arm_joint5": 12.0, "r_arm_joint6": 10.0, "r_arm_joint7": 10.0, "r_hand_mimic_joint": 6.0,
                "l_arm_joint1": 18.0, "l_arm_joint2": 18.0, "l_arm_joint3": 16.0, "l_arm_joint4": 14.0,
                "l_arm_joint5": 12.0, "l_arm_joint6": 10.0, "l_arm_joint7": 10.0, "l_hand_mimic_joint": 6.0,
            },
        )
        self.pid_ki = self._build_per_dof_vector(
            default_value=0.2,
            overrides={
                "waist_yaw_joint": 0.05,
                "neck_yaw_joint": 0.05,
                "neck_pitch_joint": 0.05,
                "r_hand_mimic_joint": 0.02,
                "l_hand_mimic_joint": 0.02,
            },
        )
        self.pid_kd = self._build_per_dof_vector(
            default_value=1.2,
            overrides={
                "waist_yaw_joint": 2.2,
                "neck_yaw_joint": 1.0,
                "neck_pitch_joint": 1.0,
                "r_arm_joint1": 2.4, "r_arm_joint2": 2.4, "r_arm_joint3": 2.0, "r_arm_joint4": 1.8,
                "r_arm_joint5": 1.5, "r_arm_joint6": 1.2, "r_arm_joint7": 1.1, "r_hand_mimic_joint": 0.8,
                "l_arm_joint1": 2.4, "l_arm_joint2": 2.4, "l_arm_joint3": 2.0, "l_arm_joint4": 1.8,
                "l_arm_joint5": 1.5, "l_arm_joint6": 1.2, "l_arm_joint7": 1.1, "l_hand_mimic_joint": 0.8,
            },
        )
        self.pid_tau_clip = self._build_per_dof_vector(
            default_value=2.5,
            overrides={
                "waist_yaw_joint": 2.0,
                "neck_yaw_joint": 1.0,
                "neck_pitch_joint": 1.0,
                "r_hand_mimic_joint": 0.6,
                "l_hand_mimic_joint": 0.6,
            },
        )
        self.pid_integral_clip = self._build_per_dof_vector(
            default_value=1.0,
            overrides={
                "waist_yaw_joint": 0.3,
                "neck_yaw_joint": 0.2,
                "neck_pitch_joint": 0.2,
                "r_hand_mimic_joint": 0.1,
                "l_hand_mimic_joint": 0.1,
            },
        )
        ############################################
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

    def _build_controlled_dof_labels(self):
        labels = []
        for dof in range(min(21, self.model.nv)):
            jnt_id = int(self.model.dof_jntid[dof])
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            labels.append(name if name is not None else f"dof_{dof}")
        return labels

    def _build_per_dof_vector(self, default_value, overrides=None, overrides_by_joint=None):
        vec = np.full(21, float(default_value), dtype=np.float64)
        if overrides_by_joint is None:
            overrides_by_joint = overrides if overrides is not None else {}
        for i, jname in enumerate(self.controlled_dof_labels):
            if jname in overrides_by_joint:
                vec[i] = float(overrides_by_joint[jname])
        return vec

    def _build_initial_qpos_target(self, joint_names, joint_values):
        target = self.model.qpos0[:21].copy()
        if len(joint_names) != len(joint_values):
            rospy.logwarn(
                f"Initial pose name/value count mismatch: {len(joint_names)} names vs {len(joint_values)} values. Using zipped pairs."
            )
        for j_name, j_val in zip(joint_names, joint_values):
            if j_name not in self.joint_map:
                rospy.logwarn(f"Initial pose joint '{j_name}' not found in joint_map; skipping.")
                continue
            qadr = int(self.joint_map[j_name])
            if qadr < 0 or qadr >= target.shape[0]:
                rospy.logwarn(f"Initial pose joint '{j_name}' qpos adr {qadr} outside target[0:{target.shape[0]}]; skipping.")
                continue
            target[qadr] = float(j_val)
        return target

    def _split_and_publish_tau(self, tau):
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

    def get_pid_torque_to_initial_pose(self):
        now = rospy.Time.now().to_sec()
        if self.pid_prev_time is None:
            self.pid_prev_time = now
        dt = max(now - self.pid_prev_time, 1e-4)
        self.pid_prev_time = now

        q = self.data.qpos[:21].copy()
        qd = self.data.qvel[:21].copy()
        err = self.initial_qpos_target - q

        self.pid_integral += err * dt
        self.pid_integral = np.clip(self.pid_integral, -self.pid_integral_clip, self.pid_integral_clip)

        tau = (
            self.pid_kp * err
            + self.pid_ki * self.pid_integral
            - self.pid_kd * qd
        )
        tau = np.clip(tau, -self.pid_tau_clip, self.pid_tau_clip)
        return tau, err, qd
    def log_values(self, torque_vmc, torque_gc, joint_positions, object_position, time_stamp):
        self.torque_vmc.append(torque_vmc)
        self.torque_gc.append(torque_gc)
        self.joint_positions.append(joint_positions)
        self.object_position.append(object_position)
        self.logger_time.append(time_stamp)
        return

    def log_save(self):
        log_dict = {
            "time": list(self.logger_time),

            "torque_vmc": np.array(self.torque_vmc).T.tolist(),
            "torque_gc": np.array(self.torque_gc).T.tolist(),

            "joint_positions": np.array(self.joint_positions).T.tolist(),
            "object_position": np.array(self.object_position).T.tolist()
        }

        file_path = os.path.join(self.plot_directory, "experiment_log.json")

        with open(file_path, "w") as f:
            json.dump(log_dict, f, indent=4)

        rospy.loginfo(f"Log saved to {file_path}")
        return
    
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

    def get_VMC_torque(self): # 
        tau_total = np.zeros(self.model.nv)

        for spring in self.Virtual_Descriptor:

            if spring["name"] == "left_lifter" or spring["name"] == "right_lifter" or spring["name"] == "left_end_lifter" or spring["name"] == "right_end_lifter":
                if self.real_time < 5:
                    continue

            ID = self.tendon_name_to_id[spring["name"]]
            # rospy.loginfo(f"spring name {spring['name']}, ID: {ID}")
            J = self.get_tendon_jacobian(self.data)[ID]
            l = self.data.ten_length[ID]
            v = self.data.ten_velocity[ID]
            
            # rospy.loginfo_throttle(0.5, f"{spring['name']} length: {l}, Velocity: {v} ")
            k = spring.get("cosntant", 0)
            rospy.loginfo_throttle(0.5, f"{spring['name']} length: {l}, Velocity: {v}, k: {k} ")
            b = spring.get("damping", 1)
            l_0 = spring.get("rest_length", 0)
            force = self.get_tendon_force(l, l_0, k, v, b) * 0.2
            
            # jitter
            # multiply = 1            
            # if self.real_time - self.timer > 0.03:
            #     self.timer = self.real_time
            #     multiply = 30
            # force = force * multiply 

            if spring["name"] == "left_lifter" or spring["name"] == "right_lifter" or spring["name"] == "left_end_lifter" or spring["name"] == "right_end_lifter":
                force = force * min( (self.real_time - 5 ) / 10, 30)
                # if spring["name"] == "left_end_lifter" or spring["name"] == "right_end_lifter":
                #     force = 0 
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
        if self.cup_qpos_adr != -1:
            start = self.cup_qpos_adr
            
            # CHANGE 3: Access data via 'msg.pose'
            # (PoseStamped has header and pose, we need the pose part)
            p = msg.pose
            
            self.data.qpos[start+0] = p.position.x
            self.data.qpos[start+1] = p.position.y
            self.data.qpos[start+2] = p.position.z 
            
            self.data.qpos[start+3] = p.orientation.w
            self.data.qpos[start+4] = p.orientation.x
            self.data.qpos[start+5] = p.orientation.y
            self.data.qpos[start+6] = p.orientation.z


    def joint_callback(self, msg):
        updated_any = False
        for name, pos, vel in zip(msg.name, msg.position, msg.velocity):
            if name in self.joint_map:
                addr = self.joint_map[name]
                self.data.qpos[addr] = pos

                self.data.qvel[addr] = vel
                updated_any = True
                # self.joint_damping = - vel * 10
        if updated_any:
            self.joint_state_received = True

    def control_loop(self, event):
        if self.real_time >= 10:

            joint_positions = np.array(self.joint_positions).T
            rospy.loginfo(f"joint list for plotting: {self.torque_vmc}")
            self.save_plot(
                x=self.logger_time,
                y_list=joint_positions,
                title="Joint Positions Plot",
                xlabel="Time (Seconds)",
                ylabel="Joint Angle (rad)",
                labels=self.controlled_dof_labels,
            )
            torque_vmc = np.array(self.torque_vmc).T
            self.save_plot(
                x=self.logger_time,
                y_list=torque_vmc,
                title="VMC Torque Plot",
                xlabel="Time (Seconds)",
                ylabel="Joint Torque (Nm)",
                labels=self.controlled_dof_labels,
            )
            body_position = np.array(self.object_position).T
            rospy.loginfo(f"body postiion for plotting: {body_position}")
            self.save_3d_plot(x = body_position[0], y = body_position[1], z = body_position[2], title = "Body Positions Plot")
          
            rospy.loginfo(f"Experiment Complete. Plots can be Viewed: {self.plot_directory}")
            self.log_save()
            rospy.signal_shutdown("Exiting........")


        mujoco.mj_forward(self.model, self.data)

        if not self.joint_state_received:
            rospy.logwarn_throttle(1.0, "Waiting for joint state before running PID/VMC.")
            return

        if not self.vmc_active:
            if self.pid_start_time is None:
                self.pid_start_time = self.real_time
            tau, err, qd = self.get_pid_torque_to_initial_pose()
            settled = (np.max(np.abs(err)) < self.pid_position_tol) and (np.max(np.abs(qd)) < self.pid_velocity_tol)
            timed_out = (self.real_time - self.pid_start_time) >= self.pid_activate_timeout
            if settled or timed_out:
                self.vmc_active = True
                reason = "settled" if settled else "timeout"
                rospy.loginfo(
                    f"Switching PID -> VMC at t={self.real_time:.2f}s ({reason}), "
                    f"max|err|={np.max(np.abs(err)):.4f}, max|qd|={np.max(np.abs(qd)):.4f}"
                )
        else:
            tau = self.get_VMC_torque() # - self.data.qvel[:21] * 0.15 # check spring gain. try low pass filter on joint velocities.

        self._split_and_publish_tau(tau)

        # Loop timing diagnostics

        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.real_time += dt
        self.last_time = current_time
        self.time.append(self.loop_time)
        self.total_torque.append(float(tau[self.left_indices[3]]))
        self.loop_time += 1
        rospy.loginfo_throttle(1.0, f"Loop dt: {dt:.6f}  |  Hz: {1.0/dt:.1f}")
        

        ############# loop logging ##############3
        self.log_values(
            torque_vmc = tau,
            torque_gc = 0,
            joint_positions = self.data.qpos[0:21].copy(),
            object_position = self.data.xpos[self.body_id].copy(),
            time_stamp= self.real_time
        )

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

    def save_plot(self, x, y_list, title, xlabel=None, ylabel=None, labels=None):

        fig, ax = plt.subplots(figsize=(8,6))

        # Plot each y series
        for i, y in enumerate(y_list):
            label = None
            if labels is not None and i < len(labels):
                label = labels[i]
            elif labels is not None:
                label = f"Series {i}"
            ax.plot(x, y, label=label)

        # Titles and labels
        ax.set_title(title)

        if xlabel is not None:
            ax.set_xlabel(xlabel)

        if ylabel is not None:
            ax.set_ylabel(ylabel)

        ax.grid()
        if labels is not None:
            ax.legend(loc="best", ncol=3, fontsize=7, framealpha=0.7)

        # Ensure plot directory exists
        os.makedirs(self.plot_directory, exist_ok=True)

        # Create filename from title
        filename = title.replace(" ", "_") + ".png"
        filepath = os.path.join(self.plot_directory, filename)

        fig.tight_layout()
        fig.savefig(filepath)
        plt.close(fig)

        rospy.loginfo(f"PLOT SAVED: {filepath}")

    def save_3d_plot(self, x, y, z, title, xlabel="x", ylabel="y", zlabel="z"):



        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectory
        ax.plot(x, y, z, label="Trajectory")

        # Mark start point with cross
        ax.scatter(x[0], y[0], z[0], marker='x', s=100, label="Start")

        # Titles and labels
        ax.set_title(title)

        if xlabel is not None:
            ax.set_xlabel(xlabel)

        if ylabel is not None:
            ax.set_ylabel(ylabel)

        if zlabel is not None:
            ax.set_zlabel(zlabel)

        ax.grid()
        ax.legend(loc="best")

        # Set isometric view
        ax.view_init(elev=25, azim=225)

        # Equal axis scaling (important for trajectories)
        ax.set_box_aspect([1,1,1])
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.1, 0.5)
        # Ensure plot directory exists
        os.makedirs(self.plot_directory, exist_ok=True)

        # Create filename from title
        filename = title.replace(" ", "_") + ".png"
        filepath = os.path.join(self.plot_directory, filename)

        fig.tight_layout()
        fig.savefig(filepath)
        plt.close(fig)

        rospy.loginfo(f"PLOT SAVED: {filepath}")


if __name__ == '__main__':
    try:
        MujocoVizNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
