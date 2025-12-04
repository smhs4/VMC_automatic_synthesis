#!/usr/bin/env python3
"""
Object-Oriented MuJoCo Simulation Template

This module provides convenient wrapper classes for MuJoCo simulations, making it easy to:
- Access joints, bodies, and actuators by name
- Get/set positions, velocities, forces, etc. without managing indices
- Run simulations with passive or active viewers
- Define control callbacks in a clean way
- Monitor contacts between bodies with contact sensors

Usage Example:
    sim = MuJoCoSimulation("path/to/model.xml")
    
    # Register entities you want to work with
    crane = sim.add_joint("crane", "crane_base__up_down")
    red_box = sim.add_body("red_box", "red_box")
    lift_actuator = sim.add_actuator("lift", "lift")
    
    # Add contact sensor
    sensor = sim.reg_contact_sensor("box_contacts", "red_box", ["gripper_left", "gripper_right"])
    
    # Access properties easily
    crane.qpos = 0.5
    velocity = red_box.qvel
    
    # Check contacts
    contacts = sensor.get_contacts()
    for contact in contacts:
        print(f"Contact with {contact.body_name}: {contact.total_force:.2f} N")
    
    # Define control callback
    def my_control(sim):
        lift_actuator.ctrl = compute_target(crane.qpos)
    
    sim.run(control_callback=my_control, passive=True)
"""

import mujoco
from mujoco import viewer
import numpy as np
import argparse
from typing import Optional, Callable, Dict, List, Union
from dataclasses import dataclass
import time
import importlib


# ============================================================================
# Utility Functions (quaternions, etc.)
# ============================================================================

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (w, x, y, z format)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Compute quaternion conjugate."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to axis-angle representation (small angles)."""
    return 2.0 * q[1:]  # x, y, z components


def quat_error(q_current: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    """Compute orientation error from current to target quaternion."""
    q_err = quat_mul(quat_conj(q_current), q_target)
    return quat_to_axis_angle(q_err)


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z).
    
    Args:
        roll: Rotation around x-axis (radians)
        pitch: Rotation around y-axis (radians)
        yaw: Rotation around z-axis (radians)
    
    Returns:
        Quaternion as [w, x, y, z]
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


# ============================================================================
# Entity Wrapper Classes
# ============================================================================

class Joint:
    """Wrapper for a MuJoCo joint with convenient property access."""
    
    def __init__(self, name: str, model: mujoco.MjModel, data: mujoco.MjData):
        self.name = name
        self.model = model
        self.data = data
        
        # Get joint ID and addresses
        self.id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if self.id == -1:
            raise ValueError(f"Joint '{name}' not found in model.")
        self.qpos_adr = model.jnt_qposadr[self.id]
        self.qvel_adr = model.jnt_dofadr[self.id]
        
        # Joint type and size
        self.type = model.jnt_type[self.id]
        self.nq = 7 if self.type == mujoco.mjtJoint.mjJNT_FREE else 1
        self.nv = 6 if self.type == mujoco.mjtJoint.mjJNT_FREE else 1
    

    @property
    def qpos(self) -> np.ndarray:
        """Get joint position(s)."""
        return self.data.qpos[self.qpos_adr:self.qpos_adr + self.nq].copy()
    
    @qpos.setter
    def qpos(self, value):
        """Set joint position(s)."""
        # Convert to numpy array if needed
        value = np.atleast_1d(value)
        
        # Check correct number of values
        if value.size != self.nq:
            raise ValueError(f"Expected {self.nq} values for joint '{self.name}', got {value.size}")
        self.data.qpos[self.qpos_adr:self.qpos_adr + self.nq] = value
    
    @property
    def qvel(self) -> np.ndarray:
        """Get joint velocity(ies)."""
        return self.data.qvel[self.qvel_adr:self.qvel_adr + self.nv].copy()
    
    @qvel.setter
    def qvel(self, value):
        """Set joint velocity(ies)."""
        # Convert to numpy array if needed
        value = np.atleast_1d(value)
        
        # Check correct number of values
        if value.size != self.nv:
            raise ValueError(f"Expected {self.nv} values for joint '{self.name}', got {value.size}")
        
        self.data.qvel[self.qvel_adr:self.qvel_adr + self.nv] = value
    
    @property
    def qfrc_applied(self) -> np.ndarray:
        """Get applied forces at this joint."""
        return self.data.qfrc_applied[self.qvel_adr:self.qvel_adr + self.nv].copy()
    
    @qfrc_applied.setter
    def qfrc_applied(self, value):
        """Set applied forces at this joint."""
        # Convert to numpy array if needed
        value = np.atleast_1d(value)
        
        # Check correct number of values
        if value.size != self.nv:
            raise ValueError(f"Expected {self.nv} values for joint '{self.name}', got {value.size}")
        
        self.data.qfrc_applied[self.qvel_adr:self.qvel_adr + self.nv] = value
    
    def add_qfrc(self, force: np.ndarray):
        """Add force to the joint (useful for control)."""
        self.data.qfrc_applied[self.qvel_adr:self.qvel_adr + self.nv] += force


class Body:
    """Wrapper for a MuJoCo body with convenient property access."""
    
    def __init__(self, name: str, model: mujoco.MjModel, data: mujoco.MjData):
        self.name = name
        self.model = model
        self.data = data
        
        # Get body ID
        self.id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        
        # Get DOF address (if body has joints)
        self.dof_adr = model.body_dofadr[self.id]
        self.dof_num = model.body_dofnum[self.id]
    
    @property
    def pos(self) -> np.ndarray:
        """Get body position in world frame."""
        return self.data.xpos[self.id].copy()
    
    @property
    def quat(self) -> np.ndarray:
        """Get body orientation quaternion (w, x, y, z)."""
        return self.data.xquat[self.id].copy()
    
    @property
    def xmat(self) -> np.ndarray:
        """Get body orientation as rotation matrix."""
        return self.data.xmat[self.id].reshape(3, 3).copy()
    
    @property
    def cvel(self) -> np.ndarray:
        """Get body center-of-mass velocity (angular, linear)."""
        return self.data.cvel[self.id].copy()
    
    @property
    def angular_vel(self) -> np.ndarray:
        """Get body angular velocity (first 3 components of cvel)."""
        return self.data.cvel[self.id, 0:3].copy()
    
    @property
    def linear_vel(self) -> np.ndarray:
        """Get body linear velocity (last 3 components of cvel)."""
        return self.data.cvel[self.id, 3:6].copy()
    
    def add_force(self, force: np.ndarray):
        """Add linear force to body (for free joints). Force in world frame."""
        if self.dof_num >= 6:  # Free joint has 6 DOFs (3 rotation + 3 translation)
            # Linear forces go in indices 3:6
            self.data.qfrc_applied[self.dof_adr+3:self.dof_adr+6] += force
    
    def add_torque(self, torque: np.ndarray):
        """Add torque to body (for free joints). Torque in world frame."""
        if self.dof_num >= 6:  # Free joint has 6 DOFs (3 rotation + 3 translation)
            # Angular torques go in indices 0:3
            self.data.qfrc_applied[self.dof_adr+0:self.dof_adr+3] += torque


class Actuator:
    """Wrapper for a MuJoCo actuator with convenient property access."""
    
    def __init__(self, name: str, model: mujoco.MjModel, data: mujoco.MjData):
        self.name = name
        self.model = model
        self.data = data
        
        # Get actuator ID
        self.id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    
    @property
    def ctrl(self) -> float:
        """Get actuator control signal."""
        return self.data.ctrl[self.id]
    
    @ctrl.setter
    def ctrl(self, value: float):
        """Set actuator control signal."""
        self.data.ctrl[self.id] = value
    
    @property
    def force(self) -> np.ndarray:
        """Get actuator force."""
        return self.data.actuator_force[self.id]


class Site:
    """Wrapper for a MuJoCo site with convenient property access."""
    
    def __init__(self, name: str, model: mujoco.MjModel, data: mujoco.MjData):
        self.name = name
        self.model = model
        self.data = data
        
        # Get site ID
        self.id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    
    @property
    def pos(self) -> np.ndarray:
        """Get site position in world frame."""
        return self.data.site_xpos[self.id].copy()
    
    @property
    def mat(self) -> np.ndarray:
        """Get site orientation matrix."""
        return self.data.site_xmat[self.id].reshape(3, 3).copy()
    
class Tendon:
    """Wrapper for a MuJoCo tendon with convenient property access."""
    
    def __init__(self, name: str, model: mujoco.MjModel, data: mujoco.MjData):
        self.name = name
        self.model = model
        self.data = data
        
        # Get tendon ID
        self.id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name)
    
    @property
    def length(self) -> float:
        """Get tendon length."""
        return self.data.ten_length[self.id]
    
    @property
    def velocity(self) -> float:
        """Get tendon velocity (rate of length change)."""
        return self.data.ten_velocity[self.id]
    
    @property
    def force(self) -> float:
        """
        Get force in tendon.
        Note: Tendon force is computed from spring/damper properties:
        force = stiffness * (length - springlength) + damping * velocity
        """
        # Handle both scalar and array values
        sl_raw = self.springlength
        k_raw = self.stiffness
        d_raw = self.damping
        
        # Extract scalar values (handles both np.ndarray and np.float64/np.float32)
        sl = float(np.atleast_1d(sl_raw)[0])
        k = float(np.atleast_1d(k_raw)[0])
        d = float(np.atleast_1d(d_raw)[0])
        
        stretch = self.length - sl if sl > 0 else 0
        return k * stretch + d * self.velocity
    
    @property
    def stiffness(self) -> np.ndarray:
        """Get tendon stiffness (may be array for multi-dimensional tendons)."""
        return self.model.tendon_stiffness[self.id]
    
    @property
    def damping(self) -> np.ndarray:
        """Get tendon damping (may be array for multi-dimensional tendons)."""
        return self.model.tendon_damping[self.id]
    
    @property
    def springlength(self) -> np.ndarray:
        """Get tendon spring rest length (may be array for multi-dimensional tendons)."""
        return self.model.tendon_lengthspring[self.id]


@dataclass
class ContactInfo:
    """Information about a single contact."""
    body_name: str
    body_id: int
    geom_id: int
    normal_force: float
    tangent_force: np.ndarray
    total_force: float
    contact_pos: np.ndarray
    contact_normal: np.ndarray


class ContactSensor:
    """
    Sensor for detecting contacts between a body and other registered bodies.
    
    Tracks which bodies are in contact and the forces involved.
    
    Example usage:
        sensor = sim.add_contact_sensor("target_block", ["left_chopstick", "right_chopstick"])
        
        # Get contact information
        contacts = sensor.get_contacts()
        for contact in contacts:
            print(f"Contact with {contact.body_name}: {contact.total_force:.2f} N")
        
        # Check if specific body is in contact
        if sensor.is_in_contact("left_chopstick"):
            print("Left chopstick is touching the target!")
    """
    
    def __init__(self, body_name: str, monitored_bodies: List[str], 
                 model: mujoco.MjModel, data: mujoco.MjData):
        self.body_name = body_name
        self.monitored_bodies = monitored_bodies
        self.model = model
        self.data = data
        
        # Get body ID
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if self.body_id == -1:
            raise ValueError(f"Body '{body_name}' not found in model.")
        
        # Get geom IDs for this body
        self.geom_ids = []
        for i in range(model.ngeom):
            if model.geom_bodyid[i] == self.body_id:
                self.geom_ids.append(i)
        
        # Get body and geom IDs for monitored bodies
        self.monitored_body_ids = {}
        self.monitored_geom_ids = {}
        
        for mon_body_name in monitored_bodies:
            mon_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mon_body_name)
            if mon_body_id == -1:
                raise ValueError(f"Monitored body '{mon_body_name}' not found in model.")
            
            self.monitored_body_ids[mon_body_name] = mon_body_id
            
            # Get geom IDs for this monitored body
            geom_ids = []
            for i in range(model.ngeom):
                if model.geom_bodyid[i] == mon_body_id:
                    geom_ids.append(i)
            self.monitored_geom_ids[mon_body_name] = geom_ids
    
    def get_contacts(self) -> List[ContactInfo]:
        """
        Get list of current contacts with monitored bodies.
        
        Returns:
            List of ContactInfo objects, one for each monitored body in contact
        """
        contacts = []
        
        # Iterate through all contacts in the simulation
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            
            # Check if this contact involves our body
            our_geom = None
            other_geom = None
            
            if geom1 in self.geom_ids:
                our_geom = geom1
                other_geom = geom2
            elif geom2 in self.geom_ids:
                our_geom = geom2
                other_geom = geom1
            else:
                continue  # This contact doesn't involve our body
            
            # Check if the other geom belongs to a monitored body
            for mon_body_name, mon_geom_ids in self.monitored_geom_ids.items():
                if other_geom in mon_geom_ids:
                    # This is a contact we care about!
                    
                    # Extract contact forces from the frame
                    # contact.frame contains: [normal, tangent1, tangent2] forces
                    normal_force = abs(contact.frame[0])  # Normal force magnitude
                    tangent_force = contact.frame[1:3].copy()  # Tangent forces
                    total_force = np.sqrt(normal_force**2 + np.sum(tangent_force**2))
                    
                    # Contact position and normal in world frame
                    contact_pos = contact.pos.copy()
                    contact_normal = contact.frame[0:3].copy()  # First 3 elements are normal direction
                    
                    contact_info = ContactInfo(
                        body_name=mon_body_name,
                        body_id=self.monitored_body_ids[mon_body_name],
                        geom_id=other_geom,
                        normal_force=normal_force,
                        tangent_force=tangent_force,
                        total_force=total_force,
                        contact_pos=contact_pos,
                        contact_normal=contact_normal
                    )
                    contacts.append(contact_info)
        
        return contacts
    
    def is_in_contact(self, body_name: str) -> bool:
        """
        Check if a specific monitored body is currently in contact.
        
        Args:
            body_name: Name of the monitored body to check
            
        Returns:
            True if in contact, False otherwise
        """
        contacts = self.get_contacts()
        return any(c.body_name == body_name for c in contacts)
    
    def get_total_contact_force(self) -> float:
        """
        Get sum of all contact forces with monitored bodies.
        
        Returns:
            Total contact force magnitude
        """
        contacts = self.get_contacts()
        return sum(c.total_force for c in contacts)
    
    def get_contact_force_by_body(self, body_name: str) -> float:
        """
        Get contact force with a specific monitored body.
        
        Args:
            body_name: Name of the monitored body
            
        Returns:
            Contact force magnitude (0 if not in contact)
        """
        contacts = self.get_contacts()
        for c in contacts:
            if c.body_name == body_name:
                return c.total_force
        return 0.0


# ============================================================================
# Model Builder Class (for dynamic model manipulation)
# ============================================================================

class ModelBuilder:
    """
    Helper class for dynamically building/modifying MuJoCo models using MjSpec.
    
    Example usage:
        builder = ModelBuilder()
        builder.from_xml_path("base_model.xml")
        
        # Add a site to an existing body
        builder.add_site("gripper_site", "gripper_link", pos=[0, 0, 0.05])
        
        # Add another site
        builder.add_site("target_site", "target_box", pos=[0.1, 0, 0])
        
        # Create a tendon between them
        builder.add_tendon("gripper_tendon", 
                          sites=["gripper_site", "target_site"],
                          stiffness=100.0, damping=1.0)
        
        # Compile to get model
        model = builder.compile()
    """
    
    def __init__(self, G):
        importlib.reload(mujoco)
        self.spec: Optional[mujoco.MjSpec] = None
        self.graph = G
        # print(self.spec)
    
    def from_xml_path(self, xml_path: str):
        """Load base model from XML file."""
        self.spec = mujoco.MjSpec.from_file(xml_path)
        return self
    
    def from_xml_string(self, xml_string: str):
        """Load base model from XML string."""
        self.spec = mujoco.MjSpec.from_string(xml_string)
        return self
    
    def add_mesh(self, name: str,
                 file: Optional[str] = None,
                 vertex: Optional[List[List[float]]] = None,
                 face: Optional[List[List[int]]] = None,
                 scale: List[float] = [1, 1, 1],
                 smoothnormal: bool = True) -> 'ModelBuilder':
        """
        Add a mesh asset to the model.
        
        Meshes can be loaded from file (STL, OBJ) or defined with vertices/faces.
        
        Args:
            name: Name for the mesh asset (used in geom mesh="name")
            file: Path to mesh file (STL, OBJ, MSH, etc.)
            vertex: List of vertex positions [[x,y,z], ...] for inline mesh
            face: List of triangle face indices [[v0,v1,v2], ...] for inline mesh
            scale: Scale factors [x, y, z] applied to mesh
            smoothnormal: If True, smooth vertex normals for better visual
        
        Returns:
            self for method chaining
            
        Example:
            # From file
            builder.add_mesh("robot_gripper", file="meshes/gripper.stl", scale=[0.001, 0.001, 0.001])
            
            # Inline definition (simple tetrahedron)
            builder.add_mesh("tetra", 
                           vertex=[[0,0,0], [1,0,0], [0.5,1,0], [0.5,0.5,1]],
                           face=[[0,1,2], [0,1,3], [1,2,3], [0,2,3]])
        """
        if not self.spec:
            raise RuntimeError("Must load a base model first")
        
        # Add mesh to spec's meshes list
        mesh = self.spec.add_mesh()
        mesh.name = name
        mesh.scale = scale
        mesh.smoothnormal = smoothnormal
        
        if file is not None:
            mesh.file = file
        elif vertex is not None and face is not None:
            # Inline mesh definition
            import numpy as np
            mesh.uservert = np.array(vertex, dtype=np.float32).flatten()
            mesh.userface = np.array(face, dtype=np.int32).flatten()
        else:
            raise ValueError("Must provide either 'file' or both 'vertex' and 'face'")
        
        return self
    
    def add_material(self, name: str,
                     texture: Optional[str] = None,
                     rgba: List[float] = [1, 1, 1, 1],
                     specular: float = 0.5,
                     shininess: float = 0.5,
                     reflectance: float = 0.0,
                     emission: float = 0.0) -> 'ModelBuilder':
        """
        Add a material asset for use with meshes and geoms.
        
        Args:
            name: Name for the material
            texture: Name of texture to use (optional)
            rgba: Base color [r, g, b, a]
            specular: Specular reflection coefficient [0-1]
            shininess: Shininess coefficient [0-1]
            reflectance: Reflectance coefficient [0-1]
            emission: Emission coefficient [0-1]
            
        Returns:
            self for method chaining
        """
        if not self.spec:
            raise RuntimeError("Must load a base model first")
        
        material = self.spec.add_material()
        material.name = name
        material.rgba = rgba
        material.specular = specular
        material.shininess = shininess
        material.reflectance = reflectance
        material.emission = emission
        
        if texture:
            material.texture = texture
        
        return self
    
    def _find_body(self, name: str, body=None):
        """Recursively find a body by name in the spec tree."""
        if body is None:
            body = self.spec.worldbody
        
        if hasattr(body, 'name') and body.name == name:
            return body
        
        # Search children
        if hasattr(body, 'bodies'):
            for child in body.bodies:
                result = self._find_body(name, child)
                if result:
                    return result
        
        return None
    
    def _find_site(self, name: str):
        """Find a site by name in the spec."""
        def search_body(body):
            # Check sites in this body
            if hasattr(body, 'sites'):
                for site in body.sites:
                    if hasattr(site, 'name') and site.name == name:
                        return site
            # Search children
            if hasattr(body, 'bodies'):
                for child in body.bodies:
                    result = search_body(child)
                    if result:
                        return result
            return None
        
        return search_body(self.spec.worldbody)
    
    

    def add_site(self, name: str, body_name: str, 
                 pos: List[float] = [0, 0, 0],
                 quat: Optional[List[float]] = None,
                 size: float = 0.01,
                 rgba: List[float] = [1, 0, 0, 1]) -> 'ModelBuilder':
        """
        Add a site to an existing body.
        
        Args:
            name: Name for the new site
            body_name: Name of the body to attach the site to
            pos: Position relative to body [x, y, z]
            quat: Orientation quaternion [w, x, y, z] (optional)
            size: Visual size of site
            rgba: Color [r, g, b, a]
        """
        if not self.spec:
            raise RuntimeError("Must load a base model first")
        
        # Find the body
        body = self._find_body(body_name)
        if not body:
            raise ValueError(f"Body '{body_name}' not found")
        
        # Add site to body
        site = body.add_site()
        site.name = name
        site.pos = pos
        if quat:
            site.quat = quat
        site.size = [size, size, size]  # MjsSite.size expects 3 elements
        site.rgba = rgba
        self.graph.add_node(name)
        
        return self
    
    def add_tendon(self, name: str,
                   sites: List[str],
                   stiffness: float = 0.0,
                   damping: float = 0.0,
                   tendon_type: str = "spatial",
                   springlength: Optional[float] = None,
                   rgba: List[float] = [0.9, 0.7, 0.3, 1],
                   limited: bool = False,
                   range: Optional[List[float]] = None,
                   width: float = 0.003) -> 'ModelBuilder':
        """
        Add a tendon connecting multiple sites.
        
        Args:
            name: Name for the tendon
            sites: List of site names to connect (in order)
            stiffness: Spring stiffness (0 = constraint only)
            damping: Damping coefficient
            springlength: Rest length (None = auto from initial config)
            rgba: Color [r, g, b, a]
            limited: Whether to limit tendon length
            range: Length limits [min, max] if limited=True
            width: Visual width of tendon
        """
        if not self.spec:
            raise RuntimeError("Must load a base model first")
        
        if len(sites) < 2:
            raise ValueError("Tendon must connect at least 2 sites")
        
        # Create spatial tendon
        tendon = self.spec.add_tendon()
        tendon.name = name
        
        # Set properties
        if stiffness > 0:
            tendon.stiffness = stiffness
        if damping > 0:
            tendon.damping = damping
        if springlength is not None:
            tendon.springlength = springlength
        
        # Set limits
        if limited and range:
            tendon.limited = limited
            tendon.range = range
        
        # Visual properties
        tendon.rgba = rgba
        tendon.width = width
        
        # Add wrapping sites to make it spatial
        for site_name in sites:
            # Verify site exists
            site_spec = self._find_site(site_name)
            if not site_spec:
                raise ValueError(f"Site '{site_name}' not found")
            # wrap_site takes the site name as a string
            tendon.wrap_site(site_name)
        
        return self
    
    def add_body(self, name: str,
                 pos: List[float] = [0, 0, 0],
                 quat: Optional[List[float]] = None,
                 mass: float = 1.0,
                 geom_type: str = "box",
                 geom_size: List[float] = [0.05, 0.05, 0.05],
                 geom_rgba: List[float] = [0.5, 0.5, 0.8, 1],
                 geom_mesh: Optional[str] = None,
                 geom_material: Optional[str] = None,
                 free_joint: bool = True,
                 parent: Optional[str] = None,
                 intersection: bool = True,
                 joints: Optional[List[float]] = None,
                 friction: Optional[List[float]] = None) -> 'ModelBuilder':
        """
        Add a new body with a geom.
        
        Args:
            name: Name for the body
            pos: Initial position [x, y, z]
            quat: Initial orientation [w, x, y, z]
            mass: Body mass
            geom_type: "box", "sphere", "cylinder", "capsule", "mesh"
            geom_size: Geom dimensions (interpretation depends on type, ignored for mesh)
            geom_rgba: Color [r, g, b, a]
            geom_mesh: Mesh name (required when geom_type="mesh", must call add_mesh first)
            geom_material: Material name (optional, must call add_material first)
            free_joint: Add a free joint (6 DOF)
            parent: Parent body name (None = world)
            intersection: Whether geom has collision detection enabled
            joints: List of joint axes for sliding joints (e.g., [[1,0,0], [0,1,0]])
            friction: Friction coefficients [sliding, torsional, rolling] (None = MuJoCo default [1, 0.005, 0.0001])
                     - sliding: friction for sliding motion (most important)
                     - torsional: friction for spinning in place
                     - rolling: friction for rolling motion
        """
        if not self.spec:
            raise RuntimeError("Must load a base model first")
        
        # Find parent body or use worldbody
        if parent:
            parent_body = self._find_body(parent)
            if not parent_body:
                raise ValueError(f"Parent body '{parent}' not found")
        else:
            parent_body = self.spec.worldbody
        
        # Create body
        body = parent_body.add_body()
        body.name = name
        body.pos = pos
        if quat is not None:
            body.quat = quat  # Default orientation

        # Note: mass is set on the geom (below) so MuJoCo can compute inertia from geometry
        
        # Add free joint if requested
        if free_joint:
            joint = body.add_freejoint()
            joint.name = f"{name}_freejoint"  # Explicitly name the joint

        if joints:
            for axis in joints:
                joint = body.add_joint()
                joint.type = mujoco.mjtJoint.mjJNT_SLIDE
                joint.axis = axis
                joint.name = f"{name}_joint_{axis[0]}{axis[1]}{axis[2]}"

        # Add geom
        geom = body.add_geom()
        geom.name = f"{name}_geom"
        
        # Handle mesh geom type separately
        if geom_type.lower() == "mesh":
            if geom_mesh is None:
                raise ValueError("geom_mesh must be specified when geom_type='mesh'")
            geom.type = mujoco.mjtGeom.mjGEOM_MESH
            geom.meshname = geom_mesh
        else:
            geom.type = getattr(mujoco.mjtGeom, f"mjGEOM_{geom_type.upper()}")
            # Ensure geom_size has 3 elements (pad with zeros if needed)
            if len(geom_size) == 1:
                geom.size = [geom_size[0], geom_size[0], geom_size[0]]
            elif len(geom_size) == 2:
                geom.size = [geom_size[0], geom_size[1], 0]
            else:
                geom.size = geom_size
        
        # Set material if specified
        if geom_material is not None:
            geom.material = geom_material
        
        geom.rgba = geom_rgba
        
        # Set mass on geom so MuJoCo computes inertia from geometry
        # This is especially important for mesh bodies
        geom.mass = mass

        # Set friction if provided
        if friction is not None:
            if len(friction) == 1:
                # If only one value, use it for sliding friction
                geom.friction = [friction[0], 0.005, 0.0001]
            elif len(friction) == 2:
                # If two values, use for sliding and torsional
                geom.friction = [friction[0], friction[1], 0.0001]
            else:
                # Use all three values
                geom.friction = friction

        if not intersection:
            geom.contype = 0
            geom.conaffinity = 0

        return self
    
    def add_deformable_object(self, name: str,
                             pos: List[float] = [0, 0, 0],
                             shape: str = "box",
                             size: List[float] = [0.1, 0.1, 0.1],
                             spacing: float = 0.02,
                             stiffness: float = 1000.0,
                             damping: float = 10.0,
                             mass: float = 1.0,
                             rgba: List[float] = [0.8, 0.3, 0.3, 0.8],
                             particle_size: float = 0.01,
                             solref: List[float] = None,
                             solimp: List[float] = None) -> 'ModelBuilder':
        """
        Add a deformable object using MuJoCo's composite/flexcomp feature.
        
        Creates a soft body made of particles connected by springs/tendons that can
        deform when grasped or manipulated.
        
        Args:
            name: Name prefix for the composite object
            pos: Initial position [x, y, z]
            shape: Shape type - "box", "sphere", "ellipsoid", "cylinder"
            size: Dimensions [x, y, z] - interpretation depends on shape:
                  - box: [half_width, half_depth, half_height]
                  - sphere: [radius, 0, 0]
                  - ellipsoid: [radius_x, radius_y, radius_z]
                  - cylinder: [radius, half_height, 0]
            spacing: Distance between particles (smaller = more particles, more deformable)
            stiffness: Spring stiffness connecting particles (higher = stiffer)
            damping: Spring damping (higher = more energy dissipation)
            mass: Total mass distributed across all particles
            rgba: Color [r, g, b, alpha]
            particle_size: Visual size of each particle sphere
            solref: Contact solver reference [timeconst, dampratio] for soft contacts
            solimp: Contact solver impedance [dmin, dmax, width, mid, power]
        
        Returns:
            self for method chaining
            
        Example:
            # Soft sponge-like object
            builder.add_deformable_object("sponge", 
                                         pos=[0.5, 0, 0.2],
                                         shape="box",
                                         size=[0.05, 0.05, 0.05],
                                         spacing=0.015,
                                         stiffness=500.0,
                                         damping=5.0,
                                         rgba=[1, 1, 0, 0.6])
            
            # Deformable ball
            builder.add_deformable_object("ball",
                                         pos=[0.5, 0, 0.2],
                                         shape="sphere",
                                         size=[0.04, 0, 0],
                                         spacing=0.01,
                                         stiffness=2000.0)
        """
        if not self.spec:
            raise RuntimeError("Must load a base model first")
        
        # Set default contact parameters for soft objects if not provided
        if solref is None:
            solref = [0.01, 1.0]  # Soft, compliant contacts
        if solimp is None:
            solimp = [0.9, 0.95, 0.001, 0.5, 2]  # Penetration tolerance for soft body
        
        # Create composite/flexcomp object - add_composite is a method of MjSpec, not MjsBody
        composite = self.spec.add_composite()
        composite.prefix = name
        composite.type = mujoco.mjtComposite.mjCOMPOS_PARTICLE  # Particle-based soft body
        
        # Set geometry type
        if shape == "box":
            composite.add_box(
                pos=pos,
                size=size,
                count=[
                    max(2, int(2 * size[0] / spacing)),
                    max(2, int(2 * size[1] / spacing)),
                    max(2, int(2 * size[2] / spacing))
                ]
            )
        elif shape == "sphere":
            composite.add_ellipsoid(
                pos=pos,
                size=[size[0], size[0], size[0]],  # Uniform radius
                count=[
                    max(3, int(2 * size[0] / spacing)),
                    max(3, int(2 * size[0] / spacing)),
                    max(2, int(2 * size[0] / spacing))
                ]
            )
        elif shape == "ellipsoid":
            composite.add_ellipsoid(
                pos=pos,
                size=size,
                count=[
                    max(3, int(2 * size[0] / spacing)),
                    max(3, int(2 * size[1] / spacing)),
                    max(2, int(2 * size[2] / spacing))
                ]
            )
        elif shape == "cylinder":
            composite.add_cylinder(
                pos=pos,
                size=[size[0], size[1]],  # [radius, half_height]
                count=[
                    max(4, int(2 * np.pi * size[0] / spacing)),
                    max(2, int(2 * size[1] / spacing))
                ]
            )
        else:
            raise ValueError(f"Unsupported shape: {shape}. Use 'box', 'sphere', 'ellipsoid', or 'cylinder'")
        
        # Configure particle properties
        composite.spacing = spacing
        composite.solrefsmooth = solref
        composite.solimpsmooth = solimp
        
        # Add pin to fix the composite (can be removed if free-floating desired)
        # composite.add_pin(0, 0, 0)  # Uncomment to pin first particle
        
        # Set material properties for particles
        # Note: Individual particle mass will be total_mass / num_particles
        geom = composite.geom
        geom.size = [particle_size]
        geom.rgba = rgba
        geom.mass = mass / (composite.count[0] * composite.count[1] * composite.count[2] if hasattr(composite, 'count') else 100)
        geom.solref = solref
        geom.solimp = solimp
        
        # Add internal springs/tendons connecting particles
        if stiffness > 0 or damping > 0:
            # Skin creates visual mesh and internal constraints
            composite.add_skin()
            skin = composite.skin
            skin.rgba = rgba
            skin.inflate = particle_size * 0.5
            
            # Internal tendons for structural integrity
            # Note: Tendons are automatically created between nearby particles
            tendon = composite.tendon
            tendon.kind = mujoco.mjtTendon.mjTENDON_FIXED
            tendon.stiffness = stiffness
            tendon.damping = damping
        
        return self
    
    def remove_body(self, name: str) -> 'ModelBuilder':
        """
        Remove a body by name.
        
        Args:
            name: Name of the body to remove
        """
        if not self.spec:
            raise RuntimeError("Must load a base model first")
        
        body = self._find_body(name)
        if not body:
            raise ValueError(f"Body '{name}' not found")
        
        
        # Remove body from parent's bodies list
        self.spec.delete(body)
        
        return self

    def add_tendons_from_graph(self, 
                               graph,  # networkx.Graph or similar
                               site_prefix: str = "",
                               default_stiffness: float = 100.0,
                               default_damping: float = 1.0,
                               default_springlength: Optional[Union[float, List[float]]] = None,
                               default_rgba: List[float] = [0.9, 0.7, 0.3, 1],
                               tendon_name_format: str = "tendon_{i}_{j}") -> 'ModelBuilder':
        """
        Automatically create tendons from a NetworkX graph structure.
        Prints basic timing information to help diagnose slowness.
        
        Nodes in the graph represent sites, and edges represent tendons connecting them.
        Node and edge attributes can specify properties:
        
        Node attributes (optional):
            - All attributes are passed to add_site if the site doesn't exist
            - Examples: 'pos', 'size', 'rgba'
        
        Edge attributes (optional):
            - 'stiffness': Tendon stiffness (default: default_stiffness)
            - 'damping': Tendon damping (default: default_damping)
            - 'springlength': Spring rest length (default: default_springlength)
            - 'rgba': Color (default: default_rgba)
            - 'type': 'spatial' or 'fixed' (default: 'spatial')
            - 'name': Custom tendon name (default: use tendon_name_format)
        
        Args:
            graph: NetworkX graph where nodes are site names and edges define tendons
            site_prefix: Prefix to add to node names to get site names
            default_stiffness: Default stiffness for tendons without edge attribute
            default_damping: Default damping for tendons without edge attribute
            default_springlength: Default spring rest length
            default_rgba: Default color for tendons
            tendon_name_format: Format string for tendon names (receives i, j node names)
        
        Returns:
            self for method chaining
        
        Example:
            import networkx as nx
            G = nx.Graph()
            G.add_edge('site1', 'site2', stiffness=200, springlength=0.5)
            G.add_edge('site2', 'site3', stiffness=150, springlength=0.3)
            builder.add_tendons_from_graph(G, default_damping=2.0)
        """
        if not self.spec:
            raise RuntimeError("Must load a base model first")

        t_total_start = time.perf_counter()
        
        try:
            t_import_start = time.perf_counter()
            import networkx as nx
            t_import = time.perf_counter() - t_import_start
        except ImportError:
            raise ImportError("NetworkX is required for graph-based tendon generation. "
                            "Install with: pip install networkx")
        
        # Materialize edges to measure traversal separately
        t_edges_start = time.perf_counter()
        edges = list(graph.edges())
        t_edges_list = time.perf_counter() - t_edges_start

        edge_times = []
        spatial_count = 0
        fixed_count = 0
        add_tendon_time = 0.0

        # Iterate over edges to create tendons
        t_iter_start = time.perf_counter()
        for i, j in edges:
            edge_t0 = time.perf_counter()

            # Get edge attributes
            edge_data = graph.get_edge_data(i, j) or {}
            
            # Determine tendon properties from edge attributes or defaults
            tendon_type = edge_data.get('type', 'spatial')
            stiffness = edge_data.get('stiffness', default_stiffness)
            damping = edge_data.get('damping', default_damping)
            springlength = edge_data.get('springlength', default_springlength)
            rgba = edge_data.get('rgba', default_rgba)
            
            # Generate tendon name
            if 'name' in edge_data:
                tendon_name = edge_data['name']
            else:
                tendon_name = tendon_name_format.format(i=i, j=j)
            
            # Get site names
            site_i = f"{site_prefix}{i}"
            site_j = f"{site_prefix}{j}"
            
            # Create tendon connecting the two sites
            add_tendon_t0 = time.perf_counter()
            if tendon_type == 'spatial':
                self.add_tendon(
                    name=tendon_name,
                    sites=[site_i, site_j],
                    tendon_type='spatial',
                    stiffness=stiffness,
                    damping=damping,
                    springlength=springlength,
                    rgba=rgba
                )
                spatial_count += 1
            elif tendon_type == 'fixed':
                # For fixed tendons, nodes should represent joints
                joint_i = f"{site_prefix}{i}"
                joint_j = f"{site_prefix}{j}"
                self.add_tendon(
                    name=tendon_name,
                    sites=[joint_i, joint_j],  # keep same call path; joints handled upstream if supported
                    tendon_type='fixed',
                    stiffness=stiffness,
                    damping=damping,
                    springlength=springlength,
                    rgba=rgba
                )
                fixed_count += 1
            add_tendon_time += time.perf_counter() - add_tendon_t0

            edge_dt = time.perf_counter() - edge_t0
            edge_times.append((i, j, tendon_name, edge_dt))
        t_iter = time.perf_counter() - t_iter_start

        t_total = time.perf_counter() - t_total_start

        # Summary timing
        n_edges = len(edges)
        mean_edge = (sum(dt for _, _, _, dt in edge_times) / n_edges) if n_edges else 0.0
        max_edge_entry = max(edge_times, key=lambda x: x[3]) if edge_times else None

        # print("[Timing] ModelBuilder.add_tendons_from_graph:")
        # print(f"  total={t_total:.3f}s | import_nx={t_import:.3f}s | list_edges={t_edges_list:.3f}s | iterate={t_iter:.3f}s | add_tendon={add_tendon_time:.3f}s")
        # print(f"  edges={n_edges} | spatial={spatial_count} | fixed={fixed_count} | mean/edge={mean_edge:.6f}s")
        # if max_edge_entry:
        #     i, j, name, dt = max_edge_entry
        #     print(f"  slowest_edge: ({i}, {j}) name={name} time={dt:.6f}s")

        # Top 5 slowest edges
        # topk = 5
        # if n_edges > 0:
        #     slowest = sorted(edge_times, key=lambda x: x[3], reverse=True)[:min(topk, n_edges)]
        #     print("  top_slowest_edges:")
        #     for i, j, name, dt in slowest:
        #         print(f"    ({i}, {j}) name={name} time={dt:.6f}s")

        return self
    
    def compile(self) -> mujoco.MjModel:
        """
        Compile the spec into a model.
        
        Note: After compilation, the spec becomes invalid and cannot be reused.
        Create a new ModelBuilder for each model you want to build.
        """
        if not self.spec:
            raise RuntimeError("Must load a base model first")
        # print(self.spec)
        model = self.spec.compile()
        
        # Clear the spec after compilation to prevent reuse
        del self.spec
        
        return model
    


# ============================================================================
# Main Simulation Class
# ============================================================================

class MuJoCoSimulation:
    """
    Object-oriented wrapper for MuJoCo simulations.
    
    Example usage:
        sim = MuJoCoSimulation("model.xml")
        crane = sim.add_joint("crane", "crane_joint")
        box = sim.add_body("box", "red_box")
        
        def control(sim):
            box_pos = box.pos
            crane.qpos = compute_target(box_pos)
        
        sim.run(control_callback=control, passive=True)
    """
    
    def __init__(self, xml_path: Optional[str] = None, xml_string: Optional[str] = None):
        """
        Initialize simulation from XML file or string.
        
        Args:
            xml_path: Path to XML file
            xml_string: XML content as string
        """
        if xml_path:
            # Use from_xml_path to properly handle relative paths (e.g., mesh files)
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        elif xml_string:
            self.model = mujoco.MjModel.from_xml_string(xml_string)
        else:
            raise ValueError("Must provide either xml_path or xml_string")
        
        self.data = mujoco.MjData(self.model)
        
        # Entity registries
        self.joints: Dict[str, Joint] = {}
        self.bodies: Dict[str, Body] = {}
        self.actuators: Dict[str, Actuator] = {}
        self.sites: Dict[str, Site] = {}
        self.tendons: Dict[str, Tendon] = {}
        self.contact_sensors: Dict[str, ContactSensor] = {}
        
        # Control callback
        self._control_callback: Optional[Callable] = None
        
        # Viewer settings
        self.viewer_distance = 2.5
        self.viewer_lookat = np.array([0.0, 0.0, 0.25])
    
    # ------------------------------------------------------------------------
    # Entity Registration
    # ------------------------------------------------------------------------
    
    def reg_joint(self, alias: str, name: str) -> Joint:
        """Register a joint for easy access."""
        joint = Joint(name, self.model, self.data)
        self.joints[alias] = joint
        return joint
    
    def reg_body(self, alias: str, name: str) -> Body:
        """Register a body for easy access."""
        body = Body(name, self.model, self.data)
        self.bodies[alias] = body
        return body

    def reg_actuator(self, alias: str, name: str) -> Actuator:
        """Register an actuator for easy access."""
        actuator = Actuator(name, self.model, self.data)
        self.actuators[alias] = actuator
        return actuator
    
    def reg_site(self, alias: str, name: str) -> Site:
        """Register a site for easy access."""
        site = Site(name, self.model, self.data)
        self.sites[alias] = site
        return site
    
    def reg_tendon(self, alias: str, name: str) -> Tendon:
        """Register a tendon for easy access."""
        tendon = Tendon(name, self.model, self.data)
        self.tendons[alias] = tendon
        return tendon
    
    def reg_contact_sensor(self, alias: str, body_name: str, 
                          monitored_bodies: List[str]) -> ContactSensor:
        """
        Register a contact sensor for a body.
        
        Args:
            alias: Name to access the sensor
            body_name: Name of the body to monitor contacts for
            monitored_bodies: List of body names to detect contacts with
            
        Returns:
            ContactSensor instance
            
        Example:
            # Monitor contacts between target and chopsticks
            sensor = sim.reg_contact_sensor("target_sensor", 
                                           "target_block",
                                           ["left_chopstick", "right_chopstick"])
            
            # Check contacts
            contacts = sensor.get_contacts()
            for contact in contacts:
                print(f"{contact.body_name}: {contact.total_force:.2f} N")
        """
        sensor = ContactSensor(body_name, monitored_bodies, self.model, self.data)
        self.contact_sensors[alias] = sensor
        return sensor
    
    # ------------------------------------------------------------------------
    # Alternative Constructor with ModelBuilder
    # ------------------------------------------------------------------------
    
    @classmethod
    def from_builder(cls, builder: ModelBuilder) -> 'MuJoCoSimulation':
        """
        Create simulation from a ModelBuilder.
        
        Args:
            builder: ModelBuilder instance with modifications
            
        Returns:
            MuJoCoSimulation instance
            
        Example:
            builder = ModelBuilder()
            builder.from_xml_path("base.xml")
            builder.add_site("gripper_site", "gripper", pos=[0, 0, 0.05])
            builder.add_tendon("my_tendon", sites=["gripper_site", "target_site"])
            
            sim = MuJoCoSimulation.from_builder(builder)
        """
        model = builder.compile()
        sim = cls.__new__(cls)
        sim.model = model
        sim.data = mujoco.MjData(model)
        sim.joints = {}
        sim.bodies = {}
        sim.actuators = {}
        sim.sites = {}
        sim.tendons = {}
        sim.contact_sensors = {}
        sim._control_callback = None
        sim.viewer_distance = 2.5
        sim.viewer_lookat = np.array([0.0, 0.0, 0.25])
        return sim
    
    # ------------------------------------------------------------------------
    # Convenience Properties
    # ------------------------------------------------------------------------
    
    @property
    def time(self) -> float:
        """Current simulation time."""
        return self.data.time
    
    def reset(self):
        """Reset simulation to initial state."""
        mujoco.mj_resetData(self.model, self.data)
    
    def step(self):
        """Advance simulation by one timestep."""
        mujoco.mj_step(self.model, self.data)
    
    def forward(self):
        """Compute forward kinematics."""
        mujoco.mj_forward(self.model, self.data)
    
    # ------------------------------------------------------------------------
    # Control Callback
    # ------------------------------------------------------------------------
    
    def set_control_callback(self, callback: Callable[['MuJoCoSimulation'], None], gravity_comp: bool = False):
        """
        Set control callback function.
        
        Args:
            callback: Function that takes the simulation object as argument
            gravity_comp: If True, automatically adds gravity compensation to controls
        """
        self._control_callback = callback
        self._gravity_comp = gravity_comp

        # if gravity_comp:
        #     print("Gravity compensation enabled in control callback.")
        
        def mujoco_control(model, data):


            # Apply gravity compensation FIRST if enabled
            if gravity_comp:
                # Compute gravity and Coriolis forces using inverse dynamics
                # Set acceleration to zero to get just the bias forces
                acc_zero = np.zeros(model.nv)
                qacc_old = data.qacc.copy()
                data.qacc[:] = acc_zero
                
                # Compute generalized forces needed for zero acceleration
                mujoco.mj_rne(model, data, 0, data.qfrc_bias)
                
                # Restore original acceleration
                data.qacc[:] = qacc_old
                
                # Map generalized forces to actuator space
                for i in range(model.nu):
                    # Get the joint associated with this actuator
                    jnt_id = model.actuator_trnid[i, 0]
                    
                    if jnt_id >= 0 and jnt_id < model.njnt:
                        # Get the DOF address for this joint
                        dof_adr = model.jnt_dofadr[jnt_id]
                        
                        # Apply gravity compensation force for this DOF
                        # Scale by gear ratio if needed
                        gear = model.actuator_gear[i, 0]
                        if gear != 0:
                            data.ctrl[i] = data.qfrc_bias[dof_adr] / gear
                        else:
                            data.ctrl[i] = data.qfrc_bias[dof_adr]
            
            # Now run user callback (which can add to or override ctrl)
            callback(self)
        
        mujoco.set_mjcb_control(mujoco_control)
    
    # ------------------------------------------------------------------------
    # Simulation Execution
    # ------------------------------------------------------------------------
    
    def run(self, 
            control_callback: Optional[Callable[['MuJoCoSimulation'], None]] = None,
            passive: bool = True,
            duration: Optional[float] = None,
            realtime_speed: float = 1.0,
            viewer_distance: Optional[float] = None,
            viewer_lookat: Optional[np.ndarray] = None,
            gravity_comp: bool = False,
            control_preset: bool = False,
            record_video: bool = False):
        """
        Run the simulation.
        
        Args:
            control_callback: Optional control function called each timestep
            passive: Use passive viewer (True) or active viewer (False)
            duration: Simulation duration in seconds (None = run until closed)
            realtime_speed: Simulation speed multiplier
            viewer_distance: Camera distance override
            viewer_lookat: Camera target override
        """
        if not control_preset:
            if control_callback:
                self.set_control_callback(control_callback, gravity_comp=gravity_comp)
            else:
                self.set_control_callback(lambda sim: None, gravity_comp=gravity_comp)  # No-op
        
        if viewer_distance is not None:
            self.viewer_distance = viewer_distance
        if viewer_lookat is not None:
            self.viewer_lookat = viewer_lookat
        
        if record_video:
            video_path = "videos/simulation_recording.mp4"
            video_duration = duration if duration is not None else 5.0  # Default to 10s if no duration
            self._record_video(video_path, video_duration, fps=30)
            print(f"Video recorded to: {video_path}")
            return
        if passive:
            self._run_passive(duration, realtime_speed)
        else:
            self._run_active()
        

    
    def _run_passive(self, duration: Optional[float], realtime_speed: float):
        """Run with passive viewer."""
        with viewer.launch_passive(self.model, self.data) as v:
            v.cam.distance = self.viewer_distance
            v.cam.lookat[:] = self.viewer_lookat
            
            t0 = self.data.time
            step_duration = self.model.opt.timestep / realtime_speed
            
            while v.is_running():
                if duration and (self.data.time - t0) >= duration:
                    break
                
                self.step()
                v.sync()
                time.sleep(step_duration)
    
    def _run_active(self):
        """Run with active viewer (built-in control loop)."""
        viewer.launch(self.model, self.data)

    def _record_video(self, video_path: str, duration: float, fps: int = 15):
        """
        Record a video of the simulation.
        
        Args:
            video_path: Output video file path
            duration: Duration of the video in seconds
            fps: Frames per second
        """
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.lookat = [0, 0, 0.25]
        cam.distance = 2
        cam.azimuth = -180  # Rotate camera to face robot head-on (0=side, 90=front, 180=side, 270=back)
        cam.elevation = -15  # Slight downward angle for better view
        frames = []
        steps_per_frame = int(1 / (fps * self.model.opt.timestep))
        with mujoco.Renderer(self.model, width=1920 // 3, height=1080 // 3) as renderer:
            while self.data.time < duration:
                for i in range(steps_per_frame):
                    self.step()

                renderer.update_scene(self.data, cam)
                pixels = renderer.render()
                frames.append(pixels)
        import imageio
        imageio.mimwrite(video_path, frames, fps=fps)
        return


    def close(self):
        """Close the simulation and free all resources."""
        # Clear all registered entities
        self.joints.clear()
        self.bodies.clear()
        self.actuators.clear()
        self.sites.clear()
        self.tendons.clear()
        self.contact_sensors.clear()
        
        # Clear callback
        self._control_callback = None

        # Ensure the global MuJoCo control callback doesn't hold a stale reference
        # to this instance after its MjModel/MjData have been deleted.
        mujoco.set_mjcb_control(None)
        
        # Delete data and model to free memory
        # Python's garbage collector will clean up when references are gone

        del self.data
        del self.model

# ============================================================================
# Example Usage Template
# ============================================================================

def example_usage():
    """Example showing how to use the simulation template."""
    
    # 1. Create simulation
    sim = MuJoCoSimulation(xml_path="testing/model.xml")
    
    # 2. Register entities you care about
    crane_joint = sim.add_joint("crane", "crane_base__up_down")
    lift_actuator = sim.add_actuator("lift", "lift")
    
    dummy_joint = sim.add_joint("dummy", "dummy__dummy_joint")
    target_joint = sim.add_joint("target", "blue_box__blue_joint")
    
    red_box = sim.add_body("red_box", "red_box")
    green_box = sim.add_body("green_box", "green_box")
    
    # 3. Define control parameters
    target_height = 1.5
    t_rise = 3.0
    Kp_orient = 20.0
    Kd_orient = 50.0
    
    # 4. Define control callback
    def control_callback(sim: MuJoCoSimulation):
        # Make dummy follow target
        dummy_joint.qpos = target_joint.qpos
        
        # Compute crane trajectory
        t = sim.time
        if 0.5 < t < 0.5 + t_rise:
            alpha = 0.5 - 0.5*np.cos(np.pi * (t - 0.5) / t_rise)
            target = crane_joint.qpos[0]*(1 - alpha) + target_height*alpha
        elif t <= 0.5:
            target = 0.0
        else:
            target = target_height
        
        lift_actuator.ctrl = target
        
        # Orientation synchronization control
        qA = red_box.quat
        qB = green_box.quat
        
        q_err = quat_mul(quat_conj(qA), qB)
        ang_err = quat_to_axis_angle(q_err)
        
        wA = red_box.angular_vel
        wB = green_box.angular_vel
        w_err = wB + wA / 2
        
        tau = -Kp_orient*ang_err - Kd_orient*w_err
        
        red_box.add_torque(-tau)
        green_box.add_torque(tau)
    
    # 5. Run simulation
    sim.run(control_callback=control_callback, passive=True)


# ============================================================================
# Command-line Interface
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="MuJoCo Simulation Template")
    parser.add_argument("--xml", type=str, help="Path to XML file")
    parser.add_argument("--passive", action="store_true", 
                        help="Use passive viewer (default: active)")
    parser.add_argument("--duration", type=float, help="Simulation duration (seconds)")
    parser.add_argument("--speed", type=float, default=1.0, 
                        help="Realtime speed multiplier")
    return parser.parse_args()


if __name__ == "__main__":
    # Run example
    example_usage()
