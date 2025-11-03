import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.mujoco_sim_template import MuJoCoSimulation, ModelBuilder, quat_from_euler
import numpy as np
import networkx
import mujoco
import pickle
from datetime import datetime
from typing import List, Tuple, Dict
import math
import random
import argparse
import networkx as nx
import gc  # For garbage collection

G = networkx.Graph()
xml_path = "Sciurus17_mujoco_sim_example/URDFs/sciurus17_description/urdf/sciurus17.xml"
builder = ModelBuilder(G)
builder.from_xml_path(xml_path)
builder.remove_body("r_link7")
builder.add_body(
    name="right_chopstick",
    parent="r_link6",
    free_joint=False,
    pos=[0, 0, 0.1],
    quat=quat_from_euler(0, 0, 0),
    geom_type="cylinder",
    geom_size=[0.01, 0.2],
    geom_rgba=[0, 0, 1, 1],
    mass=0.05
)
sim = MuJoCoSimulation.from_builder(builder)
sim.step()
print("Simulation step completed.")

G = networkx.Graph()
xml_path = "Sciurus17_mujoco_sim_example/URDFs/sciurus17_description/urdf/sciurus17.xml"
builder = ModelBuilder(G)
builder.from_xml_path(xml_path)
sim = MuJoCoSimulation.from_builder(builder)
sim.step()
print("Simulation step completed.")

