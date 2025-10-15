import mujoco, mujoco.viewer
import numpy as np

xml = """
<mujoco>

<asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
     rgb2=".2 .3 .4" width="300" height="300" mark="none"/>
    <material name="grid" texture="grid" texrepeat="1 1"
     texuniform="true" reflectance=".2"/>
  </asset>


  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" pos="0 0 -.3" size="2 2 .1" material="grid"/>
    
    <body name="red_box" pos="0 0 0.3">
        <freejoint/>
        <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
        <site name="red_site" pos="0 0 0" size=".1"/>
    </body>
    <body name="green_box" pos="0.25 0 0.3">
        <freejoint/>
        <geom name="green_box" type="box" size=".2 .2 .2" pos=".5 0 0" rgba="0 1 0 1"/>
        <site name="green_site" pos="0 0 0" size=".1"/>
    </body>

    

    
  </worldbody>
</mujoco>
"""


model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)


mujoco.viewer.launch(model)
print("controller number:", model.nu)