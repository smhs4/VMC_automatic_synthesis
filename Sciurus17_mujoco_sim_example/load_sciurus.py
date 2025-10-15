import mujoco, mujoco.viewer
m = mujoco.MjModel.from_xml_path(r"URDFs/sciurus17_description/urdf/sciurus17.xml")  # or .\sciurus17.xml
d = mujoco.MjData(m)
with mujoco.viewer.launch_passive(m, d) as v:
    while v.is_running():
        mujoco.mj_step(m, d); v.sync()
