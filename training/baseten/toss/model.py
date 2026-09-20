"""Generic parallel-axis two-link mechanism; full 3-D rigid pancake contacts."""
import math
import xml.etree.ElementTree as ET


def build_xml(case,cfg):
    p=cfg["physics"]; c=cfg["control"]
    L=case["elbow_wrist_mm"]/1000;r=case["wrist_pan_center_mm"]/1000
    R=case["pan_diameter_mm"]/2000;wall=p["pan_wall_mm"]/1000
    bottom_R=case["pan_bottom_diameter_mm"]/2000
    floor=p["pan_floor_mm"]/1000; depth=p["pan_depth_mm"]/1000
    root=ET.Element("mujoco",model="two_joint_domain_randomized_v4")
    ET.SubElement(root,"compiler",angle="radian",autolimits="true")
    option=ET.SubElement(root,"option",timestep=str(p["timestep_s"]),gravity="0 0 -9.81",
                  integrator="implicitfast",iterations="50")
    # Required by variable-geometry Warp batching: dof_length is shared and
    # only relevant to sleeping in the pinned simulator. Keep all bodies awake.
    ET.SubElement(option,"flag",sleep="disable")
    default=ET.SubElement(root,"default")
    ET.SubElement(default,"geom",friction=f'{case["friction"]} .005 .0001',condim="3",solref=".008 1")
    world=ET.SubElement(root,"worldbody")
    ET.SubElement(world,"geom",name="ground",type="plane",size="2 2 .01",rgba=".2 .2 .2 1")
    ET.SubElement(world,"light",pos="0 -.5 1.5")
    ET.SubElement(world,"camera",name="side",pos=".5 -1 .65",xyaxes="1 .5 0 -.15 .3 1")
    arm=ET.SubElement(world,"body",name="arm",pos=f'0 0 {p["shoulder_height_mm"]/1000}')
    ET.SubElement(arm,"joint",name="shoulder",type="hinge",axis="0 -1 0",
                  range=" ".join(str(math.radians(v)) for v in c["shoulder_limits_deg"]),damping=".015",armature=".00015")
    ET.SubElement(arm,"geom",name="link",type="capsule",fromto=f'0 0 0 {L} 0 0',
                  size=".01",mass=str(case["link_mass_kg"]),contype="0",conaffinity="0",rgba=".1 .6 .8 1")
    # Motor body mass is located at the wrist. Its collision envelope is omitted:
    # actual lateral packaging and motor-to-motor clearance are not finalized.
    ET.SubElement(arm,"geom",name="wrist_motor_mass",type="box",pos=f'{L} -.05 0',
                  size=".0282 .028 .0282",mass=str(case["wrist_motor_mass_kg"]),
                  contype="0",conaffinity="0",rgba=".3 .3 .3 .5")
    pan=ET.SubElement(arm,"body",name="pan",pos=f'{L} 0 0')
    ET.SubElement(pan,"joint",name="wrist",type="hinge",axis="0 -1 0",
                  range=" ".join(str(math.radians(v)) for v in c["wrist_limits_deg"]),damping=".005",armature=".00003")
    # Separate pan at bowl centre and attachment at wrist; parallel-axis theorem.
    mp=case["pan_mass_kg"];ma=case["mount_mass_kg"];mass=mp+ma;com=mp*r/mass
    ix=mp*(3*R*R+depth*depth)/12+ma*(.03**2+.02**2)/12
    iy=ix+mp*(r-com)**2+ma*com**2
    iz=mp*R*R/2+ma*(.03**2+.03**2)/12+mp*(r-com)**2+ma*com**2
    scale=case["inertia_multiplier"]
    ET.SubElement(pan,"inertial",pos=f'{com} 0 {-floor/2}',mass=str(mass),
                  diaginertia=f'{ix*scale} {iy*scale} {iz*scale}')
    ET.SubElement(pan,"geom",name="pan_floor",type="cylinder",pos=f'{r} 0 {-floor/2}',
                  size=f'{bottom_R+wall} {floor/2}',rgba=".15 .16 .17 1")
    # Tapered rim: flat inner floor radius -> top inner rim radius.
    radial_rise=R-wall-bottom_R
    tilt=math.atan2(radial_rise,depth)
    rim_centre=bottom_R+wall/2+radial_rise/2
    half_height=math.hypot(depth,radial_rise)/2+floor
    for i in range(p["rim_segments"]):
        a=2*math.pi*i/p["rim_segments"]
        quat=(math.cos(a/2)*math.cos(tilt/2),-math.sin(a/2)*math.sin(tilt/2),
              math.cos(a/2)*math.sin(tilt/2),math.sin(a/2)*math.cos(tilt/2))
        ET.SubElement(pan,"geom",name=f'rim_{i}',type="box",
                      pos=f'{r+rim_centre*math.cos(a)} {rim_centre*math.sin(a)} {depth/2}',
                      size=f'{wall/2} {R*math.tan(math.pi/p["rim_segments"])*1.03} {half_height}',
                      quat=" ".join(map(str,quat)),rgba=".25 .26 .27 1")
    cake=ET.SubElement(world,"body",name="pancake")
    ET.SubElement(cake,"freejoint",name="pancake_free")
    cr=case["pancake_diameter_mm"]/2000;ct=case["pancake_thickness_mm"]/2000
    ET.SubElement(cake,"geom",name="pancake",type="cylinder",size=f'{cr} {ct}',
                  mass=str(case["pancake_mass_g"]/1000),rgba=".85 .65 .2 1")
    ET.SubElement(cake,"geom",name="top_marker",type="box",pos=f'0 0 {ct+.0001}',
                  size=f'{cr*.5} {cr*.5} .0001',mass="0",contype="0",conaffinity="0",rgba=".8 .1 .1 1")
    act=ET.SubElement(root,"actuator")
    for name in ("shoulder","wrist"):
        torque=case[name+"_torque_nm"]*case["torque_multiplier"]
        ET.SubElement(act,"motor",name=name,joint=name,gear="1",ctrllimited="true",ctrlrange=f'{-torque} {torque}')
    return ET.tostring(root,encoding="unicode")
