import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ACTION_NAMES = ["prepare_shoulder_deg", "shoulder_excursion_deg", "release_pitch_deg",
                "launch_s", "return_s", "wrist_timing", "release_hold_s"]
ACTION_LOW = np.array([45., 10., -15., .10, .10, -.3, 0.])
ACTION_HIGH = np.array([60., 40., 60., .80, 1.0, .3, .15])
# Static physical context is supplied at inference, not discovered magically.
FEATURE_SCALES = {
    "pancake_mass_g": 40., "pancake_diameter_mm": 120.,
    "elbow_wrist_mm": 400., "wrist_pan_center_mm": 200.,
    "pan_mass_kg": .45, "mount_mass_kg": .14, "link_mass_kg": .25,
    "wrist_motor_mass_kg": .7, "pan_diameter_mm": 240.,
    "pan_bottom_diameter_mm": 240.,
    "pancake_thickness_mm": 6.,
    "shoulder_torque_nm": 3., "wrist_torque_nm": 1.2356,
    "shoulder_speed_deg_s": 360., "wrist_speed_deg_s": 360.,
    "shoulder_accel_deg_s2": 3000., "wrist_accel_deg_s2": 3000.,
    "shoulder_zero_torque_speed_deg_s": 1200., "wrist_zero_torque_speed_deg_s": 1200.,
    "initial_x_mm": 60., "initial_y_mm": 60.,
    "home_shoulder_deg": 180., "home_wrist_deg": 180.,
}


def load_settings(path="settings.yaml"):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg.get("schema_version") != 1 or cfg.get("physics_revision") != 4:
        raise ValueError("Unsupported configuration/physics revision")
    for group in [cfg["domain"], *cfg["motor_profiles"].values()]:
        for name, bounds in group.items():
            if isinstance(bounds, list):
                if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
                    raise ValueError(f"Invalid range: {name}")
                if bounds[0] < 0:
                    raise ValueError(f"Negative physical range: {name}")
    for name,bounds in cfg['domain'].items():
        if isinstance(bounds,list) and name!='pancake_diameter_mm' and bounds[0]!=bounds[1]:
            raise ValueError(f'Only pancake diameter/location may vary; {name} must be fixed')
    if cfg['domain']['vision_noise_mm']!=0:
        raise ValueError('Vision noise must remain zero for size/location-only randomization')
    for profile in cfg['motor_profiles'].values():
        if any(bounds[0]!=bounds[1] for bounds in profile.values()):
            raise ValueError('Motor parameters must be fixed')
    p=cfg["physics"]
    if not 0 < p["timestep_s"] <= .005:
        raise ValueError("Use a positive physics step no larger than 5 ms")
    if p["rim_segments"] < 24:
        raise ValueError("At least 24 rim segments required")
    usable=min(p["pan_bottom_diameter_mm"],p["pan_diameter_mm"]-2*p["pan_wall_mm"])
    if p["pan_bottom_diameter_mm"]<=0 or p["pan_bottom_diameter_mm"]>p["pan_diameter_mm"]-2*p["pan_wall_mm"]:
        raise ValueError("Flat bottom must fit inside the rim's inner diameter")
    if max(cfg["domain"]["pancake_diameter_mm"]) >= usable-2*p["edge_margin_mm"]:
        raise ValueError("Pancakes do not fit inside the pan with the configured margin")
    t=cfg["training"]
    for name in ("workers","n_steps","batch_size","n_epochs","episodes","validation_every_episodes",
                 "validation_cases","checkpoint_every_episodes","checkpoint_every_seconds"):
        if not isinstance(t[name],int) or t[name]<=0:raise ValueError(f"training.{name} must be a positive integer")
    for name in ("max_training_seconds","stop_if_no_airborne_after"):
        if t[name] is not None and t[name]<=0:raise ValueError(f"training.{name} must be positive or null")
    if t.get("backend","cpu") not in ("cpu","warp"):
        raise ValueError("training.backend must be cpu or warp")
    if t["workers"]*t["n_steps"]%t["batch_size"]:
        raise ValueError("workers × n_steps must be divisible by batch_size")
    return cfg


def contract(cfg):
    return {"physics_revision":cfg["physics_revision"],
            "simulator_contract":"awesome-fixed-240-100-v1",
            "features":FEATURE_SCALES,
            "actions":ACTION_NAMES, "action_low":ACTION_LOW.tolist(),
            "action_high":ACTION_HIGH.tolist(), "physics":cfg["physics"],
            "control":cfg["control"], "domain":cfg["domain"],
            "motor_profiles":cfg["motor_profiles"]}


def optimistic_holding_screen(cfg,profile):
    """Analytic lower bound for the configured positive-cosine home pose.

    No physics stepping. If even the lightest/shortest arm fails with the
    strongest available motor, rejection sampling cannot produce a valid case.
    """
    case=sample_case(cfg,np.random.default_rng(0),profile,noisy=False)
    q1,q2=np.radians(cfg["control"]["home_deg"])
    if np.cos(q1)<0 or np.cos(q1+q2)<0:
        return {"conclusive":False,"feasible":None}
    for name in ("pancake_mass_g","pan_mass_kg","mount_mass_kg","elbow_wrist_mm","wrist_pan_center_mm"):
        case[name]=cfg["domain"][name][0]
    case["link_mass_kg"]=cfg["physics"]["link_fixed_mass_kg"]+cfg["physics"]["link_linear_density_kg_m"]*case["elbow_wrist_mm"]/1000
    demand=abs(gravity_torque(case,*cfg["control"]["home_deg"]))
    available=np.array([cfg["motor_profiles"][profile][joint+"_torque_nm"][1]
                        for joint in ("shoulder","wrist")])*cfg["domain"]["torque_multiplier"][1]
    allowed=available*cfg["physics"]["initial_holding_torque_fraction"]
    return {"conclusive":True,"feasible":bool(np.all(demand<=allowed)),
            "minimum_load_nm":demand.tolist(),"maximum_allowed_load_nm":allowed.tolist(),
            "note":"Necessary static condition only; passing does not prove dynamic feasibility."}


def contract_hash(cfg):
    return hashlib.sha256(json.dumps(contract(cfg),sort_keys=True).encode()).hexdigest()[:16]


def sample_case(cfg, rng, profile="hardware", overrides=None, noisy=True):
    case={}
    for group in (cfg["domain"], cfg["motor_profiles"][profile]):
        for name, bounds in group.items():
            if isinstance(bounds,list):
                case[name]=float(rng.uniform(*bounds))
    p=cfg["physics"]
    case.update(wrist_motor_mass_kg=p["wrist_motor_mass_kg"],pan_diameter_mm=p["pan_diameter_mm"],
                pan_bottom_diameter_mm=p["pan_bottom_diameter_mm"],
                home_shoulder_deg=cfg["control"]["home_deg"][0],home_wrist_deg=cfg["control"]["home_deg"][1])
    xy=rng.uniform(-cfg["domain"]["initial_offset_mm"],cfg["domain"]["initial_offset_mm"],2)
    case.update(initial_x_mm=float(xy[0]),initial_y_mm=float(xy[1]))
    if overrides:
        unknown=set(overrides)-set(case)
        if unknown:
            raise ValueError(f"Unknown case fields: {sorted(unknown)}")
        case.update({k:float(v) for k,v in overrides.items()})
    case["link_mass_kg"]=p["link_fixed_mass_kg"]+p["link_linear_density_kg_m"]*case["elbow_wrist_mm"]/1000
    if not np.isfinite(list(case.values())).all():
        raise ValueError("Nonfinite case input")
    for k in ("pancake_mass_g","pancake_diameter_mm","elbow_wrist_mm","wrist_pan_center_mm",
              "pan_mass_kg","mount_mass_kg","pancake_thickness_mm"):
        if case[k] <= 0:
            raise ValueError(f"{k} must be positive")
    case["observed_initial_x_mm"]=case["initial_x_mm"]
    case["observed_initial_y_mm"]=case["initial_y_mm"]
    if noisy:
        noise=rng.normal(0,cfg["domain"]["vision_noise_mm"],2)
        case["observed_initial_x_mm"]+=float(noise[0])
        case["observed_initial_y_mm"]+=float(noise[1])
    return case


def observe(case):
    values=case.copy()
    values["initial_x_mm"]=case.get("observed_initial_x_mm",case["initial_x_mm"])
    values["initial_y_mm"]=case.get("observed_initial_y_mm",case["initial_y_mm"])
    obs=np.array([values[k]/scale for k,scale in FEATURE_SCALES.items()],np.float32)
    if not np.isfinite(obs).all() or np.any(abs(obs)>2):
        raise ValueError("Observation outside supported range")
    return obs


def geometry_issues(case,cfg):
    p=cfg["physics"]
    available=min(case["pan_bottom_diameter_mm"]/2,
                  case["pan_diameter_mm"]/2-p["pan_wall_mm"])-p["edge_margin_mm"]
    offset=np.hypot(case["initial_x_mm"],case["initial_y_mm"])
    return ["pancake_does_not_fit"] if offset+case["pancake_diameter_mm"]/2 > available else []


def gravity_torque(case, shoulder_deg=45., wrist_deg=-45.):
    L=case["elbow_wrist_mm"]/1000;r=case["wrist_pan_center_mm"]/1000
    q=np.radians([shoulder_deg,shoulder_deg+wrist_deg])
    pan=case["pan_mass_kg"]+case["pancake_mass_g"]/1000
    wrist=9.81*pan*r*np.cos(q[1])
    shoulder=9.81*((case["wrist_motor_mass_kg"]+case["mount_mass_kg"]+pan)*L*np.cos(q[0])
                     +case["link_mass_kg"]*L/2*np.cos(q[0]))+wrist
    return np.array([shoulder,wrist])


def feasibility_issues(case,cfg):
    issues=geometry_issues(case,cfg)
    load=abs(gravity_torque(case,*cfg["control"]["home_deg"]))
    cap=np.array([case["shoulder_torque_nm"],case["wrist_torque_nm"]])*case["torque_multiplier"]
    if np.any(load > cap*cfg["physics"]["initial_holding_torque_fraction"]):
        issues.append("insufficient_home_holding_torque")
    return issues


def condition_motor_capacity(case,cfg,profile,rng):
    """Sample inside existing motor bounds, conditional on holding this load.

    This changes the training sampling distribution, never the hardware bounds.
    Explicit evaluation cases keep the original independent motor sampling.
    """
    load=abs(gravity_torque(case,*cfg["control"]["home_deg"]))
    fraction=cfg["physics"]["initial_holding_torque_fraction"]
    limits=cfg["motor_profiles"][profile]
    max_torque=np.array([limits[j+"_torque_nm"][1] for j in ("shoulder","wrist")])
    low,high=cfg["domain"]["torque_multiplier"]
    low=max(low,float(np.max(load/(fraction*max_torque))))
    if low>high:return
    multiplier=float(rng.uniform(low,high))
    torques={}
    for j,name in enumerate(("shoulder","wrist")):
        lo,hi=limits[name+"_torque_nm"]
        lo=max(lo,float(load[j]/(fraction*multiplier)))
        if lo>hi:return
        torques[name+"_torque_nm"]=float(rng.uniform(lo,hi))
    case.update(torques,torque_multiplier=multiplier)


def decode_action(action):
    a=np.asarray(action,dtype=float)
    if a.shape != ACTION_LOW.shape or not np.isfinite(a).all():
        raise ValueError("Expected seven finite trajectory actions")
    values=ACTION_LOW+(np.clip(a,-1,1)+1)/2*(ACTION_HIGH-ACTION_LOW)
    return {k:float(v) for k,v in zip(ACTION_NAMES,values)}


def encode_action(parameters):
    vals=np.array([parameters[k] for k in ACTION_NAMES])
    return (2*(vals-ACTION_LOW)/(ACTION_HIGH-ACTION_LOW)-1).astype(np.float32)
