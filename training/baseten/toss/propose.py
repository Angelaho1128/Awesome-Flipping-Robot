"""Generate an offline trajectory proposal; never writes G-code or motor I/O."""
import argparse
import json
from pathlib import Path
import numpy as np
from .domain import sample_case,observe,feasibility_issues
from .trajectory import build_trajectory
from .training import load_policy,write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--policy",required=True)
    parser.add_argument("--spec",required=True)
    parser.add_argument("--output",default="proposal.json")
    args=parser.parse_args()
    meta=json.loads((Path(args.policy).parent/"metadata.json").read_text())
    cfg=meta["config"];spec=json.loads(Path(args.spec).read_text())
    profile=spec.pop("motor_profile",cfg["training"]["profile"])
    defaults={k:sum(v)/2 for group in (cfg["domain"],cfg["motor_profiles"][profile])
              for k,v in group.items() if isinstance(v,list)}
    defaults.update(initial_x_mm=0.,initial_y_mm=0.)
    defaults.update(spec)
    for name,expected in (("pan_diameter_mm",cfg["physics"]["pan_diameter_mm"]),
                          ("pan_bottom_diameter_mm",cfg["physics"]["pan_bottom_diameter_mm"]),
                          ("wrist_motor_mass_kg",cfg["physics"]["wrist_motor_mass_kg"])):
        if name in spec and spec[name]!=expected:
            raise ValueError(f"{name} differs from the trained fixed geometry; update physics and retrain")
    for name in ("pancake_mass_g","pancake_diameter_mm","elbow_wrist_mm","wrist_pan_center_mm",
                 "pan_mass_kg","mount_mass_kg","pancake_thickness_mm"):
        lo,hi=cfg["domain"][name]
        if not lo <= defaults[name] <= hi:
            raise ValueError(f"{name} is outside training range [{lo}, {hi}]; expand training and evaluate first")
    # Compare against the union of both configured motor envelopes.
    for name in cfg["motor_profiles"][profile]:
        lo=min(p[name][0] for p in cfg["motor_profiles"].values())
        hi=max(p[name][1] for p in cfg["motor_profiles"].values())
        if not lo<=defaults[name]<=hi: raise ValueError(f"{name} outside trained motor envelope")
    case=sample_case(cfg,np.random.default_rng(0),profile,defaults,noisy=False)
    issues=feasibility_issues(case,cfg)
    if issues: raise ValueError("Proposal refused: "+", ".join(issues))
    policy,_=load_policy(args.policy,cfg)
    action=policy.predict(observe(case),deterministic=True)[0]
    phases,parameters=build_trajectory(action,case,cfg)
    write_json(args.output,{"hardware_ready":False,"simulated_policy_only":True,"case":case,
                            "action":action.tolist(),"parameters":parameters,"phases":phases,
                            "note":"Validate measured motor response, tracking, geometry and physical trials before translating into local controller commands."})
    print(args.output)


if __name__=="__main__":main()
