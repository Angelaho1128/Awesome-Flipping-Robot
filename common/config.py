"""Shared configuration. Units on these compatibility constants are SI."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
TRAINING_ROOT=ROOT/'training'/'baseten'
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0,str(TRAINING_ROOT))
from toss.domain import load_settings,sample_case
SETTINGS_PATH=TRAINING_ROOT/'settings.yaml'
def settings():
    return load_settings(SETTINGS_PATH)
def nominal_case():
    import numpy as np
    cfg=settings()
    overrides={k:(v[0]+v[1])/2 for k,v in cfg['domain'].items() if isinstance(v,list)}
    overrides.update(initial_x_mm=0.,initial_y_mm=0.)
    return sample_case(cfg,np.random.default_rng(0),'hardware',overrides,noisy=False)
_cfg=settings()
ELBOW_TO_WRIST=_cfg['domain']['elbow_wrist_mm'][0]/1000
WRIST_TO_PAN=_cfg['domain']['wrist_pan_center_mm'][0]/1000
PANCAKE_RADIUS=sum(_cfg['domain']['pancake_diameter_mm'])/4000
PANCAKE_THICKNESS=_cfg['domain']['pancake_thickness_mm'][0]/1000
PANCAKE_MASS=_cfg['domain']['pancake_mass_g'][0]/1000
PAN_MASS=_cfg['domain']['pan_mass_kg'][0]
SHOULDER_TORQUE_NM=_cfg['motor_profiles']['hardware']['shoulder_torque_nm'][0]
WRIST_TORQUE_NM=_cfg['motor_profiles']['hardware']['wrist_torque_nm'][0]
