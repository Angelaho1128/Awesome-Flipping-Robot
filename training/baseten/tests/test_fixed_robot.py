"""Cloud tests. No simulation or training is executed locally."""
from pathlib import Path
import copy
import numpy as np
import pytest
import mujoco
from toss.domain import load_settings,sample_case,observe,gravity_torque,feasibility_issues
from toss.model import build_xml
from toss.environment import TrainableTossEnv
from toss.trajectory import build_trajectory,sample_many


def config():
    return load_settings(Path(__file__).resolve().parents[1]/'settings.yaml')


def test_only_diameter_and_xy_are_randomized():
    cfg=config();rng=np.random.default_rng(41)
    cases=[sample_case(cfg,rng,'hardware') for _ in range(64)]
    varying={k for k in cases[0] if len({c[k] for c in cases})>1}
    assert varying=={'pancake_diameter_mm','initial_x_mm','initial_y_mm',
                     'observed_initial_x_mm','observed_initial_y_mm'}
    for c in cases:
        assert c['elbow_wrist_mm']==240 and c['wrist_pan_center_mm']==100
        assert c['pan_mass_kg']==.315 and c['pancake_mass_g']==25
        assert 95<=c['pancake_diameter_mm']<=105
        assert abs(c['initial_x_mm'])<=10 and abs(c['initial_y_mm'])<=10
        assert not feasibility_issues(c,cfg)


def test_geometry_mass_and_actuators_agree_with_configuration():
    cfg=config();c=sample_case(cfg,np.random.default_rng(2),'hardware')
    m=mujoco.MjModel.from_xml_string(build_xml(c,cfg))
    assert m.body_pos[m.body('pan').id,0]==pytest.approx(.24)
    assert m.geom_pos[m.geom('pan_floor').id,0]==pytest.approx(.1)
    assert m.body_mass[m.body('pan').id]==pytest.approx(.315+.11)
    assert m.body_mass[m.body('pancake').id]==pytest.approx(.025)
    assert m.actuator_gear[:,0]==pytest.approx([1,1])
    assert m.actuator_ctrlrange[:,1]==pytest.approx([3.,1.2356379])
    assert m.geom_size[m.geom('pancake').id,1]==pytest.approx(.00125)
    # The wrist motor is a geom on the arm body; its mass contributes to body inertia.
    assert m.body_mass[m.body('arm').id]==pytest.approx(.7+.18)
    assert gravity_torque(c,45,-45)[0]<3*.85


def test_valid_initialization_and_complete_toss_interface():
    env=TrainableTossEnv(config(),'hardware')
    obs,info=env.reset(seed=19)
    assert obs.shape==(23,) and not info['screen_issues']
    obs,reward,terminated,truncated,info=env.step(np.zeros(7,np.float32))
    assert np.isfinite(reward) and terminated and not truncated
    assert not info['rejected']
    assert isinstance(info['success'],bool)
    env.close()


def test_preparation_does_not_lower_below_home():
    cfg=config();c=sample_case(cfg,np.random.default_rng(1),'hardware')
    for x in [-1,0,1]:
        action=np.zeros(7);action[0]=x
        phases,_=build_trajectory(action,c,cfg)
        assert phases[0]['end'][0]>=cfg['control']['home_deg'][0]
        q=sample_many(phases,np.linspace(0,sum(p['duration'] for p in phases),10001))
        dt=sum(p['duration'] for p in phases)/10000
        assert np.abs(np.gradient(q,dt,axis=0)).max()<90.1
