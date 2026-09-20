"""Cloud-only regression tests for bounded search and launch qualification."""
import numpy as np
import pytest
from toss.launch_probe import candidate_actions,case_grid,valid_launch
from toss.domain import load_settings
from toss.environment import TrainableTossEnv


def test_search_covers_bounds_and_midpoint_deterministically():
    a=candidate_actions(512,41)
    assert a.shape==(512,7)
    assert np.max(abs(a))<=1
    assert np.array_equal(a[0],np.zeros(7))
    assert len({tuple(x) for x in a[1:129]})==128
    assert np.array_equal(a,candidate_actions(512,41))


def test_grid_changes_only_requested_pancake_quantities():
    rows=case_grid(load_settings())
    assert len(rows)==9
    assert {r['pancake_diameter_mm'] for r in rows}=={95.,100.,105.}
    assert all(set(r)=={'pancake_diameter_mm','initial_x_mm','initial_y_mm'} for r in rows)


@pytest.mark.parametrize('fault',[{'unsafe':True},{'rejected':True},{'invalid_action':'limits'}])
def test_airborne_fault_is_not_qualifying_launch(fault):
    assert not valid_launch({'airborne':True,**fault})


def test_launch_does_not_claim_flip():
    assert valid_launch({'airborne':True,'unsafe':False,'success':False})
    assert not valid_launch({'airborne':False})


def test_motor_diagnostics_respect_caps_and_reset_each_attempt():
    cfg=load_settings();env=TrainableTossEnv(cfg,'hardware')
    try:
        for _ in range(2):
            env.reset(seed=17)
            assert env.motor_diagnostics is None
            _,_,_,_,info=env.step(np.zeros(7))
            d=info['motor_diagnostics']
            assert d['steps']>0
            assert np.all(np.array(d['peak_applied_torque_nm'])<=np.array([3.,1.2356379])+1e-9)
            assert np.all(np.array(d['saturation_fraction'])>=0)
            assert np.all(np.array(d['saturation_fraction'])<=1)
    finally:env.close()


def test_failed_launch_gate_prevents_ppo_and_training_bank(monkeypatch,tmp_path):
    import torch
    import stable_baselines3
    from toss import training,launch_probe,gpu_checks,warp_backend,feasibility
    cfg=load_settings()
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(feasibility,'coverage_report',lambda *args:{})
    monkeypatch.setattr(gpu_checks,'check_backend',lambda *args:{'passed':True})
    monkeypatch.setattr(launch_probe,'run_probe',lambda *args,**kwargs:{'ready_for_launch_training':False})
    def forbidden(*args,**kwargs):
        pytest.fail('PPO/training bank must not be created after a failed launch gate')
    monkeypatch.setattr(stable_baselines3,'PPO',forbidden)
    monkeypatch.setattr(warp_backend,'WarpVecEnv',forbidden)
    with pytest.raises(RuntimeError,match='PPO was not started'):
        training.train(cfg,tmp_path/'run',device='cpu')
