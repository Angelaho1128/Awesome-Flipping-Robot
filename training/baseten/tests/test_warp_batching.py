"""Cloud-only regression checks for the actual compiled-model batching failure."""
from types import SimpleNamespace
import numpy as np
import pytest
import mujoco
import warp as wp
import mujoco_warp as mjw

from toss.domain import load_settings,sample_case
from toss.model import build_xml
from toss.warp_backend import _merge_models,_require_awake_models


def _models():
    cfg=load_settings()
    models=[]
    # Model compilation only: unequal lengths intentionally reproduce the failure.
    with wp.ScopedDevice('cpu'):
        for length in (270.,330.):
            case=sample_case(cfg,np.random.default_rng(5),overrides={'elbow_wrist_mm':length})
            model=mujoco.MjModel.from_xml_string(build_xml(case,cfg))
            assert not int(model.opt.enableflags)&int(mujoco.mjtEnableBit.mjENBL_SLEEP)
            models.append(mjw.put_model(model))
    return models


def test_different_compiled_dof_lengths_are_allowed_only_with_sleep_disabled(monkeypatch):
    from toss import warp_backend
    models=_models()
    first=models[0].dof_length.numpy().copy()
    assert not np.array_equal(first,models[1].dof_length.numpy())
    copied=[]
    # Exercise the real compatibility guard without allocating CUDA copies here;
    # check_backend separately runs actual GPU copies and dynamics on the cloud.
    monkeypatch.setattr(warp_backend,'_copy_batched_fields',lambda *args:copied.append(args))
    _merge_models(models[0],models)
    assert len(copied)==1
    assert np.array_equal(first,models[0].dof_length.numpy())


@pytest.mark.parametrize('enabled_world',[0,1,2])
def test_sleep_enabled_in_destination_or_any_source_is_refused(enabled_world):
    models=[SimpleNamespace(opt=SimpleNamespace(enableflags=0)) for _ in range(3)]
    models[enabled_world].opt.enableflags=int(mujoco.mjtEnableBit.mjENBL_SLEEP)
    with pytest.raises(RuntimeError,match='Sleeping must be disabled'):
        _require_awake_models(models)


def test_other_unbatchable_differences_are_still_rejected_and_all_reported():
    models=_models()
    with wp.ScopedDevice('cpu'):
        for name in ('dof_parentid','geom_bodyid'):
            array=getattr(models[1],name)
            values=array.numpy().copy()
            values[-1]+=1
            setattr(models[1],name,wp.array(values,dtype=array.dtype))
    with pytest.raises(RuntimeError) as error:
        _merge_models(models[0],models)
    assert 'dof_parentid' in str(error.value)
    assert 'geom_bodyid' in str(error.value)


def test_dependency_upgrade_requires_reaudit(monkeypatch):
    from toss import warp_backend
    monkeypatch.setattr(warp_backend.importlib.metadata,'version',lambda name:'99.0.0')
    with pytest.raises(RuntimeError,match='audited MuJoCo Warp 3.13.0'):
        _require_awake_models([SimpleNamespace(opt=SimpleNamespace(enableflags=0))])
