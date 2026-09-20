"""Run ONLY on the cloud GPU. A gate, not a claim of physical validation."""
import copy
import numpy as np
from .environment import TrainableTossEnv
from .warp_backend import WarpBatch
from .training import write_json


def check_backend(cfg,output,profile):
    envs=[]
    sampler=TrainableTossEnv(cfg,profile)
    # Distinct compiled masses, inertias, sizes and geometry must survive batching.
    for i in range(4):
        sampler.reset(seed=7000000 if i==0 else None)
        envs.append(copy.copy(sampler))
    gpu=WarpBatch(cfg,envs)
    report={'simulated_only':True,'passed':False,'checks':[],
            'batching':{'sleeping_enabled':False,'shared_inactive_fields':['dof_length']},
            'note':'CPU/GPU parity at these cases is necessary, not real-world validation.'}
    initial=[(e.data.qpos.copy(),e.data.qvel.copy()) for e in envs]
    # First establish short-horizon dynamics parity, before contact divergence.
    import mujoco
    import warp as wp
    targets=np.tile(np.radians(cfg['control']['home_deg']), (4,gpu.max_steps,1)).astype(np.float32)
    gpu.target.assign(targets)
    gpu.velocity.zero_()
    gpu.lengths.assign(np.full(4,10,np.int32))
    gpu.reset()
    for e in envs:
        e.data.qacc_warmstart[:]=0
    with wp.ScopedDevice('cuda:0'):
        for _ in range(10):
            gpu._step()
            for e in envs:
                e._advance(e.qhome,np.zeros(2))
    actual_q=gpu.d.qpos.numpy()
    actual_v=gpu.d.qvel.numpy()
    for i,e in enumerate(envs):
        qerr=float(np.max(abs(e.data.qpos-actual_q[i])))
        verr=float(np.max(abs(e.data.qvel-actual_v[i])))
        passed=bool(np.isfinite(actual_q[i]).all() and np.isfinite(actual_v[i]).all()
                    and qerr<.005 and verr<.10)
        report['checks'].append({'type':'20ms_dynamics','world':i,'max_q_error':qerr,
                                  'max_qvel_error':verr,'passed':passed})
    # Restore the exact same settled states before each entire toss comparison.
    # Slow, midpoint and rapid trajectories exercise bounds/faults and contacts.
    for label,action in [('midpoint',np.zeros(7)),('slow',np.array([.8,-.8,0,1,1,0,0])),
                         ('rapid',np.array([1,1,.5,-1,-1,0,-1]))]:
        for e,(q,v) in zip(envs,initial):
            e.data.qpos[:]=q;e.data.qvel[:]=v;e.data.qacc_warmstart[:]=0
            mujoco.mj_forward(e.model,e.data);e.done=False
        rewards,infos=gpu.run(np.tile(action,(4,1)))
        for i,e in enumerate(envs):
            _,reward,_,_,info=e.step(action)
            error=abs(float(rewards[i])-reward)
            # Contact timing need not be bit-identical. Success/fault/airborne
            # labels and useful reward agreement must match for this gate.
            passed=(error<.25 and info['success']==infos[i]['success']
                    and info['airborne']==infos[i]['airborne']
                    and info.get('reason')==infos[i].get('reason')
                    and info.get('invalid_action')==infos[i].get('invalid_action'))
            report['checks'].append({'type':label,'world':i,'cpu_reward':reward,
                'gpu_reward':float(rewards[i]),'cpu_reason':info.get('reason'),
                'gpu_reason':infos[i].get('reason'),'passed':bool(passed)})
    report['passed']=all(r['passed'] for r in report['checks'])
    write_json(output/'gpu_parity.json',report)
    if not report['passed']:
        raise RuntimeError('GPU/CPU parity gate failed; read gpu_parity.json. Training was not started.')
    print('GPU/CPU parity gate passed; no physical accuracy claim is implied.',flush=True)
    return report
