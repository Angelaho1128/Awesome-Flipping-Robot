"""Replay the canonical bounded baseline or a saved policy; no independent physics."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.config import settings

def main():
    import time
    import numpy as np
    import mujoco
    import mujoco.viewer
    from toss.environment import TrainableTossEnv
    from toss.training import load_policy
    from toss.domain import contract_hash
    import json
    parser=argparse.ArgumentParser()
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--policy',help='policy.zip with adjacent metadata.json')
    source.add_argument('--action-file',help='baseline_action.json from a successful cloud launch search')
    parser.add_argument('--seed',type=int,default=17)
    args=parser.parse_args()
    cfg=settings()
    env=TrainableTossEnv(cfg,'hardware',record=True)
    obs,_=env.reset(seed=args.seed)
    action=np.zeros(7,np.float32)
    if args.policy:
        policy,_=load_policy(args.policy,cfg)
        action=policy.predict(obs,deterministic=True)[0]
    if args.action_file:
        payload=json.loads(Path(args.action_file).read_text())
        if payload['contract_hash']!=contract_hash(cfg):
            raise ValueError('Baseline belongs to a different physics/action contract')
        action=np.asarray(payload['action'],dtype=np.float32)
        if action.shape!=(7,) or not np.isfinite(action).all() or np.any(abs(action)>1):
            raise ValueError('Invalid baseline action')
    q0=env.data.qpos.copy()
    _,reward,_,_,info=env.step(action)
    print({'reward':reward,'success':info['success'],'airborne':info['airborne'],'fault':info.get('reason')})
    frames=env.history or [{'t':0.,'qpos':q0.tolist()}]
    # Render recorded states. Wall-clock speed cannot change their physics.
    data=mujoco.MjData(env.model)
    with mujoco.viewer.launch_passive(env.model,data) as viewer:
        while viewer.is_running():
            started=time.monotonic()
            for frame in frames:
                if not viewer.is_running():break
                delay=started+frame['t']-time.monotonic()
                if delay>0:time.sleep(delay)
                data.qpos[:]=frame['qpos']
                mujoco.mj_forward(env.model,data)
                viewer.sync()
            time.sleep(.5)
    env.close()
if __name__=='__main__':main()
