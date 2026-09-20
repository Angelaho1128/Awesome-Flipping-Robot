import copy
from collections import Counter
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

from .domain import load_settings,contract,contract_hash,optimistic_holding_screen
from .environment import TossEnv,TrainableTossEnv


def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    temp.replace(path)


def versions():
    return {name:importlib.metadata.version(name) for name in
            ("torch","mujoco","mujoco-warp","warp-lang","gymnasium","stable-baselines3","numpy")}


def save_checkpoint(model,directory,cfg,profile,summary):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    model.save(directory/"policy.partial.zip")
    (directory/"policy.partial.zip").replace(directory/"policy.zip")
    write_json(directory/"metadata.json",{
        "contract":contract(cfg),"contract_hash":contract_hash(cfg),"config":cfg,
        "profile":profile,"simulated_only":True,"hardware_ready":False,
        "backend":cfg["training"].get("backend","cpu"), "versions":versions(),
        "algorithm":"PPO contextual one-action trajectory selection", **summary})
    write_json(directory/"READY.json",{"ready":True,"policy":"policy.zip"})
    return directory/"policy.zip"


def load_policy(path,cfg,device="cpu"):
    from stable_baselines3 import PPO
    path=Path(path)
    metadata=json.loads((path.parent/"metadata.json").read_text())
    if metadata["contract_hash"]!=contract_hash(cfg):
        raise ValueError("Policy physics/observation/action contract differs; use its saved configuration")
    return PPO.load(path,device=device),metadata


def validate_policy(policy,cfg,profile,count,seed,should_stop):
    """Fixed validation seeds select checkpoints; these are not final test data."""
    env=TrainableTossEnv(cfg,profile=profile,filter_infeasible=True)
    rewards=[];successes=0;airborne=0;rejected=0
    try:
        for i in range(count):
            if should_stop():return None
            obs,_=env.reset(seed=seed+i)
            action=policy.predict(obs,deterministic=True)[0]
            _,reward,_,_,info=env.step(action)
            rewards.append(float(reward));successes+=bool(info["success"])
            airborne+=bool(info["airborne"]);rejected+=bool(info.get("rejected"))
    finally:env.close()
    return {"mean_reward":float(np.mean(rewards)),"successes":successes,"airborne":airborne,
            "rejected":rejected,"cases":count,"seed":seed,
            "distribution":"conditioned feasible training domain; separate from unbiased final grid evaluation"}


def train(cfg,output,profile=None,smoke=False,resume=None,device="cuda"):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv,DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    cfg=copy.deepcopy(cfg);settings=cfg["training"]
    profile=profile or settings["profile"]
    if device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Check the H100 allocation and PyTorch image; no silent CPU learner fallback.")
    torch.set_num_threads(1)
    if device=="cuda":torch.set_float32_matmul_precision("high")
    if smoke:
        settings.update(indefinite=False,episodes=32,workers=2,n_steps=8,batch_size=16,n_epochs=2,
                        checkpoint_every_episodes=16,max_training_seconds=300,stop_if_no_airborne_after=10000,
                        validation_every_episodes=32,validation_cases=4)
    backend=settings.get("backend","cpu")
    if smoke and backend=="warp":
        settings.update(workers=4,n_steps=2,batch_size=8,episodes=16,validation_every_episodes=16)
    workers=int(settings["workers"])
    if workers<1 or settings["n_steps"]*workers % settings["batch_size"]:
        raise ValueError("workers × n_steps must be divisible by batch_size")
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    screen=optimistic_holding_screen(cfg,profile)
    write_json(output/"holding_screen.json",screen)
    if screen["conclusive"] and not screen["feasible"]:
        raise ValueError(f"No design can pass the initial static torque screen: {screen}. Change measured mechanics/limits; more training cannot fix this.")
    write_json(output/"run_config.json",cfg)
    write_json(output/"runtime.json",{"versions":versions(),"device":device,
               "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "workers":workers,"backend":backend,
               "note":"Warp physics on CUDA; model preparation and reference evaluation on cloud CPUs" if backend=="warp" else "MuJoCo physics on cloud CPUs"})
    print(json.dumps({"event":"run_start","profile":profile,"workers":workers,"device":device,
                      "indefinite":settings.get("indefinite",False),
                      "episode_budget":None if settings.get("indefinite") else settings["episodes"],
                      "time_budget_seconds":settings["max_training_seconds"]}),flush=True)
    from .feasibility import coverage_report
    write_json(output/"domain_coverage.json",coverage_report(cfg,profile))
    def factory():
        return Monitor(TrainableTossEnv(cfg,profile=profile,filter_infeasible=True))
    if backend=="warp":
        from .gpu_checks import check_backend
        from .warp_backend import WarpVecEnv
        check_backend(cfg,output,profile)
        if not smoke:
            from .launch_probe import run_probe
            launch=run_probe(cfg,output/'launch_probe',profile,parity_checked=True)
            if not launch['ready_for_launch_training']:
                raise RuntimeError('No confirmed launch: PPO was not started. Inspect launch_probe/launch_report.json')
            settings['baseline_action']=launch['selected']['action']
            write_json(output/'run_config.json',cfg)
        env=WarpVecEnv(cfg,profile,workers,cfg["seed"],output)
    elif backend=="cpu":
        if not smoke:raise ValueError("Full training requires the Warp launch-feasibility gate")
        env=DummyVecEnv([factory]) if workers==1 else SubprocVecEnv([factory for _ in range(workers)],start_method="spawn")
    else:
        raise ValueError(f"Unknown physics backend: {backend}")
    env.seed(cfg["seed"])
    signals={"stop":False}
    def stop_signal(signum,frame): signals["stop"]=True
    old_handlers={s:signal.signal(s,stop_signal) for s in (signal.SIGTERM,signal.SIGINT)}

    class Progress(BaseCallback):
        def __init__(self):
            super().__init__();self.start=time.monotonic();self.count=0;self.airborne=0;self.success=0
            self.rejections=Counter();self.faults=Counter();self.last_save=0;self.last_log=0;self.reason="running"
            self.reward_sum=0.;self.best_reward=None;self.best_policy=None
            self.last_save_time=0.
        def summary(self):
            return {"episodes_this_run":self.count,"total_model_timesteps":int(self.model.num_timesteps),
                    "airborne_attempts":self.airborne,"successful_flips":self.success,
                    "sampling_rejections":dict(self.rejections),"faults":dict(self.faults),
                    "mean_training_reward":self.reward_sum/max(1,self.count),
                    "best_validation_reward":self.best_reward,"best_policy":self.best_policy,
                    "elapsed_seconds":time.monotonic()-self.start,
                    "attempts_per_second":self.count/max(.001,time.monotonic()-self.start),
                    "stop_reason":self.reason}
        def _on_step(self):
            self.reward_sum+=float(np.sum(self.locals.get("rewards",[])))
            for info in self.locals.get("infos",[]):
                self.count+=1;self.airborne+=bool(info.get("airborne"));self.success+=bool(info.get("success"))
                self.rejections.update(info.get("rejected_before_reset",{}))
                if info.get("reason"):
                    reason=info["reason"];self.faults.update([str(reason)])
                if info.get("invalid_action"): self.faults.update([info["invalid_action"]])
            elapsed=time.monotonic()-self.start
            if (self.count-self.last_save>=settings["checkpoint_every_episodes"] or
                elapsed-self.last_save_time>=settings.get("checkpoint_every_seconds",300)):
                # Each named checkpoint is a direct child of BT_CHECKPOINT_DIR.
                # Loading one checkpoint therefore yields exactly one policy.
                save_checkpoint(self.model,output.parent/f"{output.name}_checkpoint_{self.model.num_timesteps:09d}",cfg,profile,self.summary())
                self.last_save=self.count
                self.last_save_time=elapsed
            if elapsed-self.last_log>=30:
                print(json.dumps({"event":"progress",**self.summary()}),flush=True);self.last_log=elapsed
            if signals["stop"]: self.reason="termination_signal";return False
            if (output/"STOP").exists(): self.reason="stop_file";return False
            if settings["max_training_seconds"] is not None and elapsed>=settings["max_training_seconds"]:
                self.reason="time_budget";return False
            if settings["stop_if_no_airborne_after"] is not None and self.count>=settings["stop_if_no_airborne_after"] and self.airborne==0:
                self.reason="no_airborne_attempts_check_actuator_feasibility";return False
            return True
    callback=Progress()
    try:
        if resume:
            model,meta=load_policy(resume,cfg,device)
            if meta.get("backend","cpu")!=backend:
                raise ValueError("Backend changed; start a new policy, do not resume old CPU training")
            if meta["profile"]!=profile: raise ValueError("Resume profile differs; keep the original profile")
            previous=meta["config"]["training"]
            if any(previous[k]!=settings[k] for k in ("workers","n_steps","batch_size","n_epochs","policy_hidden_sizes")):
                raise ValueError("Resume requires matching PPO rollout settings; do not resume a smoke checkpoint as a full run")
            model.set_env(env)
        else:
            model=PPO("MlpPolicy",env,device=device,seed=cfg["seed"],
                      n_steps=settings["n_steps"],batch_size=settings["batch_size"],
                      n_epochs=settings["n_epochs"],learning_rate=settings["learning_rate"],
                      ent_coef=settings["ent_coef"],gamma=1.,gae_lambda=1.,
                      policy_kwargs={"net_arch":{"pi":settings["policy_hidden_sizes"],"vf":settings["policy_hidden_sizes"]}},
                      tensorboard_log=str(output/"tensorboard"),verbose=1)
        if not resume and settings.get('baseline_action') is not None:
            # Start exploration near a confirmed trajectory, not a claimed learned policy.
            with torch.no_grad():
                model.policy.action_net.weight.zero_()
                model.policy.action_net.bias.copy_(torch.as_tensor(settings['baseline_action'],device=model.device))
                model.policy.log_std.fill_(float(np.log(.2)))
        first=True
        while callback.reason=="running":
            if signals["stop"] or (output/"STOP").exists():
                callback.reason="termination_signal" if signals["stop"] else "stop_file";break
            if settings["max_training_seconds"] is not None and time.monotonic()-callback.start>=settings["max_training_seconds"]:
                callback.reason="time_budget";break
            budget=int(settings["validation_every_episodes"])
            if not settings.get("indefinite",False):
                remaining=settings["episodes"]-callback.count
                if remaining<=0:callback.reason="episode_budget";break
                budget=min(budget,remaining)
            model.learn(total_timesteps=budget,callback=callback,reset_num_timesteps=first and not bool(resume))
            first=False
            if callback.reason!="running":break
            result=validate_policy(model,cfg,profile,settings["validation_cases"],settings["validation_seed"],
                                   lambda:signals["stop"] or (output/"STOP").exists())
            if result is None:continue
            write_json(output/f"validation_{model.num_timesteps:09d}.json",result)
            if callback.best_reward is None or result["mean_reward"]>callback.best_reward:
                callback.best_reward=result["mean_reward"]
                directory=output.parent/f"{output.name}_best_{model.num_timesteps:09d}"
                callback.best_policy=str(directory/"policy.zip")
                save_checkpoint(model,directory,cfg,profile,{**callback.summary(),"validation":result})
                write_json(output/"best.json",{"policy":callback.best_policy,"validation":result})
            print(json.dumps({"event":"validation",**result,"best_policy":callback.best_policy}),flush=True)
            if backend=="warp" and (settings.get("indefinite") or callback.count<settings["episodes"]):
                env.refresh_bank()
                # SB3 otherwise retains the previous bank observation at the boundary.
                model._last_obs=env.reset()
                model._last_episode_starts=np.ones(workers,dtype=bool)
        policy=save_checkpoint(model,output.parent/f"{output.name}_final",cfg,profile,callback.summary())
        write_json(output/"summary.json",callback.summary())
        print(json.dumps({"event":"training_finished","policy":str(policy),**callback.summary()}),flush=True)
        return policy
    finally:
        env.close()
        for sig,old in old_handlers.items():signal.signal(sig,old)


def evaluate(cfg,policy_path,output,profile="hardware",count=64,seed=1000000):
    policy,metadata=load_policy(policy_path,cfg)
    # Evaluation seeds never participate in the training callback or model tuning.
    env=TossEnv(cfg,profile=profile,filter_infeasible=False)
    rows=[]
    lengths=np.unique(np.linspace(*cfg["domain"]["elbow_wrist_mm"],4))
    offsets=np.unique(np.linspace(*cfg["domain"]["wrist_pan_center_mm"],4))
    for i in range(count):
        case_override={"elbow_wrist_mm":float(lengths[i%len(lengths)]),
                       "wrist_pan_center_mm":float(offsets[(i//len(lengths))%len(offsets)])}
        for method in ("baseline","policy"):
            obs,initial=env.reset(seed=seed+i,options={"case":case_override})
            action=np.asarray(metadata["config"]["training"].get("baseline_action",[0.]*7),dtype=np.float32) if method=="baseline" else policy.predict(obs,deterministic=True)[0]
            _,reward,_,_,info=env.step(action)
            rows.append({"case_id":i,"method":method,"reward":reward,"action":action.tolist(),**info})
    env.close()
    summaries={}
    for method in ("baseline","policy"):
        subset=[r for r in rows if r["method"]==method];eligible=[r for r in subset if not r.get("rejected")]
        summaries[method]={"total_cases":len(subset),"eligible_cases":len(eligible),
                           "mean_reward":float(np.mean([r["reward"] for r in subset])) if subset else None,
                           "rejected_cases":len(subset)-len(eligible),
                           "airborne":sum(r["airborne"] for r in eligible),
                           "successes":sum(r["success"] for r in eligible),
                           "success_rate_all_cases":sum(r["success"] for r in subset)/max(1,len(subset)),
                           "success_rate_eligible":sum(r["success"] for r in eligible)/max(1,len(eligible))}
    by_length=[]
    for L in lengths:
        for r in offsets:
            subset=[row for row in rows if row["method"]=="policy" and
                    row["case"]["elbow_wrist_mm"]==L and row["case"]["wrist_pan_center_mm"]==r]
            if subset:
                by_length.append({"elbow_wrist_mm":float(L),"wrist_pan_center_mm":float(r),
                                  "cases":len(subset),"rejected":sum(bool(x.get("rejected")) for x in subset),
                                  "successes":sum(x["success"] for x in subset)})
    report={"simulated_only":True,"profile":profile,"seed":seed,"summary":summaries,
            "coverage_by_lengths":by_length,"trials":rows,
            "interpretation":"Results on unseen pancake size/location cases with fixed measured geometry; not physical validation or proof of arbitrary-design generalization."}
    write_json(output,report)
    print(json.dumps({"event":"evaluation","profile":profile,**summaries}),flush=True)
    return report
