import numpy as np
from .domain import decode_action


def smooth(s):
    s=np.clip(s,0,1)
    return 10*s**3-15*s**4+6*s**5


def build_trajectory(action,case,cfg):
    p=decode_action(action); home=np.asarray(cfg["control"]["home_deg"],float)
    prep=np.array([p["prepare_shoulder_deg"],-p["prepare_shoulder_deg"]])
    release=np.array([prep[0]+p["shoulder_excursion_deg"],0.])
    release[1]=p["release_pitch_deg"]-release[0]
    vmax=np.array([case[k+"_speed_deg_s"] for k in ("shoulder","wrist")])
    amax=np.array([case[k+"_accel_deg_s2"] for k in ("shoulder","wrist")])
    phases=[]
    for name,a,b,timing,requested in [("prepare",home,prep,0.,.5),
                                      ("launch",prep,release,p["wrist_timing"],p["launch_s"]),
                                      ("hold",release,release,0.,p["release_hold_s"]),
                                      ("return",release,home,0.,p["return_s"])]:
        delta=abs(b-a)
        required=np.maximum(1.875*delta/vmax,np.sqrt(5.7735027*delta/amax))
        required[1]/=1-abs(timing)
        duration=float(max(requested,required.max(),.01)*1.02)
        phases.append(dict(name=name,start=a.tolist(),end=b.tolist(),duration=duration,wrist_timing=float(timing)))
    total=sum(p["duration"] for p in phases)
    if total+cfg["physics"]["settle_after_s"] > cfg["physics"]["max_episode_s"]:
        raise ValueError("trajectory_exceeds_episode_duration")
    times=np.linspace(0,total,max(501,int(total/.005)))
    q=sample_many(phases,times)
    limits=np.array([cfg["control"]["shoulder_limits_deg"],cfg["control"]["wrist_limits_deg"]])
    if np.any(q<limits[:,0]) or np.any(q>limits[:,1]):
        raise ValueError("command_joint_limits")
    pitch=q.sum(axis=1);lo,hi=cfg["control"]["pan_pitch_limits_deg"]
    if pitch.min()<lo or pitch.max()>hi:
        raise ValueError("command_pan_pitch_limits")
    return phases,p


def sample(phases,t):
    for phase in phases:
        if t <= phase["duration"]:
            s=max(0,t)/phase["duration"]; offset=phase["wrist_timing"]
            sw=(s-max(offset,0))/(1-abs(offset))
            return np.asarray(phase["start"])+(np.asarray(phase["end"])-phase["start"])*np.array([smooth(s),smooth(sw)])
        t-=phase["duration"]
    return np.asarray(phases[-1]["end"],float)


def sample_many(phases,times):
    times=np.maximum(np.asarray(times,float),0.)
    out=np.tile(phases[-1]["end"],(len(times),1)).astype(float)
    elapsed=0.
    for phase in phases:
        mask=(times>=elapsed)&(times<=elapsed+phase["duration"])
        s=(times[mask]-elapsed)/phase["duration"];offset=phase["wrist_timing"]
        sw=(s-max(offset,0))/(1-abs(offset))
        factors=np.column_stack((smooth(s),smooth(sw)))
        out[mask]=np.asarray(phase["start"])+(np.asarray(phase["end"])-phase["start"])*factors
        elapsed+=phase["duration"]
    return out
