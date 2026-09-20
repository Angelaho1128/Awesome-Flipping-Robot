"""Each episode chooses a bounded complete toss; no cloud motor feedback loop."""
from collections import Counter
import copy

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from .domain import (FEATURE_SCALES, sample_case, observe, feasibility_issues,
                     geometry_issues, gravity_torque,condition_motor_capacity)
from .model import build_xml
from .trajectory import build_trajectory,sample_many


def successful_state(rel,normal_dot,radius,inner_radius,half_thickness,linear_speed,
                     angular_speed,airborne,floor_contact,margin):
    return bool(airborne and floor_contact and normal_dot < -.90
                and np.linalg.norm(rel[:2])+radius <= inner_radius-margin
                and -.002 <= rel[2] <= half_thickness+.008
                and linear_speed < .06 and angular_speed < .6)


class TossEnv(gym.Env):
    metadata={"render_modes":[]}

    def __init__(self,cfg,profile="hardware",filter_infeasible=True,record=False):
        self.cfg=copy.deepcopy(cfg);self.profile=profile
        self.filter_infeasible=filter_infeasible;self.record=record
        self.observation_space=spaces.Box(-2.,2.,shape=(len(FEATURE_SCALES),),dtype=np.float32)
        self.action_space=spaces.Box(-1.,1.,shape=(7,),dtype=np.float32)
        self.done=True;self.history=[]

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        overrides=(options or {}).get("case")
        rejections=Counter()
        for attempt in range(self.cfg["domain"]["maximum_sampling_attempts"]):
            case=sample_case(self.cfg,self.np_random,self.profile,overrides)
            if self.filter_infeasible and overrides is None and self.cfg["domain"].get("condition_motor_sampling_on_load",False):
                condition_motor_capacity(case,self.cfg,self.profile,self.np_random)
            issues=feasibility_issues(case,self.cfg)
            if overrides is not None or not self.filter_infeasible or not issues:
                break
            rejections.update(issues)
        else:
            # Report failed sampling as a rejected episode, not a worker crash.
            issues=list(issues)+["sampling_budget_exhausted"]
        self.case=case;self.obs=observe(case);self.done=False;self.history=[];self.motor_diagnostics=None
        self.reset_info={"case":case,"screen_issues":issues,"rejected_before_reset":dict(rejections)}
        self.blocked=issues
        # Screening is reported, never converted into a secretly stronger motor.
        if issues:
            return self.obs.copy(),self.reset_info.copy()
        self.xml=build_xml(case,self.cfg)
        self.model=mujoco.MjModel.from_xml_string(self.xml)
        self.data=mujoco.MjData(self.model)
        self.pan=self.model.body("pan").id;self.cake=self.model.body("pancake").id
        self.cake_geom=self.model.geom("pancake").id
        self.pan_floor=self.model.geom("pan_floor").id
        self.pan_geoms={self.pan_floor}|{self.model.geom(f"rim_{i}").id for i in range(self.cfg["physics"]["rim_segments"])}
        self.free_q=self.model.jnt_qposadr[self.model.joint("pancake_free").id]
        self.r=case["wrist_pan_center_mm"]/1000
        self.radius=case["pancake_diameter_mm"]/2000
        self.half_thickness=case["pancake_thickness_mm"]/2000
        self.R=case["pan_diameter_mm"]/2000
        self.inner_R=min(self.R-self.cfg["physics"]["pan_wall_mm"]/1000,
                         case["pan_bottom_diameter_mm"]/2000)
        self.qhome=np.radians(self.cfg["control"]["home_deg"])
        self.kp=np.asarray(self.cfg["control"]["kp_nm_rad"])
        self.kd=np.asarray(self.cfg["control"]["kd_nm_s_rad"])
        self.caps=np.array([case[k+"_torque_nm"] for k in ("shoulder","wrist")])*case["torque_multiplier"]
        self.cutoff=np.array([case[k+"_zero_torque_speed_deg_s"] for k in ("shoulder","wrist")])
        self.vmax=np.array([case[k+"_speed_deg_s"] for k in ("shoulder","wrist")])
        self.data.qpos[:2]=self.qhome
        mujoco.mj_forward(self.model,self.data)
        rot=self.data.xmat[self.pan].reshape(3,3)
        rel=np.array([self.r+case["initial_x_mm"]/1000,case["initial_y_mm"]/1000,self.half_thickness+.0003])
        self.data.qpos[self.free_q:self.free_q+3]=self.data.xpos[self.pan]+rot@rel
        self.data.qpos[self.free_q+3:self.free_q+7]=self.data.xquat[self.pan]
        mujoco.mj_forward(self.model,self.data)
        # Fixed preload at the home pose, analogous to referencing an energized
        # loaded mechanism. No perfect gravity compensation during a toss.
        self.preload=gravity_torque(case,*self.cfg["control"]["home_deg"])/self.kp
        for _ in range(round(self.cfg["physics"]["settle_before_s"]/self.model.opt.timestep)):
            self._advance(self.qhome,np.zeros(2))
        rel,dot,_,_,on_pan,_,other=self._measure()
        if other or not on_pan or dot<.95 or np.linalg.norm(rel[:2])+self.radius > self.inner_R:
            self.blocked=["unstable_initial_state"]
        if np.max(abs(self.data.qpos[:2]-self.qhome))>np.radians(5):
            self.blocked=["cannot_hold_initial_pose"]
        # Observe the settled starting position, not the pre-settling placement.
        noise=np.array([case["observed_initial_x_mm"]-case["initial_x_mm"],
                        case["observed_initial_y_mm"]-case["initial_y_mm"]])
        self.case.update(initial_x_mm=float(rel[0]*1000),initial_y_mm=float(rel[1]*1000),
                         observed_initial_x_mm=float(rel[0]*1000+noise[0]),
                         observed_initial_y_mm=float(rel[1]*1000+noise[1]),
                         home_shoulder_deg=float(np.degrees(self.data.qpos[0])),
                         home_wrist_deg=float(np.degrees(self.data.qpos[1])))
        if not self.blocked:
            self.obs=observe(self.case)
        self.reset_info["screen_issues"]=self.blocked
        return self.obs.copy(),self.reset_info.copy()

    def _advance(self,target,target_velocity):
        speed=abs(np.degrees(self.data.qvel[:2]))
        # Explicit uncalibrated envelope: linearly falls to zero at cutoff.
        cap=self.caps*np.clip(1-speed/self.cutoff,0,1)
        torque=self.kp*(target+self.preload-self.data.qpos[:2])+self.kd*(target_velocity-self.data.qvel[:2])
        self.data.ctrl[:]=np.clip(torque,-cap,cap)
        if self.motor_diagnostics is not None:
            d=self.motor_diagnostics;d['steps']+=1
            d['saturated_steps']+=(abs(torque)>cap+1e-9).astype(int)
            d['peak_requested_torque_nm']=np.maximum(d['peak_requested_torque_nm'],abs(torque))
            d['peak_applied_torque_nm']=np.maximum(d['peak_applied_torque_nm'],abs(self.data.ctrl))
            d['peak_speed_deg_s']=np.maximum(d['peak_speed_deg_s'],speed)

        mujoco.mj_step(self.model,self.data)
        mujoco.mj_forward(self.model,self.data)
        if self.motor_diagnostics is not None:
            self.motor_diagnostics['peak_speed_deg_s']=np.maximum(
                self.motor_diagnostics['peak_speed_deg_s'],abs(np.degrees(self.data.qvel[:2])))

    def _measure(self):
        rot=self.data.xmat[self.pan].reshape(3,3)
        rel=rot.T@(self.data.xpos[self.cake]-self.data.xpos[self.pan])-np.array([self.r,0,0])
        normal=float(rot[:,2]@self.data.xmat[self.cake].reshape(3,3)[:,2])
        cv=np.zeros(6);pv=np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_XBODY,self.cake,cv,0)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_XBODY,self.pan,pv,0)
        pv_at_cake=pv[3:]+np.cross(pv[:3],self.data.xpos[self.cake]-self.data.xpos[self.pan])
        linear=float(np.linalg.norm(cv[3:]-pv_at_cake));angular=float(np.linalg.norm(cv[:3]-pv[:3]))
        on_pan=False;on_floor=False;other=False
        for contact in self.data.contact:
            if self.cake_geom in (contact.geom1,contact.geom2):
                g=contact.geom2 if contact.geom1==self.cake_geom else contact.geom1
                on_pan|=g in self.pan_geoms;on_floor|=g==self.pan_floor;other|=g not in self.pan_geoms
            elif contact.geom1 in self.pan_geoms or contact.geom2 in self.pan_geoms:
                other=True
        return rel,normal,linear,angular,on_pan,on_floor,other

    def step(self,action):
        if self.done:
            raise RuntimeError("Reset required")
        self.done=True
        base={**self.reset_info,"success":False,"airborne":False,"unsafe":False}
        if self.blocked:
            return self.obs.copy(),-10.,True,False,{**base,"rejected":True,"reason":self.blocked}
        try:
            phases,parameters=build_trajectory(action,self.case,self.cfg)
        except ValueError as exc:
            return self.obs.copy(),-5.,True,False,{**base,"rejected":False,"invalid_action":str(exc)}
        self.motor_diagnostics={'steps':0,'saturated_steps':np.zeros(2,dtype=int),
            'peak_requested_torque_nm':np.zeros(2),'peak_applied_torque_nm':np.zeros(2),
            'peak_speed_deg_s':np.zeros(2)}
        p=self.cfg["physics"];dt=self.model.opt.timestep
        duration=sum(x["duration"] for x in phases)
        ts=np.arange(0,duration+p["settle_after_s"]+dt,dt)
        targets=np.radians(sample_many(phases,ts-self.case["actuator_delay_s"]))
        velocities=np.gradient(targets,dt,axis=0)
        airtime=0.;max_airtime=0.;airborne=False;settled=0.;max_tracking=0.;peak_height=0.
        best_orientation=0.;flight_rotation=0.;unsafe=False;reason=None
        limits=np.radians([self.cfg["control"]["shoulder_limits_deg"],self.cfg["control"]["wrist_limits_deg"]])
        for i,(target,velocity) in enumerate(zip(targets,velocities)):
            self._advance(target,velocity)
            rel,dot,linear,angular,on_pan,on_floor,other=self._measure()
            if not on_pan and rel[2]>self.half_thickness+.003:
                airtime+=dt;flight_rotation+=angular*dt
                max_airtime=max(max_airtime,airtime)
                airborne|=airtime>=p["min_airtime_s"]
            else:
                airtime=0.
            peak_height=max(peak_height,float(rel[2]-self.half_thickness))
            if airborne:
                best_orientation=max(best_orientation,(1-dot)/2)
            stable=successful_state(rel,dot,self.radius,self.inner_R,self.half_thickness,linear,
                                    angular,airborne,on_floor,p["edge_margin_mm"]/1000)
            # Exclude repeated full rotations; orientation alone cannot count turns.
            stable=stable and 2.2 <= flight_rotation <= 4.3
            settled=settled+dt if stable else 0.
            tracking=float(max(abs(np.degrees(target-self.data.qpos[:2]))))
            max_tracking=max(max_tracking,tracking)
            if other or self.data.xpos[self.cake,2]<.005:
                unsafe=True;reason="ground_or_other_contact"
            elif not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                unsafe=True;reason="nonfinite_state"
            elif np.any(self.data.qpos[:2]<limits[:,0]-.02) or np.any(self.data.qpos[:2]>limits[:,1]+.02):
                unsafe=True;reason="actual_joint_limits"
            elif np.any(abs(np.degrees(self.data.qvel[:2]))>self.vmax*1.15):
                unsafe=True;reason="actual_speed_limit"
            elif tracking>p["tracking_fault_deg"]:
                unsafe=True;reason="tracking_fault"
            if self.record and i%10==0:
                self.history.append({"t":float(ts[i]),"qpos":self.data.qpos.tolist(),"target_deg":np.degrees(target).tolist(),
                                     "torque_nm":self.data.ctrl.tolist(),"pancake_relative_m":rel.tolist(),"normal_dot":dot})
            if unsafe:
                break
        success=bool(settled>=p["success_settling_s"] and not unsafe)
        offset=float(np.linalg.norm(rel[:2]));contained=offset+self.radius<=self.inner_R-p["edge_margin_mm"]/1000
        # Dense launch/rotation shaping is explicitly separate from the success label.
        reward=.2*min(peak_height/.02,1)+float(airborne)*(1+best_orientation)
        reward+=float(airborne and contained and on_floor)*max(0,1-offset/self.R)
        reward+=15*success-.02*max_tracking
        if unsafe: reward=-5.
        info={**base,"success":success,"airborne":bool(airborne),"unsafe":unsafe,"reason":reason,
              "rejected":False,"airtime_s":max_airtime,"flight_rotation_rad":flight_rotation,
              "peak_height_m":peak_height,"normal_dot":dot,"landing_offset_m":offset,"settled_s":settled,
              "max_tracking_error_deg":max_tracking,"duration_s":duration,"parameters":parameters}
        d=self.motor_diagnostics
        info['motor_diagnostics']={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in d.items()}
        info['motor_diagnostics']['joint_order']=['shoulder','wrist']
        info['motor_diagnostics']['saturation_fraction']=(d['saturated_steps']/max(1,d['steps'])).tolist()
        return self.obs.copy(),float(reward),True,False,info


class TrainableTossEnv(TossEnv):
    """Never expose a rejected reset as a policy transition.

    Invalid policy actions still earn their normal penalty. Failure to establish
    a valid physical initial state aborts setup instead of training on -10 noise.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.sampler=None

    def reset(self,seed=None,options=None):
        from .feasibility import FeasibleCaseSampler,FeasibilityError
        if options:
            raise ValueError('Use TossEnv for explicit, possibly infeasible evaluation cases')
        if seed is not None or self.sampler is None:
            self.sampler=FeasibleCaseSampler(self.cfg,self.profile,seed)
        failures=Counter()
        before=self.sampler.counts.copy()
        for _ in range(self.cfg['training'].get('initial_state_attempts',32)):
            case=self.sampler.sample()
            # Derived/observed values are recomputed by the ordinary reset.
            overrides={k:v for k,v in case.items() if k not in
                       ('link_mass_kg','observed_initial_x_mm','observed_initial_y_mm')}
            obs,info=super().reset(seed=seed,options={'case':overrides})
            seed=None
            if not self.blocked:
                setup=(self.sampler.counts-before)+failures
                info['sampling_setup_counts']=dict(setup)
                info['rejected_before_reset']={k:v for k,v in setup.items()
                    if k not in ('geometric_candidates','static_eligible')}
                self.reset_info=info
                return obs,info
            failures.update(self.blocked)
        raise FeasibilityError(f'Cannot establish a stable initial state: {dict(failures)}. '
                               'No rejected reset was sent to PPO.')
