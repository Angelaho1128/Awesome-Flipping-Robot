"""Cloud-only batched MuJoCo Warp backend, pinned to mujoco-warp 3.13.0.

Compiles a refreshed bank of independently screened models on the host. Physics,
PD control, contact checks and reward accumulation run on CUDA. Only observations,
actions and batch outcomes cross the host boundary. This is trajectory selection,
not a real-time camera policy. No hardware commands are produced.
"""
import dataclasses
import importlib.metadata
import mujoco
import json
import time
import os
import numpy as np
import warp as wp
import mujoco_warp as mjw
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from .domain import FEATURE_SCALES
from .environment import TrainableTossEnv
from .trajectory import build_trajectory, sample_many

if os.environ.get("WARP_CACHE_PATH"):
    wp.config.kernel_cache_dir=os.environ["WARP_CACHE_PATH"]


@wp.kernel
def control(q:wp.array2d(dtype=float),v:wp.array2d(dtype=float),ctrl:wp.array2d(dtype=float),
            warm:wp.array2d(dtype=float),q0:wp.array2d(dtype=float),v0:wp.array2d(dtype=float),
            target:wp.array3d(dtype=float),velocity:wp.array3d(dtype=float),
            params:wp.array2d(dtype=float), gains:wp.array(dtype=float),
            stats:wp.array2d(dtype=float),lengths:wp.array(dtype=int),index:wp.array(dtype=int),
            flags:wp.array2d(dtype=int)):
    w=wp.tid()
    flags[w,0]=0
    flags[w,1]=0
    flags[w,2]=0
    i=index[0]
    if i>=lengths[w] or stats[w,8]!=0.0:
        for k in range(9):
            q[w,k]=q0[w,k]
        for k in range(8):
            v[w,k]=v0[w,k]
            warm[w,k]=0.0
        for k in range(2):
            ctrl[w,k]=params[w,12+k]*gains[k]
    else:
        for k in range(2):
            cap=params[w,6+k]*wp.clamp(1.0-wp.abs(v[w,k])*57.2957795/params[w,8+k],0.0,1.0)
            torque=gains[k]*(target[w,i,k]+params[w,12+k]-q[w,k])+gains[k+2]*(velocity[w,i,k]-v[w,k])
            ctrl[w,k]=wp.clamp(torque,-cap,cap)


@wp.kernel
def contacts(geom:wp.array(dtype=wp.vec2i),world:wp.array(dtype=int),n:wp.array(dtype=int),
             cake:int,floor:int,rim_start:int,rim_end:int,flags:wp.array2d(dtype=int)):
    i=wp.tid()
    if i<n[0]:
        w=world[i]
        a=geom[i][0]
        b=geom[i][1]
        ap=a==floor or (a>=rim_start and a<=rim_end)
        bp=b==floor or (b>=rim_start and b<=rim_end)
        if a==cake or b==cake:
            g=a
            if a==cake:
                g=b
            if g==floor or (g>=rim_start and g<=rim_end):
                wp.atomic_max(flags,w,0,1)
                if g==floor:
                    wp.atomic_max(flags,w,1,1)
            else:
                wp.atomic_max(flags,w,2,1)
        elif ap or bp:
            wp.atomic_max(flags,w,2,1)


@wp.kernel
def measure(q:wp.array2d(dtype=float),v:wp.array2d(dtype=float),
            pos:wp.array2d(dtype=wp.vec3),mat:wp.array2d(dtype=wp.mat33),
            target:wp.array3d(dtype=float),params:wp.array2d(dtype=float),
            config:wp.array(dtype=float),flags:wp.array2d(dtype=int),
            stats:wp.array2d(dtype=float),lengths:wp.array(dtype=int),index:wp.array(dtype=int),
            ncon:wp.array(dtype=int),nefc:wp.array(dtype=int),overflow:wp.array(dtype=int),
            nconmax:int,njmax:int,pan:int,cake:int):
    w=wp.tid()
    i=index[0]
    if ncon[0]>=nconmax or nefc[w]>=njmax:
        wp.atomic_max(overflow,0,1)
    if i<lengths[w] and stats[w,8]==0.0:
        dt=config[0]
        rotation=mat[w,pan]
        lever=pos[w,cake]-pos[w,pan]
        rel=wp.transpose(rotation)*lever-wp.vec3(params[w,0],0.0,0.0)
        normal=wp.dot(rotation*wp.vec3(0.0,0.0,1.0),mat[w,cake]*wp.vec3(0.0,0.0,1.0))
        pan_omega=wp.vec3(0.0,-v[w,0]-v[w,1],0.0)
        pan_v=wp.vec3(-params[w,5]*wp.sin(q[w,0])*v[w,0],0.0,params[w,5]*wp.cos(q[w,0])*v[w,0])
        cake_v=wp.vec3(v[w,2],v[w,3],v[w,4])
        cake_omega=mat[w,cake]*wp.vec3(v[w,5],v[w,6],v[w,7])
        linear=wp.length(cake_v-pan_v-wp.cross(pan_omega,lever))
        angular=wp.length(cake_omega-pan_omega)
        if flags[w,0]==0 and rel[2]>params[w,2]+0.003:
            stats[w,0]=stats[w,0]+dt
            stats[w,7]=stats[w,7]+angular*dt
            stats[w,1]=wp.max(stats[w,1],stats[w,0])
            if stats[w,0]>=config[1]:
                stats[w,2]=1.0
        else:
            stats[w,0]=0.0
        stats[w,5]=wp.max(stats[w,5],rel[2]-params[w,2])
        if stats[w,2]>0.0:
            stats[w,6]=wp.max(stats[w,6],(1.0-normal)/2.0)
        offset=wp.sqrt(rel[0]*rel[0]+rel[1]*rel[1])
        contained=offset+params[w,1]<=params[w,3]-config[2]
        stable=(stats[w,2]>0.0 and flags[w,1]!=0 and normal < -0.90 and contained
                and rel[2]>=-0.002 and rel[2]<=params[w,2]+0.008 and linear<0.06 and angular<0.6
                and stats[w,7]>=2.2 and stats[w,7]<=4.3)
        if stable:
            stats[w,3]=stats[w,3]+dt
        else:
            stats[w,3]=0.0
        tracking=wp.max(wp.abs(target[w,i,0]-q[w,0]),wp.abs(target[w,i,1]-q[w,1]))*57.2957795
        stats[w,4]=wp.max(stats[w,4],tracking)
        fault=int(0)
        if flags[w,2]!=0 or pos[w,cake][2]<0.005:
            fault=1
        finite=True
        for k in range(9):
            if not wp.isfinite(q[w,k]):
                finite=False
        for k in range(8):
            if not wp.isfinite(v[w,k]):
                finite=False
        if fault==0 and not finite:
            fault=2
        if fault==0 and (q[w,0]<config[5]-0.02 or q[w,0]>config[6]+0.02 or q[w,1]<config[7]-0.02 or q[w,1]>config[8]+0.02):
            fault=3
        if fault==0 and (wp.abs(v[w,0])*57.2957795>params[w,10]*1.15 or wp.abs(v[w,1])*57.2957795>params[w,11]*1.15):
            fault=4
        if fault==0 and tracking>config[4]:
            fault=5
        stats[w,8]=float(fault)
        stats[w,9]=offset
        stats[w,10]=normal
        stats[w,11]=float(flags[w,1])
        success=float(0.0)
        if stats[w,3]>=config[3] and fault==0:
            success=1.0
        stats[w,12]=success
        reward=0.2*wp.min(stats[w,5]/0.02,1.0)+stats[w,2]*(1.0+stats[w,6])+15.0*success-0.02*stats[w,4]
        if stats[w,2]>0.0 and contained and flags[w,1]!=0:
            reward=reward+wp.max(0.0,1.0-offset/params[w,4])
        if fault!=0:
            reward=-5.0
        stats[w,13]=reward


@wp.kernel
def tick(index:wp.array(dtype=int)):
    index[0]=index[0]+1


def _require_awake_models(models):
    """dof_length is shared in MJWarp 3.13 and used only by sleep.py.

    Require sleeping OFF, even if all dof_length values happen to match today;
    a later refreshed bank can contain different geometry. Never enable sleep
    after constructing this batch. A dependency change requires a new audit.
    """
    if importlib.metadata.version('mujoco-warp') != '3.13.0':
        raise RuntimeError('The shared dof_length exception requires audited MuJoCo Warp 3.13.0')
    for model in models:
        if int(model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_SLEEP):
            raise RuntimeError('Sleeping must be disabled for variable-geometry Warp batches: '
                               'dof_length is not a per-world field in MuJoCo Warp 3.13.0')


def _shared_field_issues(destination,sources,path=''):
    """Report every incompatible shared field before allocating CUDA arrays."""
    issues=[]
    for field in dataclasses.fields(destination):
        name=path+field.name
        values=[getattr(s,field.name) for s in sources]
        current=getattr(destination,field.name)
        if dataclasses.is_dataclass(current):
            issues.extend(_shared_field_issues(current,values,name+'.'))
        elif isinstance(current,wp.array):
            shape=getattr(field.type,'shape',())
            if shape and shape[0]=='*':
                continue
            # Audited in the pinned wheel: only sleep.wake/sleep.sleep consume
            # dof_length; both call sites are guarded by EnableBit.SLEEP.
            # _merge_models verifies that bit is OFF in destination AND sources.
            if name=='dof_length':
                continue
            arrays=[current.numpy()]+[v.numpy() for v in values]
            if any(not np.array_equal(arrays[0],a) for a in arrays[1:]):
                issues.append(name)
        elif isinstance(current,(int,float,bool)) and any(v!=current for v in values):
            issues.append(name)
    return issues


def _copy_batched_fields(destination,sources):
    for field in dataclasses.fields(destination):
        values=[getattr(s,field.name) for s in sources]
        current=getattr(destination,field.name)
        if dataclasses.is_dataclass(current):
            _copy_batched_fields(current,values)
        elif isinstance(current,wp.array):
            shape=getattr(field.type,'shape',())
            if shape and shape[0]=='*':
                data=np.concatenate([v.numpy() for v in values],axis=0)
                replacement=wp.array(data,dtype=current.dtype,device='cuda:0')
                replacement._is_batched=True  # pinned MJWarp metadata
                setattr(destination,field.name,replacement)


def _merge_models(destination,sources):
    """Copy per-world compiled physics; permit only audited inactive metadata.

    Every shared numeric field must match, except dof_length when sleeping is
    disabled on ALL models. It retains the first model's value, which is unused
    in that mode. No masses, inertias, motor limits or geometry are overwritten
    with the first world's values.
    """
    _require_awake_models([destination,*sources])
    issues=_shared_field_issues(destination,sources)
    if issues:
        raise RuntimeError('Unbatchable model fields differ: '+', '.join(issues))
    _copy_batched_fields(destination,sources)


class WarpBatch:
    def __init__(self,cfg,environments):
        self.cfg=cfg
        self.envs=environments
        self.n=len(environments)
        self.device='cuda:0'
        if not wp.get_device(self.device).is_cuda:
            raise RuntimeError('CUDA is required for the Warp backend')
        self.cases=[e.case.copy() for e in environments]
        self.obs=np.stack([e.obs for e in environments])
        first=environments[0]
        if (first.model.nq,first.model.nv,first.model.nu)!=(9,8,2):
            raise ValueError('Warp kernels require the two-hinge + free-pancake topology')
        with wp.ScopedDevice('cpu'):
            host_models=[mjw.put_model(e.model) for e in environments]
        with wp.ScopedDevice(self.device):
            self.m=mjw.put_model(first.model)
            _merge_models(self.m,host_models)
            self.d=mjw.make_data(first.model,nworld=self.n,nconmax=128,njmax=512)
            self.q0=wp.array(np.stack([e.data.qpos for e in environments]),dtype=float)
            self.v0=wp.array(np.stack([e.data.qvel for e in environments]),dtype=float)
            self.params=wp.array(np.array([[e.r,e.radius,e.half_thickness,e.inner_R,e.R,
                 e.case['elbow_wrist_mm']/1000,*e.caps,*e.cutoff,*e.vmax,*e.preload] for e in environments]),dtype=float)
            p,c=cfg['physics'],cfg['control']
            self.gains=wp.array(c['kp_nm_rad']+c['kd_nm_s_rad'],dtype=float)
            self.config=wp.array([p['timestep_s'],p['min_airtime_s'],p['edge_margin_mm']/1000,
                p['success_settling_s'],p['tracking_fault_deg'],*np.radians(c['shoulder_limits_deg']),
                *np.radians(c['wrist_limits_deg'])],dtype=float)
            self.max_steps=int(np.ceil(p['max_episode_s']/p['timestep_s']))+2
            self.target=wp.zeros((self.n,self.max_steps,2),dtype=float)
            self.velocity=wp.zeros_like(self.target)
            self.stats=wp.zeros((self.n,14),dtype=float)
            self.flags=wp.zeros((self.n,3),dtype=int)
            self.index=wp.zeros(1,dtype=int)
            self.overflow=wp.zeros(1,dtype=int)
            self.lengths=wp.zeros(self.n,dtype=int)
        self.ids=(first.pan,first.cake,first.cake_geom,first.pan_floor,
                  first.model.geom('rim_0').id,first.model.geom(f"rim_{p['rim_segments']-1}").id)
        self.graph=None
        self.reset()

    def reset(self):
        wp.copy(self.d.qpos,self.q0)
        wp.copy(self.d.qvel,self.v0)
        for a in (self.d.qacc_warmstart,self.d.ctrl,self.d.time,self.stats,self.flags,self.index,self.overflow):
            a.zero_()
        with wp.ScopedDevice(self.device):
            mjw.forward(self.m,self.d)

    def _step(self):
        pan,cake,cg,floor,r0,r1=self.ids
        wp.launch(control,self.n,[self.d.qpos,self.d.qvel,self.d.ctrl,self.d.qacc_warmstart,self.q0,self.v0,
                  self.target,self.velocity,self.params,self.gains,self.stats,self.lengths,self.index,self.flags],device=self.device)
        mjw.step(self.m,self.d)
        mjw.forward(self.m,self.d)
        wp.launch(contacts,self.d.naconmax,[self.d.contact.geom,self.d.contact.worldid,self.d.nacon,
                                         cg,floor,r0,r1,self.flags],device=self.device)
        wp.launch(measure,self.n,[self.d.qpos,self.d.qvel,self.d.xpos,self.d.xmat,self.target,self.params,
                  self.config,self.flags,self.stats,self.lengths,self.index,self.d.nacon,self.d.nefc,self.overflow,
                  self.d.naconmax,self.d.njmax,pan,cake],device=self.device)
        wp.launch(tick,1,[self.index],device=self.device)

    def run(self,actions):
        p=self.cfg['physics']
        targets=np.zeros((self.n,self.max_steps,2),np.float32)
        velocities=np.zeros_like(targets)
        lengths=np.zeros(self.n,np.int32)
        metadata=[]
        for w,(action,case) in enumerate(zip(actions,self.cases)):
            try:
                phases,parameters=build_trajectory(action,case,self.cfg)
                duration=sum(x['duration'] for x in phases)
                ts=np.arange(0,duration+p['settle_after_s']+p['timestep_s'],p['timestep_s'])
                q=np.radians(sample_many(phases,ts-case['actuator_delay_s']))
                lengths[w]=len(q)
                if len(q)>self.max_steps:
                    raise ValueError('trajectory_exceeds_episode_duration')
                targets[w,:len(q)]=q
                velocities[w,:len(q)]=np.gradient(q,p['timestep_s'],axis=0)
                metadata.append({'parameters':parameters,'duration_s':duration})
            except ValueError as exc:
                lengths[w]=0
                metadata.append({'invalid_action':str(exc)})
        self.target.assign(targets)
        self.velocity.assign(velocities)
        self.lengths.assign(lengths)
        self.reset()
        with wp.ScopedDevice(self.device):
            if self.graph is None:
                self._step()  # compile kernels before CUDA graph capture
                wp.synchronize()
                self.reset()
                with wp.ScopedCapture() as capture:
                    for _ in range(16):
                        self._step()
                self.graph=capture.graph
                self.reset()
            for _ in range((int(lengths.max())+15)//16):
                wp.capture_launch(self.graph)
        rows=self.stats.numpy()
        if self.overflow.numpy()[0]:
            raise RuntimeError('GPU contact/constraint capacity exhausted; batch rejected, no PPO update')
        reasons={0:None,1:'ground_or_other_contact',2:'nonfinite_state',3:'actual_joint_limits',
                 4:'actual_speed_limit',5:'tracking_fault'}
        infos=[]
        rewards=rows[:,13].copy()
        for w,s in enumerate(rows):
            if 'invalid_action' in metadata[w]:
                rewards[w]=-5.
            infos.append({'case':self.cases[w],'rejected':False,'success':bool(s[12]),'airborne':bool(s[2]),
                'unsafe':bool(s[8]),'reason':reasons[int(s[8])],'airtime_s':float(s[1]),
                'flight_rotation_rad':float(s[7]),'peak_height_m':float(s[5]),'normal_dot':float(s[10]),
                'landing_offset_m':float(s[9]),'settled_s':float(s[3]),'max_tracking_error_deg':float(s[4]),
                **metadata[w]})
        return rewards,infos


class WarpVecEnv(VecEnv):
    """SB3 auto-reset adapter. Each batch is a complete independent toss."""
    def __init__(self,cfg,profile,count,seed,output=None):
        self.cfg,self.profile,self.count,self.bank_seed=cfg,profile,count,seed
        self.output=output
        self.generation=0
        super().__init__(count,spaces.Box(-2.,2.,(len(FEATURE_SCALES),),np.float32),
                         spaces.Box(-1.,1.,(7,),np.float32))
        self.refresh_bank()

    def refresh_bank(self):
        started=time.monotonic()
        source=TrainableTossEnv(self.cfg,self.profile)
        envs=[]
        rejections={}
        for i in range(self.count):
            source.reset(seed=self.bank_seed+self.generation if i==0 else None)
            # Preserve the compiled model/data for this case; reset creates new ones.
            import copy
            envs.append(copy.copy(source))
        rejections=dict(source.sampler.counts)
        self.batch=WarpBatch(self.cfg,envs)
        self.generation+=1
        report={'event':'gpu_bank_ready','generation':self.generation,'worlds':self.count,
                'setup_seconds':time.monotonic()-started,'screening':rejections,
                'seed':self.bank_seed+self.generation-1,
                'distribution':'finite feasible bank refreshed between validation blocks',
                'sampled_ranges':{k:[min(c[k] for c in self.batch.cases),max(c[k] for c in self.batch.cases)]
                    for k in ('elbow_wrist_mm','wrist_pan_center_mm','pan_mass_kg','pancake_mass_g','shoulder_torque_nm')}}
        if self.output:
            from .training import write_json
            write_json(self.output/f'bank_{self.generation:04d}.json',report)
        print(json.dumps(report),flush=True)

    def reset(self):
        return self.batch.obs.copy()

    def step_async(self,actions):
        self.actions=actions

    def step_wait(self):
        rewards,infos=self.batch.run(self.actions)
        for i,info in enumerate(infos):
            info['terminal_observation']=self.batch.obs[i].copy()
            info['episode']={'r':float(rewards[i]),'l':1,'t':0.}
        return self.batch.obs.copy(),rewards,np.ones(self.count,bool),infos

    def close(self):
        self.batch=None

    def get_attr(self,attr_name,indices=None):
        # SB3 probes render_mode during VecEnv initialization.
        value=None if attr_name=='render_mode' else getattr(self,attr_name)
        return [value for _ in self._get_indices(indices)]

    def set_attr(self,attr_name,value,indices=None):
        raise NotImplementedError('GPU world attributes are changed through a new case bank')

    def env_method(self,method_name,*args,indices=None,**kwargs):
        raise NotImplementedError(method_name)

    def env_is_wrapped(self,wrapper_class,indices=None):
        return [False for _ in self._get_indices(indices)]
