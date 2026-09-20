"""Pure geometry and planning; importing this module cannot move hardware."""
from dataclasses import dataclass
import math
import numpy as np


def rotation(angle):
    q=math.radians(angle);c,s=math.cos(q),math.sin(q)
    return np.array([[c,0,-s],[0,1,0],[s,0,c]])


def pan_pose(angle,cfg):
    r=rotation(angle)
    return np.asarray(cfg['joint_world_m'])+r@np.asarray(cfg['pan_center_from_joint_m']),r@np.array([0.,0.,1.])


def ray_plane(pixel,K,Rcw,tcw,plane_y,min_sine=.08):
    """Pixel must already be undistorted. World motion plane is y=plane_y."""
    camera=-Rcw.T@tcw
    ray=Rcw.T@np.linalg.solve(K,np.array([*pixel,1.]))
    ray/=np.linalg.norm(ray)
    if abs(camera[1]-plane_y)<.03 or abs(ray[1])<min_sine:
        raise ValueError('Camera view cannot resolve depth in the motion plane; offset camera sideways')
    distance=(plane_y-camera[1])/ray[1]
    if distance<=0 or distance>3:raise ValueError('Ray-plane intersection behind camera or too distant')
    return camera+distance*ray


def move_time(distance,speed,accel):
    distance=abs(distance)
    if speed<=0 or accel<=0:raise ValueError('Speed/acceleration must be positive')
    return 2*math.sqrt(distance/accel) if distance<=speed*speed/accel else distance/speed+speed/accel


def speed_for_time(distance,seconds,vmax,accel):
    distance=abs(distance)
    if seconds<=0:raise ValueError('Move time must be positive')
    minimum=move_time(distance,vmax,accel)
    if seconds+1e-8<minimum:raise ValueError(f'Move needs at least {minimum:.3f}s, requested {seconds:.3f}s')
    return 2*distance/(seconds+math.sqrt(max(0.,seconds*seconds-4*distance/accel))) if distance else 0.


@dataclass
class Flight:
    t0: float
    position: np.ndarray
    velocity: np.ndarray
    residual: float

    def at(self,t):
        dt=t-self.t0
        return self.position+self.velocity*dt+np.array([0.,0.,-4.905*dt*dt])

    def vel(self,t):return self.velocity+np.array([0.,0.,-9.81*(t-self.t0)])


def fit_flight(samples,cfg):
    if len(samples)<cfg['min_samples']:raise ValueError('Need more airborne observations')
    samples=samples[-cfg['max_samples']:]
    ts=np.array([s[0] for s in samples]);points=np.array([s[1] for s in samples])
    if not np.isfinite(points).all() or np.any(np.diff(ts)<=0):raise ValueError('Invalid flight samples')
    if ts[-1]-ts[0]<cfg['min_span_s']:raise ValueError('Flight history too short')
    dt=ts-ts[-1];A=np.column_stack([np.ones(len(dt)),dt])
    adjusted=points.copy();adjusted[:,2]+=4.905*dt*dt
    coef=np.linalg.lstsq(A,adjusted,rcond=None)[0]
    residual=float(np.sqrt(np.mean(np.sum((A@coef-adjusted)**2,axis=1))))
    if residual>cfg['max_residual_m']:raise ValueError('Flight does not fit ballistic motion')
    if np.linalg.norm(coef[1])>cfg['max_speed_m_s']:raise ValueError('Implausible flight speed')
    return Flight(float(ts[-1]),coef[0],coef[1],residual)


def intercept(flight,now,angle,cfg,vmax,accel):
    """Choose a stationary pan pose reachable before descending contact.

    Assumes planar ballistic flight and no sideways drift. This is not 6D pose or
    a deformable-pancake collision model. Conservative footprint includes radius.
    """
    c=cfg['catch'];g=cfg['geometry'];best=None
    clearance=g['pan_inner_radius_m']-g['pancake_radius_m']-c['edge_margin_m']
    if clearance<=0:return None
    candidates=np.arange(c['angle_min_deg'],c['angle_max_deg']+.0001,c['angle_step_deg'])
    for target in candidates:
        if abs(target)>c['max_pan_tilt_deg']:continue
        travel=move_time(target-angle,vmax,accel)
        center,normal=pan_pose(target,g)
        # Solve first descending intersection with the *stationary* pan plane.
        p=flight.at(now)-center;v=flight.vel(now)
        aa=-4.905*normal[2];bb=float(v@normal);cc=float(p@normal)-g['pancake_half_thickness_m']
        roots=np.roots([aa,bb,cc]) if abs(aa)>1e-8 else ([-cc/bb] if abs(bb)>1e-8 else [])
        for root in roots:
            if abs(np.imag(root))>1e-8:continue
            dt=float(np.real(root))
            if not c['min_lead_s']<=dt<=c['horizon_s']:continue
            if travel+c['command_margin_s']>dt:continue
            hit=flight.at(now+dt)
            if float(flight.vel(now+dt)@normal)>=-c['min_descent_m_s']:continue
            delta=hit-center
            offset=float(np.linalg.norm(delta-(delta@normal)*normal))
            if offset>clearance:continue
            cost=offset+.01*abs(target-angle)/max(1.,c['angle_max_deg']-c['angle_min_deg'])
            candidate={'angle_deg':float(target),'impact_time':now+dt,'offset_m':offset,
                       'move_s':travel,'residual_m':flight.residual,'predicted_world_m':hit.tolist(),'cost':cost}
            if best is None or cost<best['cost']:best=candidate
    return best


def in_circle(pixel,center,radius):
    return center is not None and np.linalg.norm(np.asarray(pixel)-center)<=radius
