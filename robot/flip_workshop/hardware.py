"""Single-owner serial worker. No other thread writes to the controller."""
import copy
import json
import math
from pathlib import Path
import queue
import threading
import time
import numpy as np
from control_112 import SerialLink,Motor112,ControllerError
from core import move_time,speed_for_time,fit_flight,intercept,in_circle


class Shared:
    def __init__(self):
        self.lock=threading.Lock();self.observations=[];self.state='WATCH';self.busy=False
        self.trial=None;self.reference=False;self.message='';self.heartbeat=time.monotonic()
    def observe(self,obs):
        with self.lock:
            self.observations.append(obs);self.observations=self.observations[-300:]
    def samples(self):
        with self.lock:return list(self.observations)


class Recorder:
    def __init__(self,folder):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=False)
        self.file=(self.folder/'events.jsonl').open('w',buffering=1);self.lock=threading.Lock()
    def event(self,event,**kwargs):
        payload={'t':time.monotonic(),'event':event,**kwargs}
        with self.lock:self.file.write(json.dumps(payload,allow_nan=False)+'\n')
    def close(self):self.file.close()


def validate(cfg):
    def finite_tree(v):
        if isinstance(v,dict):
            for x in v.values():finite_tree(x)
        elif isinstance(v,list):
            for x in v:finite_tree(x)
        elif isinstance(v,(float,int)) and not math.isfinite(v):raise ValueError('Config contains nonfinite numbers')
    finite_tree(cfg)
    lim=cfg['limits'];c=cfg['catch'];tr=cfg['tracking']
    if not lim['angle_min_deg']<0<lim['angle_max_deg']:raise ValueError('Angle bounds must contain neutral')
    if min(lim['speed_deg_s'],lim['acceleration_deg_s2'],cfg['a_units_per_degree'])<=0:raise ValueError('Limits and calibration must be positive')
    for phase in ('slide','launch','return'):
        if not lim['angle_min_deg']<=cfg[phase]['angle_deg']<=lim['angle_max_deg']:raise ValueError('Phase angle outside configured bounds')
        if cfg[phase]['move_s']<=0:raise ValueError('Phase time must be positive')
    if cfg['return']['angle_deg']!=0:raise ValueError('Return must be neutral zero for repeatable trials')
    if not lim['angle_min_deg']<=c['angle_min_deg']<c['angle_max_deg']<=lim['angle_max_deg']:raise ValueError('Catch search outside bounds')
    if not lim['angle_min_deg']<=c['fallback_angle_deg']<=lim['angle_max_deg']:raise ValueError('Fallback outside bounds')
    for block,keys in [(c,['angle_step_deg','horizon_s','min_lead_s','command_margin_s','fallback_move_s','settle_s','observe_timeout_s']),
                       (tr,['max_age_s','min_span_s','max_residual_m','max_speed_m_s','airborne_height_m']),
                       (cfg['slide'],['min_hold_s','timeout_s','stable_s'])]:
        if any(block[k]<=0 for k in keys):raise ValueError('Times and tracking thresholds must be positive')
    if tr['min_samples']<3 or tr['max_samples']<tr['min_samples']:raise ValueError('Invalid sample counts')
    if tr['webcam_latency_s'] is not None and tr['webcam_latency_s']<0:raise ValueError('Latency cannot be negative')
    if c['angle_step_deg']<.1:raise ValueError('Catch search step must be at least 0.1 degree')
    if not 0<c['max_pan_tilt_deg']<90:raise ValueError('Pan tilt must be below 90 degrees')
    g=cfg['geometry']
    for k in ('joint_world_m','pan_center_from_joint_m'):
        if g[k] is not None and (not isinstance(g[k],list) or len(g[k])!=3):raise ValueError(f'{k} requires three metres coordinates')
    for k in ('pan_outer_radius_m','pancake_radius_m','pancake_half_thickness_m','camera_to_pan_center_m'):
        if g[k]<=0:raise ValueError(f'{k} must be positive')
    if g['pan_inner_radius_m'] is not None and not 0<g['pan_inner_radius_m']<=g['pan_outer_radius_m']:
        raise ValueError('Usable pan radius must fit inside outer radius')
    if not 0<=cfg['vision']['min_score']<=1:raise ValueError('Detection score threshold must be 0..1')
    for k in ('edge_target_px','pan_center_px'):
        v=cfg['vision'][k]
        if v is not None and (not isinstance(v,list) or len(v)!=2):raise ValueError('Image targets require x,y pixels')
    if min(cfg['vision']['edge_radius_px'],cfg['vision']['ready_radius_px'])<=0:raise ValueError('Target radii must be positive')
    if c['edge_margin_m']<0:raise ValueError('Catch edge margin must be nonnegative')
    if not .1<=cfg.get('tempo',1)<=10:raise ValueError('Tempo must be 0.1 to 10')
    slide=cfg['slide'];amp=slide.get('wiggle_deg',0);cycles=slide.get('wiggle_cycles',0)
    if amp<0 or cycles!=int(cycles) or not 0<=cycles<=10:raise ValueError('Wiggle: nonnegative amplitude and 0..10 whole cycles')
    if amp and cycles:
        if slide.get('wiggle_move_s',0)<=0:raise ValueError('Wiggle time must be positive')
        if not lim['angle_min_deg']<=slide['angle_deg']-amp<=slide['angle_deg']+amp<=lim['angle_max_deg']:
            raise ValueError('Slide plus wiggle exceeds angle bounds; reduce wiggle or slide angle')
    if not 0<cfg.get('zero_nudge_deg',1)<=5:raise ValueError('Zero nudge must be 0..5 degrees')
    if not lim['angle_min_deg']<=cfg.get('start_angle_deg',0)<=lim['angle_max_deg']:
        raise ValueError('Start angle outside joint bounds')
    return cfg


def sequence_targets(cfg):
    if cfg.get('six_part',False):
        return [cfg['catch']['fallback_angle_deg'],cfg['launch']['angle_deg'],
                cfg['slide']['angle_deg'],cfg['launch']['angle_deg'],
                cfg['catch']['fallback_angle_deg'],0.]
    if cfg.get('four_part',False):
        return [cfg['launch']['angle_deg'],cfg['slide']['angle_deg'],
                cfg['launch']['angle_deg'],cfg['catch']['fallback_angle_deg']]
    return [cfg['slide']['angle_deg'],cfg['launch']['angle_deg'],cfg['catch']['fallback_angle_deg']]


def effective_time(delta,requested,cfg,vmax,accel):
    return max(requested/cfg.get('tempo',1),move_time(delta,vmax,accel)+1e-6)


def wiggle_targets(cfg):
    p=cfg['slide'];q=p['angle_deg'];a=p.get('wiggle_deg',0)
    return [v for _ in range(int(p.get('wiggle_cycles',0))) for v in (q+a,q-a,q)] if a else []


class Worker:
    def __init__(self,port,cfg,shared,recorder):
        self.shared=shared;self.log=recorder;self.queue=queue.Queue(maxsize=1)
        self.stop=threading.Event();self.quit=threading.Event();self.port=port;self.initial=cfg
        self.motor=None;self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def submit(self,action,cfg):
        with self.shared.lock:
            if self.shared.busy or self.quit.is_set():return False
            self.shared.busy=True
        self.stop.clear();self.queue.put((action,copy.deepcopy(cfg)));return True
    def check_stop(self):
        if self.stop.is_set() or self.quit.is_set():raise ControllerError('Operator stop')
        if time.monotonic()-self.shared.heartbeat>.5:raise ControllerError('Vision/UI heartbeat lost')
    def status(self):
        self.check_stop();s=self.motor.link.status()
        if s.state not in ('Idle','Jog'):raise ControllerError(f'Unexpected controller state: {s.state}')
        self.log.event('motor',state=s.state,commanded_angle_deg=self.motor.angle(s))
        return s
    def limits(self,cfg):
        self.motor.settings=self.motor.firmware_settings()
        actual_speed=self.motor.settings[113]/(60*self.motor.a_scale)
        actual_accel=self.motor.settings[123]/self.motor.a_scale
        if actual_accel>cfg['limits']['acceleration_deg_s2']*1.001:
            required=cfg['limits']['acceleration_deg_s2']*self.motor.a_scale
            # Apply a downward-only change at Idle instead of making the user
            # switch apps. This persists on the controller; never raise it here.
            self.check_stop();self.motor._idle()
            value=math.floor(required*1e6)/1e6
            if value<=0:raise ValueError('Acceleration too small to encode')
            self.motor.link.command(f'$123={value:.6f}',timeout=.5)
            self.motor.settings=self.motor.firmware_settings()
            actual_accel=self.motor.settings[123]/self.motor.a_scale
            if actual_accel<=0 or actual_accel>cfg['limits']['acceleration_deg_s2']*1.001:
                raise ControllerError('Acceleration setting readback failed; no movement sent')
            self.log.event('acceleration_lowered',firmware_123=value,physical_deg_s2=actual_accel)
            self.shared.message=f'Acceleration set to {actual_accel:.0f} deg/s²; speed settings unchanged'
            print(self.shared.message,flush=True)
        return min(actual_speed,cfg['limits']['speed_deg_s']),actual_accel
    def move(self,target,duration,cfg,vmax,accel,exact=False):
        self.check_stop()
        if not cfg['limits']['angle_min_deg']<=target<=cfg['limits']['angle_max_deg']:raise ValueError('Target outside joint bounds')
        s=self.status()
        if s.state!='Idle':raise ControllerError('Previous motion not complete')
        angle=self.motor.angle(s)
        requested=duration
        if not exact:duration=effective_time(target-angle,duration,cfg,vmax,accel)
        v=speed_for_time(target-angle,duration,vmax,accel)
        if not exact and duration>requested/cfg.get('tempo',1)+.002:
            self.log.event('timing_limited',target=target,requested_s=requested/cfg.get('tempo',1),effective_s=duration)
            self.shared.message=f'Motor-limited move: {duration:.3f}s; requested timing exceeds current acceleration'
        if not v:self.hold(duration);return
        units=(target-angle)*self.motor.a_scale;feed=v*self.motor.a_scale*60
        self.log.event('move',target=target,duration=duration,peak_deg_s=v)
        self.motor.link.command(f'$J=G21 G91 A{units:.9f} F{feed:.9f}',timeout=.5)
        end=time.monotonic()+duration+1
        while True:
            s=self.status()
            if s.state=='Idle':
                if abs(self.motor.angle(s)-target)>.5:raise ControllerError('Commanded endpoint mismatch')
                return
            if time.monotonic()>end:raise ControllerError('Motion timeout')
            time.sleep(.01)
    def launch_catch(self,cfg,vmax,accel):
        """Queue the two phases without a host Idle barrier between them.

        Firmware still owns acceleration, braking and reversal stops. Individual
        phase times are planning estimates, not guaranteed arrival timestamps.
        """
        start=self.status()
        if start.state!='Idle':raise ControllerError('Launch requires Idle')
        previous=self.motor.angle(start);plans=[];estimate=0.
        for label,target,requested in (
                ('launch',cfg['launch']['angle_deg'],cfg['launch']['move_s']),
                ('catch',cfg['catch']['fallback_angle_deg'],cfg['catch']['fallback_move_s'])):
            if not cfg['limits']['angle_min_deg']<=target<=cfg['limits']['angle_max_deg']:
                raise ValueError('Launch/catch target outside joint bounds')
            delta=target-previous;duration=effective_time(delta,requested,cfg,vmax,accel)
            speed=speed_for_time(delta,duration,vmax,accel)
            if abs(delta*self.motor.a_scale)>1e-9:
                feed=math.floor(speed*self.motor.a_scale*60*1e9)/1e9
                if feed<=0:raise ValueError('Feed too small')
                command=f'$J=G21 G91 A{delta*self.motor.a_scale:.9f} F{feed:.9f}'
                plans.append((command,label,target,speed));estimate+=duration
            previous=target
        # Full preflight before dispatch; catch is not sent after an uncertain ACK.
        began=time.monotonic()
        try:
            for command,label,target,speed in plans:
                self.check_stop()
                self.motor.link.command(command,timeout=.5)
                self.log.event('queued_phase',phase=label,target=target,peak_deg_s=speed)
            deadline=began+estimate+1
            while True:
                status=self.status()
                if status.state=='Idle':
                    if abs(self.motor.angle(status)-previous)>.5:raise ControllerError('Queued catch endpoint mismatch')
                    break
                if time.monotonic()>deadline:raise ControllerError('Queued launch/catch timed out')
                time.sleep(.005)
        except BaseException:
            self.shared.reference=False;self.motor.reference=None
            try:self.motor.link.realtime(b'\x85')
            except Exception:pass
            raise
        self.log.event('queued_launch_catch_complete',rest_to_rest_estimate_s=estimate,
                       observed_s=time.monotonic()-began)

    def hold(self,seconds):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            if self.status().state!='Idle':raise ControllerError('Unexpected motion during hold')
            time.sleep(.01)
    def gate(self,cfg,center,radius,duration,timeout):
        if center is None:raise ValueError('Click a pan center and edge target before trials')
        deadline=time.monotonic()+timeout;stable=None;lastseq=None
        while time.monotonic()<deadline:
            self.check_stop();samples=self.shared.samples();obs=samples[-1] if samples else None
            fresh=obs and time.monotonic()-obs['received']<=cfg['tracking']['max_age_s']
            if fresh and obs['seq']!=lastseq:
                lastseq=obs['seq'];pixel=obs.get('pixel')
                good=pixel is not None and obs['score']>=cfg['vision']['min_score'] and in_circle(pixel,center,radius)
                stable=(stable if stable is not None else obs['received']) if good else None
                if stable is not None and obs['received']-stable>=duration:return
            elif not fresh:stable=None
            if self.status().state!='Idle':raise ControllerError('Unexpected motion during positioning')
            time.sleep(.01)
        raise ValueError('Pancake did not stabilize inside target; trial stopped before launch')
    def preflight(self,cfg,vmax,accel,mode="timed"):
        validate(cfg)
        if min(vmax,accel)<=0:raise ValueError('Invalid motor limits')
        # Timings are bounded by actual firmware ramps; never raise acceleration.
        for target in wiggle_targets(cfg):
            if not cfg['limits']['angle_min_deg']<=target<=cfg['limits']['angle_max_deg']:
                raise ValueError('Wiggle outside joint limits')
    def three_part(self,cfg,vmax,accel):
        """Queue the configured sequence at maximum allowed feed, with no dwell."""
        targets=sequence_targets(cfg)
        previous=cfg.get('start_angle_deg',0.);plans=[];estimate=0.
        feed=math.floor(vmax*self.motor.a_scale*60*1e9)/1e9
        if feed<=0:raise ValueError('Invalid maximum feed')
        for target in targets:
            if not cfg['limits']['angle_min_deg']<=target<=cfg['limits']['angle_max_deg']:
                raise ValueError('Three-part target outside configured bounds')
            delta=target-previous
            if delta==0:raise ValueError('Each sequence segment must move')
            plans.append(f'$J=G21 G91 A{delta*self.motor.a_scale:.9f} F{feed:.9f}')
            estimate+=move_time(delta,vmax,accel);previous=target
        trial=str(time.time_ns());self.shared.trial=trial;self.shared.state=f'{len(targets)}-PART SNAP'
        self.log.event('trial_start',trial=trial,mode=f'{len(targets)}_part',config=cfg,
                       firmware_limits={'speed':vmax,'acceleration':accel})
        self.log.event('sequence_plan',targets=targets,max_feed_deg_s=vmax,
                       ramp_estimate_s=estimate,programmed_dwell_s=0)
        self.shared.message=f'{len(targets)} queued moves; zero dwell; estimated ramps {estimate:.3f}s'
        started=time.monotonic()
        try:
            for part,command in enumerate(plans,1):
                self.check_stop();self.motor.link.command(command,timeout=.5)
                self.log.event('queued_phase',phase=part,target=targets[part-1],peak_request_deg_s=vmax)
            deadline=started+estimate+1
            while True:
                status=self.status()
                if status.state=='Idle':
                    if abs(self.motor.angle(status)-targets[-1])>.5:raise ControllerError('Three-part final position mismatch')
                    break
                if time.monotonic()>deadline:raise ControllerError('Three-part motion timed out')
                time.sleep(.005)
        except BaseException:
            self.shared.reference=False;self.motor.reference=None
            try:self.motor.link.realtime(b'\x85')
            except Exception:pass
            raise
        self.log.event('trial_complete',trial=trial,catch_mode=f'{len(targets)}_part',success=None,
                       final_angle_deg=targets[-1],observed_s=time.monotonic()-started)
        self.shared.message=f'Finished at {targets[-1]:g} deg. ' + ('F repeats from this start position. ' if targets[-1]==cfg.get('start_angle_deg',0) else 'P moves to the start for next F. ') + 'Rate 0/1/2.'

    def trial(self,mode,cfg):
        if not self.shared.reference:raise ValueError('Press Z at physical neutral first')
        vmax,accel=self.limits(cfg)
        expected_start=cfg.get('start_angle_deg',0.) if mode=='timed' and cfg.get('three_part',False) else 0.
        if abs(self.motor.angle(self.motor._idle())-expected_start)>.5:
            raise ValueError(f'Start must be {expected_start:g} deg; use P for three-part start or N for neutral')
        self.preflight(cfg,vmax,accel,mode)
        if mode=='timed' and cfg.get('three_part',False):
            self.three_part(cfg,vmax,accel);return
        if mode=='vision':
            if cfg['catch']['angle_min_deg']<cfg['launch']['angle_deg']:
                raise ValueError('For vision mode, catch search must be at or above launch angle')
            if not cfg['calibration']['verified']:raise ValueError('Verify camera/geometry calibration before vision trials')
            g=cfg['geometry']
            if g['pan_inner_radius_m'] is None:raise ValueError('Measure usable inner pan radius; 12 inches is outer diameter')
            if not g['pancake_radius_m']+cfg['catch']['edge_margin_m']<g['pan_inner_radius_m']<=g['pan_outer_radius_m']:
                raise ValueError('Invalid usable pan/pancake radii')
            samples=self.shared.samples();last=samples[-1] if samples else {}
            if not last.get('world_m') or last.get('exposed') is None or time.monotonic()-last['exposed']>cfg['tracking']['max_age_s']:
                raise ValueError('Need a fresh calibrated world observation and camera latency')
            if abs(last['angle_deg'])>3:raise ValueError('Marker angle and neutral reference disagree')
        trial=f'{time.time_ns()}';self.shared.trial=trial
        self.log.event('trial_start',trial=trial,mode=mode,config=cfg,firmware_limits={'speed':vmax,'acceleration':accel})
        if mode=='vision':
            self.shared.state='READY CHECK'
            self.gate(cfg,cfg['vision']['pan_center_px'],cfg['vision']['ready_radius_px'],.15,2.)
        self.shared.state='SLIDE'
        self.move(cfg['slide']['angle_deg'],cfg['slide']['move_s'],cfg,vmax,accel)
        targets=wiggle_targets(cfg)
        if targets:
            self.shared.state='SLIDE WIGGLE'
            for target in targets:self.move(target,cfg['slide']['wiggle_move_s'],cfg,vmax,accel)
        self.hold(cfg['slide']['min_hold_s'])
        if mode=='vision':
            self.gate(cfg,cfg['vision']['edge_target_px'],cfg['vision']['edge_radius_px'],cfg['slide']['stable_s'],cfg['slide']['timeout_s'])
        if mode=='slide':
            self.log.event('slide_complete',trial=trial);return
        self.shared.state='LAUNCH';launch_start=time.monotonic()
        self.log.event('launch_start',trial=trial)
        queued=mode=='timed' and cfg.get('queued_launch_catch',True)
        if queued:
            self.shared.state='LAUNCH + CATCH (QUEUED)'
            self.launch_catch(cfg,vmax,accel)
        else:self.move(cfg['launch']['angle_deg'],cfg['launch']['move_s'],cfg,vmax,accel)
        selected=None;c=cfg['catch'];reason='timed trial'
        if mode=='vision':
            self.shared.state='PREDICT CATCH';deadline=time.monotonic()+c['observe_timeout_s']
            while time.monotonic()<deadline:
                self.check_stop();samples=self.shared.samples();now=time.monotonic()
                # Only consecutive, fresh, post-launch airborne detections; never bridge a miss.
                flight_samples=[]
                for o in reversed(samples):
                    if o.get('exposed') is None or o['exposed']<launch_start:break
                    if not o.get('world_m') or o.get('height_m',0)<cfg['tracking']['airborne_height_m'] or o['score']<cfg['vision']['min_score']:break
                    flight_samples.append((o['exposed'],o['world_m']))
                    if len(flight_samples)>=cfg['tracking']['max_samples']:break
                flight_samples.reverse()
                try:
                    if not flight_samples or now-flight_samples[-1][0]>cfg['tracking']['max_age_s']:raise ValueError('Flight data stale/missing')
                    if abs(samples[-1]['angle_deg']-cfg['launch']['angle_deg'])>3:
                        raise ValueError('Camera angle and commanded launch endpoint disagree')
                    flight=fit_flight(flight_samples,cfg['tracking'])
                    selected=intercept(flight,now,cfg['launch']['angle_deg'],cfg,vmax,accel)
                    if selected:break
                    reason='No reachable descending intercept'
                except ValueError as exc:reason=str(exc)
                self.status();time.sleep(.01)
        self.shared.state='CATCH'
        if selected:
            self.log.event('vision_intercept',trial=trial,**selected)
            # Recheck deadline just before sending; no catch queue/chasing during motion.
            duration=max(.001,selected['move_s'])
            if time.monotonic()+duration+c['command_margin_s']>selected['impact_time']:
                selected=None;reason='Prediction expired before dispatch'
        if queued:
            self.hold(c['settle_s'])
        elif selected:
            self.move(selected['angle_deg'],duration,cfg,vmax,accel,exact=True)
            self.hold(max(0,selected['impact_time']-time.monotonic())+c['settle_s'])
        else:
            self.log.event('fallback_catch',trial=trial,reason=reason)
            self.shared.message=f'Timed fallback: {reason}' if mode=='vision' else 'Timed catch'
            self.move(c['fallback_angle_deg'],c['fallback_move_s'],cfg,vmax,accel)
            self.hold(c['settle_s'])
        self.shared.state='RETURN'
        self.move(0,cfg['return']['move_s'],cfg,vmax,accel)
        self.log.event('trial_complete',trial=trial,catch_mode='vision' if selected else 'timed',success=None)
        self.shared.message='Rate trial: 0 miss / 1 catch only / 2 confirmed flip. Inspect/reset before next trial.'
    def nudge(self,direction,cfg):
        """One explicit positioning step; never silently saves a new zero."""
        if self.motor.reference is None:self.motor.zero() # temporary position reference
        vmax,accel=self.limits(cfg)
        current=self.motor.angle(self.motor._idle())
        step=cfg.get('zero_nudge_deg',1.)*direction
        duration=move_time(step,min(vmax,10.),accel)+1e-6
        self.move(current+step,duration,cfg,vmax,accel,exact=True)
        self.shared.message='Nudged position. Press Z to save this as zero; F still uses the previous zero.'

    def loop(self):
        try:
            link=SerialLink(self.port);self.motor=Motor112(link,a_units_per_degree=self.initial['a_units_per_degree'])
            self.motor.initialize(configure=False);self.shared.state='CONNECTED — Z at neutral'
            while not self.quit.is_set():
                try:action,cfg=self.queue.get(timeout=.1)
                except queue.Empty:continue
                try:
                    validate(cfg)
                    if cfg['a_units_per_degree']!=self.motor.a_scale:raise ValueError('Restart after changing axis calibration')
                    if action=='zero':self.motor.zero();self.shared.reference=True;self.shared.message='Neutral saved'
                    elif action in ('nudge_up','nudge_down'):
                        self.nudge(1 if action=='nudge_up' else -1,cfg)
                    elif action=='start':
                        if not self.shared.reference:raise ValueError('Zero at physical level first')
                        vmax,accel=self.limits(cfg)
                        self.move(cfg.get('start_angle_deg',0.),cfg['return']['move_s'],cfg,vmax,accel)
                        self.shared.message='At start position; F runs the three-part sequence'
                    elif action=='neutral':
                        if not self.shared.reference:raise ValueError('Zero required')
                        vmax,accel=self.limits(cfg);self.move(0,cfg['return']['move_s'],cfg,vmax,accel)
                    elif action=='catch_test':
                        if not self.shared.reference:raise ValueError('Zero required')
                        vmax,accel=self.limits(cfg)
                        self.move(cfg['catch']['fallback_angle_deg'],cfg['catch']['fallback_move_s'],cfg,vmax,accel)
                        self.hold(cfg['catch']['settle_s'])
                    else:self.trial(action,cfg)
                except ValueError as exc:
                    self.shared.message=str(exc);self.log.event('trial_rejected',trial=self.shared.trial,reason=str(exc))
                except Exception as exc:
                    self.shared.reference=False;self.motor.reference=None
                    try:self.motor.link.realtime(b'\x85')
                    except Exception:pass
                    self.shared.message=f'STOPPED: {exc}; support arm and restore reference'
                    self.log.event('fault',trial=self.shared.trial,reason=str(exc))
                finally:
                    self.shared.busy=False;self.shared.state='IDLE'
        except Exception as exc:self.shared.state='CONNECTION ERROR';self.shared.message=str(exc)
        finally:
            self.quit.set();self.shared.busy=False
            if self.motor:
                try:self.motor.link.realtime(b'\x85')
                except Exception:pass
                self.motor.link.close()
    def close(self):
        self.stop.set();self.quit.set();self.thread.join(timeout=6)
