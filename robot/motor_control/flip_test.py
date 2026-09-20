#!/usr/bin/env python3
"""One shoulder-driven cold flip trial using the existing A/X jog controller.

Default is offline preview. No hardware access on import. No physics/training.
Angles are unwrapped degrees relative to an explicitly established session zero.
"""
import argparse
import json
import math
from pathlib import Path
import time
from control_112 import ControllerError, SerialLink
from control_ax import ArmAX


def validate(cfg, speed_scale=1, hardware=False):
    if not math.isfinite(speed_scale) or not 0 < speed_scale <= 1:
        raise ValueError('speed-scale must be >0 through 1')
    for key, value in cfg.items():
        if key.endswith('_units_per_degree') and value is None and not hardware:
            continue
        values=value if isinstance(value,list) else [value]
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in values):
            raise ValueError(f'{key} must contain finite numbers; enter measured axis scales')
    for axis in ('a','x'):
        scale=cfg[axis+'_units_per_degree']
        if scale is not None and scale<=0:raise ValueError('Axis scales must be positive')
        if cfg[axis+'_direction'] not in (-1,1):raise ValueError('Directions must be +1 or -1')
        bounds=cfg[axis+'_limits_deg']
        if len(bounds)!=2 or not bounds[0]<=0<=bounds[1] or bounds[0]>=bounds[1]:
            raise ValueError('Joint limits must contain session zero')
    if not 0<cfg['wrist_max_speed_deg_s']<=30:
        raise ValueError('This test limits the wrist to at most 30 physical deg/s')
    for key,value in cfg.items():
        if key.endswith('_speed_deg_s') and value<=0:raise ValueError('Speeds must be positive')
        if key.startswith('wrist_') and key.endswith('_speed_deg_s') and value>cfg['wrist_max_speed_deg_s']:
            raise ValueError('Wrist requested speed exceeds the configured cap')
    for key in ('release_pause_s','catch_settle_s'):
        if not 0<=cfg[key]<=10:raise ValueError('Pauses must be between 0 and 10 seconds')
    if not 0<cfg['move_timeout_s']<=120 or not 0<cfg['position_tolerance_deg']<=2:
        raise ValueError('Invalid timeout or position tolerance')
    low,high=cfg['pan_pitch_limits_deg']
    if low>=high:raise ValueError('Invalid pan pitch bounds')
    a=x=0.
    for phase,axis,target,speed in phases(cfg):
        if axis=='A':a=target
        else:x=target
        lo,hi=cfg[axis.lower()+'_limits_deg']
        if not lo<=target<=hi:raise ValueError(f'{phase} exceeds joint limits')
        if not low<=cfg['pan_pitch_at_zero_deg']+a+x<=high:
            raise ValueError(f'{phase} exceeds nominal pan pitch limits')
    if not low<=cfg['pan_pitch_at_zero_deg']<=high:raise ValueError('Initial pitch outside limits')
    return cfg


def phases(c):
    return [
        ('wrist_prepare','X',c['wrist_prepare_deg'],c['wrist_prepare_speed_deg_s']),
        ('shoulder_prepare','A',c['shoulder_prepare_deg'],c['shoulder_prepare_speed_deg_s']),
        ('shoulder_launch','A',c['shoulder_launch_deg'],c['shoulder_launch_speed_deg_s']),
        ('shoulder_catch','A',c['shoulder_catch_deg'],c['shoulder_catch_speed_deg_s']),
        ('wrist_recover','X',c['wrist_recover_deg'],c['wrist_recover_speed_deg_s'])]


def preflight(arm,c,scale):
    """Reject rate clipping instead of silently changing the requested trial."""
    s=arm.firmware_settings()
    for axis,conversion,key in [('A',arm.a_scale,113),('X',arm.x_scale,110)]:
        vmax=s[key]/(60*conversion);accel=s[key+10]/conversion
        print(f'{axis}: firmware maximum {vmax:.3f} deg/s, acceleration {accel:.3f} deg/s²')
        requested=max(p[3]*scale for p in phases(c) if p[1]==axis)
        if requested>vmax+1e-6:
            raise ControllerError(f'{axis} requested {requested:g} deg/s exceeds firmware ceiling {vmax:g}; edit recipe')
    return {str(k):s[k] for k in (100,103,110,113,120,123,140,210)}


def move(arm,c,phase,speed_scale,log):
    name,axis,target,speed=phase
    sign=c[axis.lower()+'_direction']
    started=time.monotonic()
    log('move_start',phase=name,axis=axis,target_deg=target,speed_deg_s=speed*speed_scale)
    arm.move_axis(axis,sign*target,speed*speed_scale)
    status=arm.wait_until_idle(c['move_timeout_s'])
    angles=arm.angles(status)
    if abs(angles[axis]-sign*target)>c['position_tolerance_deg']:
        raise ControllerError(f'{name}: controller position differs from target; no automatic retry')
    log('move_end',phase=name,elapsed_s=time.monotonic()-started,
        commanded_angles_deg=angles,encoder_verified=False)


def prepare(arm,c,speed_scale,log):
    for phase in phases(c)[:2]:move(arm,c,phase,speed_scale,log)


def toss(arm,c,speed_scale,log):
    # Check both starting coordinates immediately before a single trial.
    angles=arm.angles(arm._idle())
    for axis,key in [('A','shoulder_prepare_deg'),('X','wrist_prepare_deg')]:
        if abs(angles[axis]-c[axis.lower()+'_direction']*c[key])>c['position_tolerance_deg']:
            raise ControllerError('Prepared position changed; re-establish reference')
    move(arm,c,phases(c)[2],speed_scale,log)
    if c['release_pause_s']:time.sleep(c['release_pause_s'])
    move(arm,c,phases(c)[3],speed_scale,log)
    time.sleep(c['catch_settle_s'])
    move(arm,c,phases(c)[4],speed_scale,log)
    log('trial_complete',flip_success='operator_observation_required')


def corrected_scale(current, commanded, measured):
    if any(not math.isfinite(v) for v in (current,commanded,measured)):
        raise ValueError('Calibration requires finite numbers')
    if current<=0 or commanded==0 or measured==0 or (commanded>0)!=(measured>0):
        raise ValueError('Positive current scale and nonzero same-sign travel required')
    return current*commanded/measured


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('flip_test.json'))
    p.add_argument('--execute',action='store_true',help='Connect and prompt before a single trial')
    p.add_argument('--port')
    p.add_argument('--speed-scale',type=float,default=.25,help='Scale speeds only, not acceleration; default 0.25')
    p.add_argument('--log',type=Path,default=Path('flip_trial.jsonl'))
    p.add_argument('--debug',action='store_true')
    p.add_argument('--calibrate-a',type=float,nargs=3,metavar=('CURRENT_SCALE','COMMANDED_DEG','MEASURED_DEG'),help='Offline: save corrected A scale; does not move hardware')
    p.add_argument('--calibrate-x',type=float,nargs=3,metavar=('CURRENT_SCALE','COMMANDED_DEG','MEASURED_DEG'),help='Offline: save corrected X scale; does not move hardware')
    args=p.parse_args(argv)
    link=None;arm=None
    try:
        c=json.loads(args.config.read_text())
        if args.calibrate_a or args.calibrate_x:
            if args.execute:raise ValueError('Calibration edits are offline; omit --execute')
            for axis,trial in [('a',args.calibrate_a),('x',args.calibrate_x)]:
                if trial:
                    c[axis+'_units_per_degree']=corrected_scale(*trial)
                    print(f'{axis.upper()} corrected units/degree: {c[axis+"_units_per_degree"]:.12g}')
            validate(c,args.speed_scale,False)
            staged=args.config.with_suffix('.json.tmp')
            staged.write_text(json.dumps(c,indent=2)+'\n');staged.replace(args.config)
            print('Saved calibration only. Verify a small slow movement and re-reference before a trial.')
            return 0
        validate(c,args.speed_scale,args.execute)
        print('A shoulder / X wrist; physical degrees relative to session zero.')
        for name,axis,target,speed in phases(c):print(f'{name}: {axis} -> {target:g}°, requested {speed*args.speed_scale:g}°/s')
        print('Wrist stays fixed during launch/catch. Acceleration/current firmware settings are preserved.')
        if not args.execute:
            print('PREVIEW ONLY. Calibrate scales/directions in the JSON before --execute.');return 0
        if not args.port:raise ValueError('--execute requires --port')
        # No firmware mutations, no implicit axis reference or guessed scales.
        bounds={axis:tuple(sorted(c[axis+'_direction']*v for v in c[axis+'_limits_deg'])) for axis in ('a','x')}
        with args.log.open('a') as handle:
            def log(event,**values):
                handle.write(json.dumps(dict(event=event,t_monotonic=time.monotonic(),**values))+'\n');handle.flush()
            link=SerialLink(args.port,debug=args.debug)
            arm=ArmAX(link,c['x_units_per_degree'],bounds['a'],bounds['x'],
                      a_units_per_degree=c['a_units_per_degree'],modulo=False,enforce_limits=True)
            arm.initialize(configure=False)
            settings=preflight(arm,c,args.speed_scale)
            log('trial_setup',config=c,speed_scale=args.speed_scale,firmware=settings)
            print('Place at the known reference with the cold pan level. Positive geometric angles increase pan pitch.')
            if input('Type zero to reference BOTH joints here, or anything else to exit: ').strip()!='zero':return 0
            arm.zero()
            if input('Type prepare to move to the listed preparation angles: ').strip()!='prepare':return 0
            prepare(arm,c,args.speed_scale,log)
            if input('Type toss for ONE trial. Ctrl+C cancels a jog: ').strip()!='toss':return 0
            toss(arm,c,args.speed_scale,log)
            print('Trial finished. Observe/video the outcome; no automatic repeat. Motors remain configured to hold.')
        return 0
    except (Exception,KeyboardInterrupt) as exc:
        if arm is not None and link is not None and link.motion_command_attempted:
            try:arm.hold();print('Jog cancelled; controller Idle. Physical position/holding not verified.')
            except BaseException:print('Stop not confirmed. Use the physical stop/support.')
        print('Stopped:',str(exc) or type(exc).__name__)
        return 1
    finally:
        if link is not None:link.close()


if __name__=='__main__':raise SystemExit(main())
