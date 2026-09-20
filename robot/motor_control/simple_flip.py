#!/usr/bin/env python3
"""A-only position, launch, catch and return experiment. No motion on import."""
import argparse
import json
import math
from pathlib import Path
import time
from control_112 import SerialLink, ControllerError, Motor112

A_UNITS_PER_DEGREE = 0.049984558
MOVE_SPEED_DEG_S = 60.0
CONFIG_PATH = Path(__file__).with_name('toss_config.json')
PHASES = ('position', 'launch', 'catch', 'return')

def finite(value):
    value=float(value)
    if not math.isfinite(value):raise ValueError('Use a finite angle')
    return value


def motor_limits(motor):
    speed=finite(motor.settings[113])/(60*motor.a_scale)
    accel=finite(motor.settings[123])/motor.a_scale
    if min(speed,accel)<=0:raise ValueError('Firmware speed/acceleration must be positive')
    return speed,accel


def ramp_time(distance,speed,accel):
    """Rest-to-rest triangular/trapezoidal ramp estimate, including braking."""
    distance=abs(distance)
    if distance<=speed*speed/accel:return 2*math.sqrt(distance/accel)
    return distance/speed+speed/accel


def wait_idle(motor,timeout):
    deadline=time.monotonic()+timeout
    while True:
        status=motor.link.status()
        if status.state=='Idle':return status
        if status.state!='Jog':raise ControllerError(f'Motion interrupted: {status.state}')
        if time.monotonic()>=deadline:raise ControllerError('Motion timed out')
        time.sleep(.01)


def execute(motor,targets,speed):
    current=motor.angle(motor._idle());vmax,accel=motor_limits(motor)
    speeds=speed if isinstance(speed,(list,tuple)) else [speed]*len(targets)
    if len(speeds)!=len(targets):raise ValueError('One speed is required per phase')
    plans=[];end=current;expected=0.
    for target,rate in zip(targets,speeds):
        speed=min(finite(rate),vmax)
        if speed<=0:raise ValueError('Speed must be positive')
        target=finite(target)
        delta=target-end;units=round(delta*motor.a_scale,9)
        if units:
            feed=math.floor(speed*motor.a_scale*60*1e9)/1e9
            if feed<=0:raise ValueError('Speed too small')
            plans.append(f'$J=G21 G91 A{units:.9f} F{feed:.9f}')
            expected+=ramp_time(delta,speed,accel)
        end=target
    if not plans:return 0.
    began=time.monotonic()
    try:
        for command in plans:motor.link.command(command)
        status=wait_idle(motor,max(3.,expected+2.))
        if abs(motor.angle(status)-end)>.5:
            raise ControllerError('Controller did not finish at target; restore reference')
    except BaseException:
        motor.reference=None
        try:motor.link.realtime(b'\x85')
        except Exception:pass
        raise
    return time.monotonic()-began


def load_config(path=CONFIG_PATH):
    cfg=json.loads(Path(path).read_text())
    if set(cfg)!={'acceleration_deg_s2',*PHASES}:
        raise ValueError('Config requires acceleration_deg_s2 and position/launch/catch/return')
    cfg['acceleration_deg_s2']=finite(cfg['acceleration_deg_s2'])
    if cfg['acceleration_deg_s2']<=0:raise ValueError('Acceleration must be positive')
    for name in PHASES:
        phase=cfg[name]
        if set(phase)!={'angle_deg','move_s','hold_s'}:raise ValueError(f'{name}: use angle_deg, move_s, hold_s')
        for key in phase:phase[key]=finite(phase[key])
        if phase['move_s']<=0 or phase['hold_s']<0:raise ValueError('Move times must be positive; holds nonnegative')
    return cfg


def check_acceleration(motor,cfg):
    _,actual=motor_limits(motor)
    cap=cfg['acceleration_deg_s2']
    if actual>cap*1.001:
        raise ValueError(f'A acceleration is {actual:.1f} deg/s², above configured {cap:g}. '
                         'No motion sent. Restart with --apply-acceleration to lower $123, '
                         'or set a measured acceleration limit in toss_config.json.')


def apply_acceleration(motor,cfg):
    motor._idle()
    _,actual=motor_limits(motor)
    cap=cfg['acceleration_deg_s2']
    if actual>cap:
        value=math.floor(cap*motor.a_scale*1e6)/1e6
        if value<=0:raise ValueError('Acceleration is too small to encode')
        motor.link.command(f'$123={value:.6f}')
        motor.settings=motor.firmware_settings()
        print(f'Lowered persistent $123 to {value:.6f}; speed settings unchanged.')
    check_acceleration(motor,cfg)


def toss_plan(motor,cfg,angles=None):
    check_acceleration(motor,cfg)
    vmax,accel=motor_limits(motor)
    if angles is not None and len(angles)!=4:raise ValueError('Use four absolute angles')
    phases=[];previous=0.
    for i,name in enumerate(PHASES):
        p=dict(cfg[name]);p['name']=name
        if angles is not None:p['angle_deg']=finite(angles[i])
        d=abs(p['angle_deg']-previous);duration=p['move_s']
        fastest=ramp_time(d,vmax,accel)
        if duration+1e-9<fastest:
            raise ValueError(f'{name}: {duration:g}s is too short; needs at least {fastest:.3f}s '
                             'at configured limits. Increase move_s or reduce angle. No motion sent.')
        # Solve T = distance/v + v/a for the lower feasible peak speed.
        p['speed']=2*d/(duration+math.sqrt(max(0.,duration*duration-4*d/accel))) if d else 0.
        p['distance']=d
        phases.append(p);previous=p['angle_deg']
    if not any(p['distance'] for p in phases):raise ValueError('Toss contains no movement')
    return phases


def monitored_hold(motor,seconds,target):
    deadline=time.monotonic()+seconds
    while True:
        status=motor.link.status()
        if status.state!='Idle':raise ControllerError(f'Hold interrupted: {status.state}')
        if abs(motor.angle(status)-target)>.5:raise ControllerError('Commanded hold position changed')
        remaining=deadline-time.monotonic()
        if remaining<=0:return
        time.sleep(min(.02,remaining))


def perform_toss(motor,phases):
    began=time.monotonic()
    try:
        for p in phases:
            if p['distance']:execute(motor,[p['angle_deg']],p['speed'])
            # A zero-distance phase consumes its requested move time as a dwell.
            hold=p['hold_s']+(p['move_s'] if not p['distance'] else 0.)
            monitored_hold(motor,hold,p['angle_deg'])
    except BaseException:
        motor.reference=None
        try:motor.link.realtime(b'\x85')
        except Exception:pass
        raise
    return time.monotonic()-began


def parse(line):
    commands=[]
    for part in line.lower().split(';'):
        words=part.split()
        if not words:continue
        if words[0]=='move' and len(words)==2:commands.append(('move',finite(words[1])))
        elif words[0]=='zero' and len(words)==1:commands.append(('zero',None))
        elif words[0]=='toss' and len(words) in (1,5):
            commands.append(('toss',[finite(v) for v in words[1:]] if len(words)==5 else None))
        else:raise ValueError('Use move ANGLE, zero, toss, or toss POSITION LAUNCH CATCH RETURN')
    return commands


def run(motor,config_path=CONFIG_PATH):
    motor.zero();base_confirmed=False
    print('A only: move ANGLE; zero; toss [POSITION LAUNCH CATCH RETURN]. Angles are absolute.')
    print(f'Edit {config_path} for phase times/holds; reloaded before each movement. Ctrl+C cancels/exits.')
    while True:
        try:
            for command,value in parse(input('> ')):
                if command=='zero':
                    motor.zero();base_confirmed=True;print('Base saved. No movement.');continue
                cfg=load_config(config_path)
                motor.settings=motor.firmware_settings()
                check_acceleration(motor,cfg)
                if command=='move':execute(motor,[value],MOVE_SPEED_DEG_S)
                else:
                    if not base_confirmed:raise ValueError('Type zero at the physical neutral pose first')
                    if abs(motor.angle(motor._idle()))>.5:raise ValueError('Use move 0 to return to neutral first')
                    phases=toss_plan(motor,cfg,value)
                    for p in phases:
                        print(f"{p['name']}: {p['angle_deg']:g}°, move {p['move_s']:g}s, "
                              f"hold {p['hold_s']:g}s, requested peak {p['speed']:.1f}°/s")
                    total=sum(p['move_s']+p['hold_s'] for p in phases)
                    print(f'Nominal total {total:.3f}s + communication latency. Catch is not sensor-verified.')
                    elapsed=perform_toss(motor,phases)
                    print(f'Completed commanded sequence in {elapsed:.3f}s. Physical catch/position not verified.')
        except (ValueError,OSError) as exc:print(exc,'Remaining commands discarded.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',required=True)
    p.add_argument('--a-units-per-degree',type=float,default=A_UNITS_PER_DEGREE)
    p.add_argument('--config',type=Path,default=CONFIG_PATH)
    p.add_argument('--apply-acceleration',action='store_true',help='Lower persistent A acceleration to config cap; never raises it')
    args=p.parse_args();link=None;motor=None
    try:
        cfg=load_config(args.config)
        motor=Motor112(None,a_units_per_degree=args.a_units_per_degree)
        link=SerialLink(args.port);motor.link=link;motor.initialize(configure=False)
        if args.apply_acceleration:apply_acceleration(motor,cfg)
        vmax,accel=motor_limits(motor)
        print(f'A firmware limits: {vmax:.1f}°/s, {accel:.1f}°/s². Move default remains {MOVE_SPEED_DEG_S:g}°/s.')
        run(motor,args.config)
    except (Exception,KeyboardInterrupt) as exc:
        if link is not None and link.motion_command_attempted:
            try:motor.hold();print('Queued jogs cancelled; controller Idle.')
            except BaseException:print('Stop not confirmed. Use physical stop/support.')
        if not isinstance(exc,(KeyboardInterrupt,EOFError)):print('Stopped:',str(exc) or type(exc).__name__)
        return 0 if isinstance(exc,(KeyboardInterrupt,EOFError)) else 1
    finally:
        if link is not None:link.close()


if __name__=='__main__':raise SystemExit(main())
