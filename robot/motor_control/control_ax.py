#!/usr/bin/env python3
"""Manual A (57HYB112) and X (57STH56) control. No hardware access on import."""
import argparse
import shlex
import time
from control_112 import ControllerError, Motor112, SerialLink, finite, parse_settings


class ArmAX(Motor112):
    """One bounded single-axis jog at a time, with both motors held at idle.

    X remains a firmware linear axis. Its numeric units are explicitly converted
    to wrist degrees; no hidden reinterpretation of mm as degrees is allowed.
    """
    def __init__(self,link,x_units_per_degree,a_limits=(-360,360),x_limits=(-360,360),speed=5,a_units_per_degree=1.0,modulo=False,enforce_limits=True):
        super().__init__(link,*a_limits,speed,a_units_per_degree)
        self.x_scale=finite(x_units_per_degree,"X units per degree")
        if self.x_scale<=0:raise ValueError("X units per degree must be positive")
        xlo,xhi=(finite(x,"X angle limit") for x in x_limits)
        if not xlo<=0<=xhi or xlo>=xhi:raise ValueError("X limits must contain zero")
        self.limits={"A":(self.minimum,self.maximum),"X":(xlo,xhi)}
        self.pending={}
        self.modulo=modulo
        self.enforce_limits=enforce_limits

    def initialize(self,configure=False,x_hold_percent=None):
        if x_hold_percent is not None:
            x_hold_percent=finite(x_hold_percent,"X hold percentage")
            if not configure or not 0<x_hold_percent<=100:
                raise ValueError("X hold percentage requires --configure and must be >0 through 100")
        info=self.link.command("$I")
        if not any("grblhal" in line.lower() for line in info):
            raise ControllerError("Expected SLB/grblHAL firmware")
        for line in info:
            if line.startswith("[AXS:") and not line.rstrip("]").split(":")[-1].startswith("XYZA"):
                raise ControllerError("Expected controller axis order X,Y,Z,A")
        self.link.ready=True
        if self.link.status().state!="Idle":raise ControllerError("Controller must be Idle")
        self.settings=parse_settings(self.link.command("$$"))
        required=(1,13,37,100,103,110,113,120,123,140,210,338,376)
        if any(k not in self.settings for k in required):
            raise ControllerError("Missing A/X settings; run --diagnose first")
        s=self.settings
        if any(s[k]!=int(s[k]) for k in (37,338,376)):
            raise ControllerError("Axis masks must be integers")
        if s[13]!=0 or not int(s[376])&1:
            raise ControllerError("Require metric reporting $13=0 and rotational A in $376")
        if s[1]<=0 or any(s[k]<=0 for k in (100,103,110,113,120,123,140)):
            raise ControllerError("Idle delay, calibrations, rates, accelerations and X current must be positive")
        if not int(s[338])&1:
            raise ControllerError("This X hold-current setup requires the SLB onboard Trinamic X driver")
        requested_hold=s[210] if x_hold_percent is None else x_hold_percent
        if not 0<requested_hold<=100:
            raise ControllerError("X hold current must be >0 through 100%; specify --configure --x-hold-percent")
        changes=[]
        if configure:
            # Preserve Y/Z and any additional axes; set X=1 and A=8.
            if x_hold_percent is not None and s[210]!=x_hold_percent:
                self.link.command(f"$210={x_hold_percent:g}")
                changes.append(f"$210: {s[210]:g} -> {x_hold_percent:g}%")
            old=int(s[37]);new=old|9
            if new!=old:
                self.link.command(f"$37={new}");changes.append(f"$37: {old} -> {new}")
            self.settings=parse_settings(self.link.command("$$"))
        if int(self.settings[37])&9!=9:
            raise ControllerError("Both A and X must remain enabled: support the arm and use --configure")
        if abs(self.settings[210]-requested_hold)>1e-6:
            raise ControllerError("X hold-current readback did not match the requested value")
        return changes

    def configure_motion(self, configure=False, acceleration=5.0):
        """Convert joint acceleration to firmware units; preserve current/rate settings."""
        acceleration=finite(acceleration,"joint acceleration")
        if acceleration<=0:raise ValueError("Acceleration must be positive")
        self._idle()
        desired={120:acceleration*self.x_scale,123:acceleration*self.a_scale}
        settings=self.firmware_settings();changes=[]
        for key,value in desired.items():
            if abs(settings.get(key,0)-value)>max(1e-5,value*.001):
                if not configure:
                    raise ControllerError("Acceleration settings differ; use --configure once to apply --acceleration in degrees/s²")
                self.link.command(f"${key}={value:.9g}")
                changes.append(f"${key}: {settings.get(key)} -> {value:g}")
        self.settings=self.firmware_settings()
        for key,value in desired.items():
            if abs(self.settings.get(key,0)-value)>max(1e-5,value*.001):
                raise ControllerError(f"Acceleration readback mismatch for ${key}")
        return changes

    def move_to_angle(self, axis, angle, speed):
        """AI entry point: degrees from session zero, deg/s; nonblocking acceptance.

        Firmware performs the ramp. Call wait_until_idle before the next target.
        Completion is controller execution, not encoder-verified physical arrival.
        """
        self.move_axis(axis,angle,speed)

    def wait_until_idle(self, timeout=30.0):
        timeout=finite(timeout,"timeout")
        if timeout<=0:raise ValueError("Timeout must be positive")
        deadline=time.monotonic()+timeout
        while True:
            status=self.link.status()
            if status.state=="Idle":
                self.pending={}
                return status
            if status.state!="Jog":
                raise ControllerError(f"Motion interrupted: {status.state}")
            if time.monotonic()>=deadline:
                self.hold()
                raise ControllerError("Motion timed out; jog cancelled")
            time.sleep(.02)

    def position(self,status,axis):
        return status.a/self.a_scale if axis=="A" else status.x/self.x_scale

    def calibrate_axis(self,axis,commanded,measured):
        axis=axis.upper()
        if axis not in ("A","X"):raise ValueError("Choose A or X")
        commanded=finite(commanded,"commanded travel");measured=finite(measured,"measured travel")
        if commanded==0 or measured==0 or (commanded>0)!=(measured>0):
            raise ValueError("Use nonzero travel angles of the same sign; measure travel, not final orientation")
        previous=self.a_scale if axis=="A" else self.x_scale
        scale=finite(previous*(commanded/measured),"corrected scale")
        if scale<=0:raise ValueError("Corrected scale must be positive")
        self._idle()
        if axis=="A":self.a_scale=scale
        else:self.x_scale=scale
        self.reference=None
        return scale

    def angles(self,status):
        if self.reference is None:raise ControllerError("Enter zero at the supported reference first")
        if status.coordinates!=self.reference.coordinates:
            raise ControllerError("Position coordinate system changed; re-reference")
        return {axis:self.position(status,axis)-self.position(self.reference,axis) for axis in ("A","X")}

    def _idle(self):
        status=self.link.status()
        if status.state!="Idle":raise ControllerError("Wait for Idle or use hold before another move")
        # Idle means execution has ended, even if target and readback differ.
        # Never overwrite position with the requested target or infer shaft feedback.
        self.pending={}
        return status

    def zero(self):
        self.reference=self._idle()
        return self.angles(self.reference)

    def move_axis(self,axis,value,speed=None,relative=False):
        axis=axis.upper()
        if axis not in self.limits:raise ValueError("Choose axis A or X")
        status=self._idle();current=self.angles(status)
        value=finite(value,"angle");speed=finite(self.default_speed if speed is None else speed,"speed")
        if speed<=0:raise ValueError("Speed must be positive")
        target=current[axis]+value if relative else value
        if self.modulo and not relative:
            # Keep unwrapped position internally; choose the shortest angular path.
            delta_angle=((value % 360)-(current[axis] % 360)+180)%360-180
            if delta_angle == -180:delta_angle=180
            target=current[axis]+delta_angle
        lo,hi=self.limits[axis]
        if self.enforce_limits and not lo<=target<=hi:raise ValueError(f"{axis} target outside {lo:g} to {hi:g} degrees")
        for joint,(low,high) in self.limits.items():
            if self.enforce_limits and not low-.05<=current[joint]<=high+.05:
                raise ControllerError(f"{joint} position outside configured limits")
        scale=self.a_scale if axis=="A" else self.x_scale
        delta=finite((target-current[axis])*scale,"jog distance")
        feed=finite(speed*scale*60,"feed rate")
        if abs(target-current[axis])<.001:return
        self.pending={joint:self.position(status,joint) for joint in ("A","X")}
        self.pending[axis]=self.position(self.reference,axis)+target
        try:self.link.command(f"$J=G21 G91 {axis}{delta:.6f} F{feed:.3f}")
        except BaseException:
            self.reference=None
            try:self.link.realtime(b"\x85")
            except Exception:pass
            raise

    def hold(self,timeout=6):
        status=super().hold(timeout)
        self.pending={}
        return status


HELP="""Commands (degrees and degrees/second):
  zero            Reference BOTH shafts at the current supported position
  a 10 5          A -> orientation 10 degrees at 5 deg/s (shortest path)
  a to 10 5       Same as a 10 5
  x to -10 3      X -> -10 degrees at 3 deg/s
  a by 5          A +5 degrees at the session speed
  x by 5          X +5 degrees at the session speed
  speed 5         Set requested speed for subsequent moves on either axis
  hold            Cancel the current jog; both axes remain configured to hold
  status          Read commanded positions (not encoder feedback)
  settings        Show A/X calibration, rate, acceleration and hold settings
  calibrate x 45 135  Correct X using commanded/measured travel; then re-zero
  calibrate a 45 135  Same for A, only if A has that measured error
  quit            Hold and disconnect
One move at a time. Wait for Idle or cancel with hold before the next move.
CLI targets wrap modulo 360; 360 equals 0. A 180-degree tie moves positive.
No default software travel bounds; firmware limits and stop/fault checks remain.
Use a by 360 for an explicit full revolution, only after physical calibration.
Ctrl+C requests hold. Power loss or overload can still release/drop the arm.
"""


def interactive(arm):
    print(HELP)
    while True:
        try:
            words=shlex.split(input("A/X> "))
            if not words:continue
            name,*args=words;name=name.lower()
            if name=="zero" and not args:print("Session angles:",arm.zero())
            elif name in ("a","x") and len(args) in (1,2) and args[0] not in ("to","by"):
                arm.move_axis(name,float(args[0]),float(args[1]) if len(args)==2 else None)
                print("Angle target accepted.")
            elif name in ("a","x") and len(args) in (2,3) and args[0] in ("to","by"):
                arm.move_axis(name,float(args[1]),float(args[2]) if len(args)==3 else None,args[0]=="by")
                print("Move accepted; status or hold remains available.")
            elif name=="speed" and len(args)==1:arm.set_speed(float(args[0]))
            elif name=="calibrate" and len(args)==3:
                scale=arm.calibrate_axis(args[0],float(args[1]),float(args[2]))
                print(f"Session correction applied. Enter zero. Reuse at launch: --{args[0].lower()}-units-per-degree {scale:.12g}")
            elif name=="status" and not args:
                s=arm.link.status();print(s.state,s.coordinates,"A=",s.a,"X units=",s.x,
                                         "angles=",{k:v%360 for k,v in arm.angles(s).items()} if arm.reference else "unset")
            elif name=="settings" and not args:
                s=arm.firmware_settings()
                for k in (1,4,37,100,103,110,113,120,123,140,150,210,338,376):print(f"${k}={s.get(k,'unreported')}")
                print(f"X conversion: {arm.x_scale:g} controller units/degree")
                print(f"A conversion: {arm.a_scale:g} controller units/degree")
            elif name in ("hold","quit") and not args:
                arm.hold();print("Idle confirmed; both axes remain configured to hold.")
                if name=="quit":return
            elif name=="help" and not args:print(HELP)
            else:print("Unknown command; enter help")
        except KeyboardInterrupt:
            arm.hold();print("Jog stopped. Enter quit to exit.")
        except EOFError:arm.hold();return
        except (ControllerError,ValueError) as exc:
            print(f"Not executed/confirmed: {exc}")
            if getattr(arm.link,"fault",None):raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port")
    p.add_argument("--list-ports",action="store_true")
    p.add_argument("--debug",action="store_true")
    p.add_argument("--diagnose",action="store_true")
    p.add_argument("--configure",action="store_true")
    p.add_argument("--acceleration",type=float,help="Optional explicit acceleration change in degrees/s²; requires --configure")
    p.add_argument("--x-hold-percent",type=float)
    p.add_argument("--x-units-per-degree",type=float,help="Required conversion based on X calibration and microsteps")
    p.add_argument("--a-units-per-degree",type=float,
                   help="Required measured A conversion; no guessed default")
    p.add_argument("--enforce-angle-limits",action="store_true",help="Apply the configured unwrapped joint bounds")
    p.add_argument("--a-min-angle",type=float,default=-360)
    p.add_argument("--a-max-angle",type=float,default=360)
    p.add_argument("--x-min-angle",type=float,default=-360)
    p.add_argument("--x-max-angle",type=float,default=360)
    p.add_argument("--speed",type=float,default=5)
    args=p.parse_args(argv)
    if args.diagnose and (args.configure or args.x_hold_percent is not None or args.acceleration is not None):p.error("Diagnostic mode is read-only")
    if args.acceleration is not None and not args.configure:p.error("--acceleration requires --configure; otherwise existing ramps are preserved")
    if args.x_hold_percent is not None and not args.configure:p.error("--x-hold-percent requires --configure")
    if args.list_ports:
        from serial.tools import list_ports
        for item in list_ports.comports():print(item.device,item.description)
        return 0
    if not args.port:p.error("Specify --port or --list-ports")
    if not args.diagnose and args.x_units_per_degree is None:p.error("Specify --x-units-per-degree from your calibration")
    if not args.diagnose and args.a_units_per_degree is None:p.error("Specify --a-units-per-degree; the previous approximate 1/3 correction was not verified. Use measured calibration or driver pulses/revolution and $103.")
    link=None
    try:
        if not args.diagnose:
            # Validate all numeric arguments before opening a hardware connection.
            arm=ArmAX(None,args.x_units_per_degree,(args.a_min_angle,args.a_max_angle),
                      (args.x_min_angle,args.x_max_angle),args.speed,args.a_units_per_degree,modulo=True,enforce_limits=args.enforce_angle_limits)
            if args.x_hold_percent is not None and not 0<finite(args.x_hold_percent,"X hold percentage")<=100:
                raise ValueError("X hold percentage must be >0 through 100")
        if args.acceleration is not None and finite(args.acceleration,"acceleration")<=0:raise ValueError("Acceleration must be positive")
        link=SerialLink(args.port,debug=args.debug)
        if args.diagnose:
            print(link.command("$I"))
            s=parse_settings(link.command("$$"))
            for k in (1,4,13,37,100,103,110,113,120,123,140,150,210,338,376):print(f"${k}={s.get(k,'unreported')}")
            print(link.status());return 0
        arm.link=link
        for change in arm.initialize(args.configure,args.x_hold_percent):print("Persistent change:",change)
        if args.acceleration is not None:
            for change in arm.configure_motion(args.configure,args.acceleration):
                print("Persistent change:",change)
        for axis,scale,key in (("A",arm.a_scale,113),("X",arm.x_scale,110)):
            print(f"{axis}: maximum speed {arm.settings[key]/scale/60:g} deg/s; acceleration {arm.settings[key+10]/scale:g} deg/s²")
        print(f"X running current setting: {arm.settings[140]:g} mA RMS; idle percentage: {arm.settings[210]:g}%.")
        print(f"A conversion: {arm.a_scale:.12g}; X conversion: {arm.x_scale:.12g} controller units/physical degree.")
        print("These settings do not prove physical holding or shaft position. No motion commanded yet.")
        interactive(arm);return 0
    except (Exception,KeyboardInterrupt) as exc:
        if link is not None and link.motion_command_attempted:
            try:link.realtime(b"\x85")
            except Exception:pass
            print("Stop/holding not confirmed; support the arm.")
        print("ERROR:",str(exc) or type(exc).__name__);return 1
    finally:
        if link is not None:link.close()


if __name__=="__main__":raise SystemExit(main())
