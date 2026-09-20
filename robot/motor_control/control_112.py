#!/usr/bin/env python3
"""Manual angle/hold control for a Sienci 57HYB112 on the SLB external A axis.

No connection or motion occurs on import. Uses grblHAL bounded jogs, not a host
PID loop. The integrated motor driver/encoder must remain connected and enabled.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import re
import shlex
import time


class ControllerError(RuntimeError):
    pass


def finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class Status:
    state: str
    a: float
    coordinates: str
    z: float = 0.0
    x: float = 0.0


def parse_status(line: str) -> Status:
    if not (line.startswith("<") and line.endswith(">")):
        raise ControllerError("Malformed controller status")
    parts = line[1:-1].split("|")
    fields = dict(item.split(":", 1) for item in parts[1:] if ":" in item)
    for key in ("MPos", "WPos"):
        if key in fields:
            values = fields[key].split(",")
            if len(values) < 4:
                raise ControllerError("Controller does not report an A axis (expected X,Y,Z,A)")
            return Status(parts[0], finite(values[3], "A position"), key,
                          finite(values[2], "Z position"), finite(values[0], "X position"))
    raise ControllerError("Status has no MPos/WPos; enable position reporting in gSender")


NUMERIC_MOTOR_SETTINGS = frozenset((1, 4, 13, 37, 100, 102, 103, 110, 112, 113, 120, 122, 123, 140, 142, 150, 152, 210, 212, 338, 376))


def parse_settings(lines: list[str]) -> dict[int, float | str]:
    """Keep unrelated settings as text; validate motor values without truncation."""
    result = {}
    for line in lines:
        match = re.fullmatch(r"\$(\d+)=(.*)", line)
        if not match:
            continue
        key, value = int(match[1]), match[2].strip()
        if key in NUMERIC_MOTOR_SETTINGS:
            # Some firmware includes a parenthesized description after a value.
            number = re.sub(r"\s+\(.*\)$", "", value)
            try:
                result[key] = finite(number, f"setting ${key}")
            except ValueError as exc:
                raise ControllerError(
                    f"Invalid numeric motor setting ${key}={value!r}"
                ) from exc
        else:
            # Network addresses, names, passwords and empty strings are valid
            # controller settings, but are not numbers used by this program.
            result[key] = value
    return result


class SerialLink:
    """One outstanding text command at a time; never resets or unlocks the board."""

    def __init__(self, port: str, baud: int = 115200, debug: bool = False):
        try:
            import serial
        except ImportError as exc:
            raise ControllerError("Install pyserial: python3 -m pip install pyserial") from exc
        import os
        self.serial = serial.Serial(port=None, baudrate=baud, timeout=.05, write_timeout=1)
        # Native STM32 grblHAL USB uses asserted DTR to recognize an open host
        # connection. Do not pulse/toggle DTR as an Arduino-style reset sequence.
        self.serial.dtr = True
        self.serial.rts = False
        if os.name != "nt":
            self.serial.exclusive = True
        self.serial.port = port
        self.serial.open()
        time.sleep(.35)  # Allow the firmware's USB line-state transition to settle.
        self.buffer = bytearray()
        self.ready = False
        self.fault = None
        self.debug = debug
        self.last_request = "opening the USB connection"
        self.last_reply = None
        self.motion_command_attempted = False
        if debug:
            print(f"USB opened: {port}, {baud} baud, DTR=on, RTS=off")

    def realtime(self, data: bytes):
        # Intentionally allowed even after a protocol fault, for best-effort stop.
        if self.debug:
            print(f"TX {data!r}")
        try:
            if self.serial.write(data) != len(data):
                raise OSError("Short serial write")
        except Exception as exc:
            self.fault = f"Serial write failed: {exc}"
            raise ControllerError(self.fault) from exc

    def _line(self, deadline: float) -> str:
        while time.monotonic() < deadline:
            if b"\n" in self.buffer:
                raw, _, rest = self.buffer.partition(b"\n")
                self.buffer = bytearray(rest)
                line = raw.decode("ascii", errors="replace").strip()
                if not line:
                    continue
                self.last_reply = line
                if self.debug:
                    print(f"RX {line}")
                if line.startswith("ALARM:"):
                    self.fault = line
                if self.ready and line.lower().startswith(("grbl", "grblhal")):
                    self.fault = "Controller restarted; position reference is no longer valid"
                if self.fault:
                    raise ControllerError(self.fault)
                return line
            try:
                self.buffer.extend(self.serial.read(max(1, self.serial.in_waiting)))
            except Exception as exc:
                self.fault = f"Serial read failed: {exc}"
                raise ControllerError(self.fault) from exc
        reply = repr(self.last_reply) if self.last_reply is not None else "none"
        self.fault = f"Timed out waiting for {self.last_request!r}; last controller reply: {reply}"
        raise ControllerError(self.fault)

    def command(self, text: str, timeout: float = 5) -> list[str]:
        if self.fault:
            raise ControllerError(self.fault)
        self.last_request = text
        self.last_reply = None
        if text.startswith("$J="):
            self.motion_command_attempted = True
        self.realtime((text + "\n").encode("ascii"))
        lines = []
        deadline = time.monotonic() + timeout
        while True:
            line = self._line(deadline)
            if line == "ok":
                return lines
            if line.startswith("error:"):
                raise ControllerError(f"Controller rejected {text!r}: {line}")
            if not line.startswith("<"):
                lines.append(line)

    def status(self) -> Status:
        if self.fault:
            raise ControllerError(self.fault)
        self.last_request = "? (status)"
        self.last_reply = None
        self.realtime(b"?")
        deadline = time.monotonic() + 3
        while True:
            line = self._line(deadline)
            if line.startswith("<"):
                state = line[1:].split("|",1)[0].rstrip(">")
                if state.split(":")[0] in ("Alarm", "EStop", "Sleep"):
                    self.fault = f"Controller state is {state}; holding is not confirmed"
                    raise ControllerError(self.fault)
                return parse_status(line)
            if line.startswith("error:"):
                raise ControllerError(line)

    def close(self):
        self.serial.close()


class Motor112:
    """Angles are relative to the explicitly selected session zero, in degrees."""

    def __init__(self, link, minimum=-360.0, maximum=360.0, default_speed=5.0, a_units_per_degree=1.0):
        self.link = link
        self.a_scale=finite(a_units_per_degree,"A units per degree")
        if self.a_scale<=0:raise ValueError("A units per degree must be positive")
        self.minimum = finite(minimum, "minimum angle")
        self.maximum = finite(maximum, "maximum angle")
        self.set_speed(default_speed)
        if not self.minimum <= 0 <= self.maximum or self.minimum >= self.maximum:
            raise ValueError("Angle limits must contain zero and minimum must be below maximum")
        self.reference: Status | None = None
        self.pending_target: float | None = None
        self.settings = {}

    def initialize(self, configure=False):
        info = self.link.command("$I")
        if not any("grblhal" in line.lower() for line in info):
            raise ControllerError("This controller is intended for SLB/grblHAL; firmware not identified")
        for line in info:
            if line.startswith("[AXS:") and not line.rstrip("]").split(":")[-1].startswith("XYZA"):
                raise ControllerError("Unexpected axis order; expected X,Y,Z,A")
        self.link.ready = True
        state = self.link.status().state
        if state != "Idle":
            raise ControllerError(f"Controller is {state}, expected Idle; no automatic unlock, home or resume")
        self.settings = parse_settings(self.link.command("$$"))
        for key in (1, 13, 37, 103, 113, 123, 376):
            if key not in self.settings:
                raise ControllerError(f"Missing required grblHAL setting ${key}")
        if not int(self.settings[376]) & 1:
            raise ControllerError("Set A as rotational ($376 A bit) in gSender before using degree commands")
        if self.settings[13] != 0:
            raise ControllerError("Use metric status reporting ($13=0) before running this controller")
        if self.settings[1] == 0:
            raise ControllerError("$1=0 can immediately disable motors; select a positive idle delay first")
        if any(self.settings[k] <= 0 for k in (103, 113, 123)):
            raise ControllerError("A-axis steps/unit, maximum rate and acceleration must be positive")
        changes = []
        if configure:
            # Sienci's $37 A toggle corresponds to bit 3. Preserve other axes.
            old = int(self.settings[37])
            if not old & 8:
                self.link.command(f"$37={old | 8}")
                changes.append(f"$37: {old} -> {old | 8} (keep A enabled while idle)")
            self.settings = parse_settings(self.link.command("$$"))
        if not int(self.settings.get(37, 0)) & 8:
            raise ControllerError("A hold-enable bit is off. Support the arm, then run with --configure")
        return changes

    def set_speed(self, speed: float):
        """Change the session's requested speed, with no application speed ceiling."""
        speed = finite(speed, "speed")
        if speed <= 0:
            raise ValueError("Speed must be positive")
        finite(speed*60, "feed rate")
        self.default_speed = speed

    def firmware_settings(self):
        """Read the controller's rate/acceleration limits; never change them."""
        return parse_settings(self.link.command("$$"))

    def _idle(self) -> Status:
        status = self.link.status()
        if status.state != "Idle":
            raise ControllerError(f"Axis/controller is {status.state}; use hold or wait before another move")
        # Idle completes command execution, not verified physical positioning.
        # Continue from reported coordinates rather than pretending target was reached.
        self.pending_target = None
        return status

    def zero(self):
        self.reference = self._idle()
        # A software reference only: do not send G92/G10 or move the motor.
        return self.reference.a

    def angle(self, status: Status) -> float:
        if self.reference is None:
            raise ControllerError("Place the arm at its chosen reference, then enter zero")
        if status.coordinates != self.reference.coordinates:
            raise ControllerError("Position-report coordinate system changed; stop and re-reference")
        return (status.a-self.reference.a)/self.a_scale

    def calibrate(self,commanded,measured):
        """Correct a measured relative travel, without motion or firmware writes."""
        commanded=finite(commanded,"commanded travel")
        measured=finite(measured,"measured travel")
        if commanded==0 or measured==0 or (commanded>0)!=(measured>0):
            raise ValueError("Use nonzero travel angles of the same sign; measure travel, not final orientation")
        scale=finite(self.a_scale*(commanded/measured),"corrected A scale")
        if scale<=0:raise ValueError("Corrected scale must be positive")
        self._idle()
        self.a_scale=scale;self.reference=None
        return scale

    def _move(self, status: Status, target: float, speed: float | None):
        current = self.angle(status)
        target = finite(target, "target angle")
        speed = finite(self.default_speed if speed is None else speed, "speed")
        feed = finite(speed*self.a_scale*60, "feed rate")
        if not self.minimum <= target <= self.maximum:
            raise ValueError(f"Target outside session limits {self.minimum:g} to {self.maximum:g} degrees")
        if not self.minimum-.05 <= current <= self.maximum+.05:
            raise ControllerError("Current angle is outside the configured limits")
        if speed <= 0:
            raise ValueError("Speed must be positive")
        delta = finite((target-current)*self.a_scale,"jog distance")
        if abs(target-current) < .001:
            return
        # G21/G91 are local to the jog; ordinary G-code coordinate modes are preserved.
        self.pending_target = self.reference.a+target*self.a_scale
        try:
            self.link.command(f"$J=G21 G91 A{delta:.6f} F{feed:.3f}")
        except BaseException:
            # No blind retry of movement after an uncertain acknowledgement.
            self.reference = None
            try:
                self.link.realtime(b"\x85")
            except Exception:
                pass
            raise

    def move_to(self, degrees: float, speed: float | None = None):
        """Start a bounded move to an angle relative to session zero."""
        self._move(self._idle(), degrees, speed)

    def move_by(self, degrees: float, speed: float | None = None):
        """Start a bounded relative rotation from the latest controller position."""
        status = self._idle()
        self._move(status, self.angle(status)+finite(degrees, "rotation"), speed)

    def hold(self, timeout: float = 6) -> Status:
        """Decelerate/cancel this app's jog, discard its target, keep A enabled.

        This is not a power-cut emergency stop. Does not stop/resume another
        sender's ordinary G-code job, and never clears alarms or disables motors.
        """
        self.link.realtime(b"\x85")
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            status = self.link.status()
            if status.state == "Idle":
                self.pending_target = None
                return status
            if status.state != "Jog":
                raise ControllerError(f"Hold not confirmed: controller is {status.state}")
            time.sleep(.05)
        raise ControllerError("Stop was not confirmed; use the physical stop and support the arm")


HELP = """Commands (angles in degrees, speed in degrees/second):
  zero           Call this supported position 0 degrees for this session
  speed 60       Set requested speed for subsequent moves to 60 deg/s
  to 20          Move to +20 degrees from zero at the session speed
  to -10 3       Move to -10 degrees at 3 deg/s
  by 5           Rotate +5 degrees from the current position
  hold           Stop/cancel the move; remain energized at the stopping position
  status         Read controller state and commanded A position
  settings       Read firmware speed/acceleration settings (does not change them)
  calibrate 45 135  Correct A using commanded vs measured travel; then re-zero
  quit           Hold, then disconnect; leaves the A hold setting enabled
  help           Show these commands
Ctrl+C requests hold. Close gSender before connecting this program.
No application speed or acceleration cap. Firmware limits still apply.
"""


def interactive(motor: Motor112):
    print(HELP)
    while True:
        try:
            words = shlex.split(input("112> "))
            if not words:
                continue
            name, *args = words
            if name == "zero" and not args:
                print(f"Session zero set at controller A={motor.zero():.3f}; no motion commanded")
            elif name == "speed" and len(args) == 1:
                motor.set_speed(float(args[0]))
                print(f"Requested speed for subsequent moves: {motor.default_speed:g} deg/s")
            elif name == "calibrate" and len(args)==2:
                scale=motor.calibrate(float(args[0]),float(args[1]))
                print(f"A correction saved for this session. Enter zero. Next launch: --a-units-per-degree {scale:.12g}")
            elif name in ("to", "by") and len(args) in (1, 2):
                target = float(args[0]); speed = float(args[1]) if len(args) == 2 else None
                (motor.move_to if name == "to" else motor.move_by)(target, speed)
                print("Move accepted. Use status, or hold to stop before the target.")
            elif name in ("hold", "quit") and not args:
                s = motor.hold()
                print(f"Stopped: {s.coordinates} A={s.a:.3f}. A is configured to remain enabled.")
                if name == "quit":
                    return
            elif name == "status" and not args:
                s = motor.link.status()
                relative = "unset" if motor.reference is None else f"{motor.angle(s):.3f} degrees"
                print(f"{s.state}; {s.coordinates} A={s.a:.3f}; session angle={relative}")
            elif name == "settings" and not args:
                settings = motor.firmware_settings()
                print(f"Requested session speed: {motor.default_speed:g} deg/s")
                print(f"Firmware $113: {settings.get(113, 'unreported')} deg/min; "
                      f"$123: {settings.get(123, 'unreported')} deg/s²")
            elif name == "help" and not args:
                print(HELP)
            else:
                print("Unknown command or arguments; enter help")
        except KeyboardInterrupt:
            print("\nRequesting hold...")
            s = motor.hold()
            print(f"Stopped at controller A={s.a:.3f}; enter quit to exit")
        except EOFError:
            motor.hold()
            return
        except (ValueError, ControllerError) as exc:
            print(f"Not executed/confirmed: {exc}")
            if getattr(motor.link, "fault", None):
                raise


def main():
    import sys
    if "--dual-axis" in sys.argv[1:]:
        from control_ax import main as dual_main
        return dual_main([arg for arg in sys.argv[1:] if arg != "--dual-axis"])
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", help="Exact USB serial port; disconnect gSender first")
    p.add_argument("--list-ports", action="store_true")
    p.add_argument("--debug", action="store_true", help="Print USB commands and replies")
    p.add_argument("--diagnose", action="store_true", help="Read firmware/settings/status and exit without moving or configuring")
    p.add_argument("--configure", action="store_true", help="Persist A keep-enabled only; no speed/acceleration changes")
    p.add_argument("--min-angle", type=float, default=-360)
    p.add_argument("--max-angle", type=float, default=360)
    p.add_argument("--a-units-per-degree",type=float,default=1/3,
                   help="A controller units per physical degree; default 1/3 from measured 45 -> approximately 135 travel")
    p.add_argument("--speed", type=float, default=5, help="Initial requested speed in deg/s; not a cap")
    args = p.parse_args()
    if args.diagnose and args.configure:
        p.error("--diagnose is read-only; do not combine it with --configure")
    if args.list_ports:
        from serial.tools import list_ports
        for item in list_ports.comports():
            print(f"{item.device}\t{item.description}")
        return 0
    if not args.port:
        p.error("Choose --port or --list-ports")
    link = None
    motor = None
    try:
        link = SerialLink(args.port, debug=args.debug)
        if args.diagnose:
            print("Read-only connection diagnostic; no motion or configuration commands.")
            for line in link.command("$I"):
                print(line)
            settings = parse_settings(link.command("$$"))
            for key in (1,4,13,37,103,113,123,376):
                print(f"${key}={settings.get(key, 'not reported')}")
            s = link.status()
            print(f"Connection responds: {s.state}; {s.coordinates} A={s.a:.3f}")
            return 0
        motor = Motor112(link, args.min_angle, args.max_angle, args.speed,args.a_units_per_degree)
        changes = motor.initialize(configure=args.configure)
        for change in changes:
            print("Persistent controller change:", change)
        print(f"A steps/degree: {motor.settings[103]:g}; verify physical angular calibration.")
        print(f"A enable polarity ($4): {motor.settings.get(4, 'unreported')}; verify driver wiring.")
        print(f"Requested speed: {motor.default_speed:g} deg/s; firmware $113={motor.settings[113]:g} "
              f"deg/min and $123={motor.settings[123]:g} deg/s² are unchanged.")
        print("No motion commanded. Enter zero at your supported reference position.")
        interactive(motor)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        reason = str(exc).strip() or type(exc).__name__
        if link is None:
            print("The USB controller connection could not be opened.")
        elif args.diagnose or not link.motion_command_attempted:
            print("No move commands were sent by this run.")
        else:
            try:
                link.realtime(b"\x85")
                print("Jog cancel sent; physical stopping/holding is not confirmed. Support the arm.")
            except Exception as cancel_exc:
                print(f"Could not send jog cancel: {cancel_exc}. Use the physical stop and support the arm.")
        print(f"ERROR: {reason}")
        return 1
    finally:
        if link is not None:
            link.close()


if __name__ == "__main__":
    raise SystemExit(main())
