# Pancake arm angle controller

A is shoulder; X is wrist. There is no loaded mode, mandatory acceleration
profile, or application speed cap. Launch with your verified calibration and port:

```sh
.venv/bin/python control_112.py --dual-axis --port /dev/cu.usbmodem2083337842301 --a-units-per-degree YOUR_A_SCALE --x-units-per-degree 0.022222222222
```

YOUR_A_SCALE is not yet verified. X assumes the previously reported direct-drive
32-microstep setup with $100=800. Do not reuse A's earlier conditional example
unless its driver pulse setting matches.

Existing firmware acceleration and maximum speeds remain effective; both are
printed in physical degrees at startup. Earlier persistent slow settings are NOT
automatically undone by this software update. --configure alone manages motor
holding enable. Only an explicit --acceleration VALUE --configure changes ramps.
There is no defensible automatic "unlimited" setting for a loaded stepper.

Commands:
```
zero
x 10 2
status
a 20 5
hold
quit
```

Syntax is JOINT ANGLE SPEED, in degrees and degrees/second. Wait for Idle before
the next move. Targets wrap modulo 360 via the shortest path; 360 equals 0.
`x by 360 5` explicitly requests a full turn. No automatic collision checking.

The SLB performs timing locally. Configuration preserves running/holding current
and maximum rates and keeps A/X enabled. An explicit --acceleration converts
joint acceleration to $120/$123.
Previously configured maximum rates remain effective and print at startup.
Speed commands are subject to these firmware ceilings and acceleration ramps.

AI callers use an initialized ArmAX instance with an established session zero:
```python
arm.move_to_angle("X", angle=10, speed=2)
arm.wait_until_idle(timeout=10)
arm.move_to_angle("A", angle=20, speed=5)
arm.wait_until_idle(timeout=10)
```
This is one-joint-at-a-time angle control, not synchronized toss execution.
Idle means execution ended, not encoder-verified physical arrival. X has no
encoder feedback; A's encoder is local to its driver. A timeout requests jog
cancellation; faults propagate and movements must not be retried automatically.

Read settings with --diagnose or the settings command. After a new measured
calibration trial, `calibrate a 5 7` corrects the current scale if 5 commanded
degrees actually travelled 7. Re-zero, save the printed scale, and restart and explicitly reapply --acceleration if changing the conversion. A motor that slips needs a known
physical reference restored before zeroing again.
