# Four-phase A-only toss

The old dip/rebound trajectory is replaced. X is not commanded. Edit
`toss_config.json`; it is reloaded before each move/toss. All angles are absolute
physical degrees from your saved neutral position. Positive must correspond to
upward motion on your build; reverse angle signs together if necessary.

| Phase | Target | Move time | Hold after arrival | Purpose |
|---|---:|---:|---:|---|
| position | -3° | 0.22 s | 0.18 s | Tilt down and allow sliding |
| launch | +20° | 0.32 s | 0 s | Accelerate upward and brake to attempt release |
| catch | +30° | 0.22 s | 0.15 s | Continue upward to meet the descending pancake |
| return | 0° | 0.40 s | 0 s | Return to neutral after settling |

These are unvalidated starting values, not an optimized or guaranteed flip.
A 3° tilt may not overcome static friction; video determines whether the pancake
actually slides. The launch/airborne/catch window matters more than making the
entire sequence less than a second. Nominal total is 1.49 s, including positioning,
settling and return. Launch plus catch motion is 0.54 s. No automatic time cutoff.

One rotating joint couples pan height and tilt. A catch at +30° is not level;
change its target/time based on video and your geometry. A perfect airborne catch
cannot be calculated from these angles alone. No camera feedback is used here.

## Acceleration: fix the controller ramp, not just the feed

The former program used firmware acceleration as proof of motor capability.
That was incorrect. With the current calibration, $123=1000 corresponds to about
20,006 physical deg/s². A lower feed does not lower that initial acceleration.

`acceleration_deg_s2` is initially **1000 physical deg/s²**, a provisional
commissioning ceiling, not a measured safe capability. With calibration
0.049984558 this corresponds to `$123=49.984558`:

    firmware $123 = physical acceleration * a_units_per_degree

The program blocks move/toss if firmware acceleration exceeds the configured cap.
By default it writes no settings. This explicit launch option lowers $123 to the
cap, reads it back, and refuses motion if the readback is too high. It never raises
acceleration, changes speed settings, changes hold current, unlocks, or resets.
**The $123 change persists on the controller after this program exits.**

```sh
cd /Users/andrewdai/Programming/Awesome-Flipping-Robot/robot/motor_control
.venv/bin/python simple_flip.py --port /dev/cu.usbmodemYOUR_PORT --apply-acceleration
```

First resolve any E-stop/driver fault, support the arm, and re-establish physical
neutral. Do not bypass the stop circuit or keep retrying after a driver alarm.
Alarm 10 indicates an E-stop assertion, not proof of acceleration overload; check
the external driver's fault indicator and alarm wiring to distinguish causes.

```text
zero
toss
```

Or override the four angles while retaining configured phase times:

```text
toss -3 20 30 0
```

**Changed syntax:** the fourth angle is now RETURN, not another intermediate
angle with an extra implicit return. There is no automatic fifth move. `toss 20`
is removed. Plain `toss` uses all four JSON targets. `move ANGLE` remains at its
existing 60°/s request, capped by firmware speed. `zero` saves reference without
motion. Semicolons work; Ctrl+C cancels and exits.

## Tuning deliberately

1. Test with a cold pan. Confirm direction, retention, actual endpoint and driver
   fault indication after small `move` commands. An open-loop position report
   does not establish that the arm physically tracked the command.
2. Reduce acceleration further if it cannot track at this provisional ceiling;
   restart with `--apply-acceleration`. Increase move times to remain feasible.
   Do not increase current beyond the motor/driver specification.
3. Tune only position angle/hold until the pancake consistently slides to the
   desired part of the pan; then tune launch angle/time for release.
4. Tune catch angle/time from slow-motion video, then catch hold and return time.
   Longer launch time reduces peak speed, but does not reduce firmware acceleration.
5. Raising the JSON acceleration cap alone does not raise firmware acceleration.
   Higher values need physical qualification; the program never automatically
   raises them to satisfy short move times.

Each movement is planned as a rest-to-rest firmware ramp. The program solves
`T = distance / peak_speed + peak_speed / acceleration`, checks the firmware
speed ceiling, and rejects impossible times for the entire toss before moving.
Each phase waits for controller Idle/commanded endpoint, then holds, then begins
the next. A zero-distance phase consumes its move time as a dwell. Host polling
and serial latency add time; these are not hard real-time flight timestamps.
A stalled motor may still report the commanded endpoint. Holds keep existing
motor-enable settings; they are not brakes or evidence of sufficient holding torque.

All checks for this update are offline fake-controller tests. No robot, physics
simulation, or training was run.
