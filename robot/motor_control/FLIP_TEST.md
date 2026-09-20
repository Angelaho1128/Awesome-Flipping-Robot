# Shoulder-driven pancake flip test

This is a manual, single-trial experiment, not a trained policy or guaranteed
flip. A drives the 57HYB112 shoulder; X drives the 57STH56 wrist. All recipe
angles are physical degrees relative to the explicitly established reference.
All speeds are degrees/second, matching `control_112.py --dual-axis`.
The wrist's reported speed ceiling of 30 applies only if calibration is correct.

The wrist sets a tilt slowly and holds that joint angle while the shoulder
launches and returns. It does NOT keep the pan level as the shoulder rotates:
pan pitch = reference pitch + shoulder angle + wrist angle. This avoids making
the slow wrist counter-rotate as fast as the shoulder. Shoulder torque must still
accelerate the wrist, pan and mounts; the stronger motor is not a guarantee of
launch or catch. Verify the wrist can hold its angle against the dynamic load.

## Configure

1. Disconnect gSender/any other serial controller. Use the existing environment:

   ```sh
   cd /Users/andrewdai/Programming/Awesome-Flipping-Robot/robot/motor_control
   .venv/bin/python flip_test.py
   ```

   This prints an offline preview; it never opens a serial port.

2. Edit `flip_test.json`. Set BOTH `a_units_per_degree` and
   `x_units_per_degree` from your verified current manual controller calibration.
   Null values deliberately block hardware execution. Do not blindly copy the
   historical A 1/3 correction. The previously documented X value
   0.022222222222 is conditional on the actual driver/settings and measurement.
   A positive controller movement must be checked for each joint; set direction
   +1 or -1 so positive recipe angles increase the pan's pitch. Conversion
   magnitudes remain positive. Scales include any gearing.

3. Set collision-free unwrapped joint bounds and pan pitch bounds for the actual
   mechanism. These check nominal angles, not 3-D collision geometry or motor
   shaft feedback. The default reference is a level cold pan; use the same known
   shoulder/wrist reference each time. Update `pan_pitch_at_zero_deg` if needed.

4. Existing firmware acceleration, currents and maximum speeds are NOT changed.
   The program prints calibrated firmware speed/acceleration and rejects recipes
   exceeding the firmware speed ceiling. Both motors must already be configured
   to stay enabled at idle, as in the existing A/X controller. Initialization
   gives an error if that setup is missing; this script does not silently fix it.

## Run one trial

First rehearse without a pancake, with a cold pan and physical support/stop
available. This default reduces requested speeds to 25%; it does not reduce
firmware acceleration. Confirm directions/clearance before full speed.

```sh
.venv/bin/python flip_test.py --execute --port /dev/cu.usbmodemYOUR_PORT
```

Prompts:
- `zero`: label the present physical pose as the reference; no movement.
- `prepare`: wrist tilt first, then shoulder backswing, with the printed speeds.
- `toss`: exactly one launch, shoulder return, settling delay and wrist recovery.

Nothing is repeated automatically. Ctrl+C requests jog cancellation. A failed
move aborts the remaining sequence without an automatic return or retry. A
software stop cannot guarantee support after overload, power loss or a driver
fault. Use the physical stop/support if cancellation is not confirmed.

After calibration and rehearsal, run the specified recipe speeds:

```sh
.venv/bin/python flip_test.py --execute --port /dev/cu.usbmodemYOUR_PORT --speed-scale 1
```

Initial recipe (an adjustable starting experiment, not a validated toss):

| Phase | Joint target from reference | Requested speed at scale 1 |
|---|---:|---:|
| Set wrist tilt | X -5° | 15°/s |
| Prepare shoulder | A -10° | 15°/s |
| Launch shoulder | A +20° | 60°/s |
| Catch/return shoulder | A 0° | 45°/s |
| Wait | 1 s | — |
| Recover wrist | X 0° | 10°/s |

X is held at -5° during launch/return. Release-pose pan pitch is nominally +15°
for the default level reference. `release_pause_s` defaults to zero. Wrist
requested speeds above 30°/s are rejected, even if speed-scale would reduce them.

## Tune and measure

Change one recipe variable at a time: shoulder excursion, shoulder launch speed,
catch angle/speed, release pause, then wrist preparation tilt. The launch movement
ends with firmware-controlled deceleration; the return begins only after Idle.
A large wrist flick is intentionally absent. If the shoulder moves but no launch
occurs, use video to distinguish sliding, sticking, insufficient speed and slow
stopping. Do not simply keep increasing speed after stalls or lost position.

`flip_trial.jsonl` records recipe, firmware motion settings, timestamps, controller
positions and phase durations. Video provides actual angles and flip/catch outcome:
SLB positions are commanded step counts, not physical encoder measurements.
If a motor slips, restore the known physical reference before the next trial.
Each run appends a new trial_setup event; use `--log another.jsonl` to separate runs.

This uses the existing single-axis `$J` jog protocol. Firmware executes each ramp,
but phase transitions wait for USB/status responses and a 20 ms polling interval;
there is variable turnaround latency, so this is NOT deterministic playback of
the MuJoCo trajectory. A measured successful recipe can later be converted to
controller-buffered coordinated G-code and its timing revalidated. Do not claim
this manual experiment is the trained policy. Current RL wrist speed is still
90°/s; these hardware findings must be incorporated separately before policy use.

Offline logic checks (fake controller; no serial/motors/physics/training):

```sh
.venv/bin/python -m unittest discover -s . -p 'test_flip_test.py' -v
```

## Correcting the reported A 5° command → approximately 45° movement

In the running dual-axis manual controller enter:

```text
calibrate a 5 45
```

This changes the active A conversion to **old scale / 9** and invalidates the
old software reference. Restore the known physical reference, enter `zero`, and
verify a small slow `a by 1 1` move. Use the printed replacement scale in future
manual launches; do not apply the factor again to an already corrected scale.
For the older A-only prompt the equivalent command is `calibrate 5 45`.

To store that correction in the flip recipe without connecting hardware:

```sh
.venv/bin/python flip_test.py --calibrate-a CURRENT_A_SCALE 5 45
```

Replace CURRENT_A_SCALE with the actual positive `--a-units-per-degree` value
used during that measurement. There is no reliable way to infer it from the
5-to-45 observation alone. For X, use `--calibrate-x CURRENT_X_SCALE COMMAND MEASURED`.
These commands save only the JSON conversion; firmware steps/unit and current
are untouched. Because 45° was approximate, refine the scale with another
measured trial before any fast move. A scale correction changes physical speed
conversion too; the previous apparent wrist limit must likewise use verified X
calibration.
