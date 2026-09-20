# Current sequence: 0 → +45 → -75 → +45 → 0 degrees

Reload with R. From saved physical level zero, F/Space queues all four moves with
no programmed pauses and finishes at zero. Up/Down nudges and Z save-zero remain
available. Speed and acceleration settings are unchanged.

# Current sequence: 0 → +25 → -80 → +25 → 0 degrees

Reload with R. From saved physical level zero, F/Space queues all four moves with
no programmed pauses and finishes at zero. Up/Down nudges and Z save-zero remain
available. Speed and acceleration settings are unchanged.

# Current sequence: 0 → -10 → +35 → -80 → +35 → -10 → 0 degrees

Reload with R. From saved physical level zero, F/Space queues all six moves with
no programmed pauses and finishes at zero. No P setup is required. Up/Down nudges
and Z save-zero remain available. Speed and acceleration are unchanged.

# Current sequence: -10 → +40 → -80 → +40 → -10 degrees

Reload with R. Keep zero referenced to physical level; P positions the arm at
-10 degrees, then F/Space runs four queued moves without programmed pauses. It
finishes at -10, ready for another F. Up/Down nudges and Z save-zero still work.
Speed and acceleration settings are unchanged.

# Current sequence: 0 → -10 → +35 → -80 → +35 → -10 → 0 degrees

Restart the app. F/Space queues all SIX moves with no programmed pauses and ends
at zero. Speed and acceleration settings are unchanged; no P setup is needed.

Up/Down arrow keys nudge the arm by +1/-1 physical degree while idle (positive
means the calibrated positive A direction). Press Z afterward to save the adjusted
pose as zero. Nudges use at most 10 deg/s, ignore flip tempo, and obey travel and
acceleration limits. `zero_nudge_deg` configures the step size (up to 5 degrees).
Arrows never silently redefine zero; if this is initial setup, F stays disabled
until Z. Keys during a flip are ignored rather than queued. X still stops.
When editing a numeric box, finish with Enter before using the arrows for motion.

In six-part mode, Catch controls BOTH -10 approach/exit angles, Launch controls
both +35 angles, Slide controls -80. Start and final level are zero.

# Current sequence: -10 → +35 → -80 → +35 → -10 degrees

Reload with R (or restart). Keep Z referenced to physical level, then press P to
position at the -10-degree start. F/Space queues all four moves without programmed
pauses and finishes at -10, ready for another F. Speed and acceleration settings
are unchanged. Do not redefine zero at -10 degrees.

# Current sequence: 0 → +35 → -90 → +35 → 0 degrees

Reload with R (or restart). With Z saved at physical level, F/Space queues all
four moves without programmed pauses. No P setup is needed. Speed and acceleration
settings remain unchanged. Confirm mechanical clearance to -90 degrees before
operation. It finishes at level. Launch angle controls both +35 endpoints; Slide
controls -90; Catch controls final zero.

# Current sequence: 0 → +30 → -85 → +30 → 0 degrees

Restart. Z saves physical level as zero, then F/Space queues all FOUR moves:
+30, -85, +30, 0. No separate P setup is required. No dwell, wiggle, settling hold,
or extra return is inserted. Existing speed and acceleration settings are preserved.
All reversals still obey firmware braking and acceleration. Confirm -85 degrees
of travel is clear before operation. It finishes at zero, ready for another F.

T's Launch angle controls both +30 endpoints, Slide controls -85, and Catch
controls the final 0 endpoint. `four_part: true` enables this arrangement inside
the queued timed mode. V's calibrated vision workflow remains separate.

# Current sequence: +25 → -80 → +25 → 0 degrees

Restart. Z saves **physical level as zero**. Press **P** to position at +25 degrees,
then **F/Space** executes exactly three queued moves: -80, +25, 0. No holds,
wiggle or additional return. Maximum allowed feed; existing acceleration unchanged.

P is a separate setup move, not part of the throw. The app refuses a three-part
throw unless it starts at the configured +25 degrees. It finishes at level 0;
press P again before the next F. Confirm clearance for the full -80-degree travel.
Do not redefine zero at +25 degrees. `start_angle_deg` controls the required start.

# Current timed flip: exactly three parts

Restart the app, save Z with the arm physically at level 0, then press F/Space:

1. 0 → -75 degrees
2. -75 → +25 degrees
3. +25 → -75 degrees

All three jogs are queued immediately at the highest feed allowed by the current
application and firmware speed limits. No wiggle, dwell, camera wait, catch hold,
or automatic return is inserted. Each reversal still requires firmware braking
and re-acceleration. Acceleration and firmware speed settings are not increased.
The arm FINISHES AT -75 degrees. N returns to zero before the next trial; do not
press Z at -75 to restart, since that would change the physical reference.
Confirm the expanded -75-degree travel is mechanically clear before operating.

`three_part: true` enables this mode. T edits its three angles via Slide, Launch,
and Catch, and the speed ceiling. Move/hold times, tempo and wiggle do not affect
this mode. Other diagnostic modes remain separate. X cancels the queued motion.

# Queued launch/catch update

Timed F/Space flips now queue the launch and catch jogs back-to-back, without
waiting for Idle between them. The slide/wiggle/positioning hold still happens
first; catch settling and return happen afterward. No speed, angle, timing or
acceleration defaults were changed by this update. Set `queued_launch_catch`
false to compare the previous separate-phase behavior.

Same-direction phases may blend in the firmware. Reversals still require braking;
this is not an instantaneous torque impulse. A blended launch point is no longer
a full-stop release point, so release/catch timing must be retuned with video.
The launch/catch phase durations are estimates, not exact timestamps when blended.
The queued commands can be cancelled together with X; faults prevent return/retry.
Camera-guided V still observes after its launch instead of precommitting a catch.

At 1000 deg/s², a 40-degree rest-to-rest stroke takes at least 0.40 s and peaks at
200 deg/s. Increasing the speed ceiling above that does not speed up that stroke.
Removing the intermediate stop helps, but cannot overcome motor/acceleration
limits. No acceleration increases or emergency-stop bypasses are included.

# Signed controls, faster tempo, slide wiggle

Restart the app. T opens a separate, readable controls panel. Click a numeric box,
type a signed value such as **-8**, and press Enter. No offset conversion: displayed
angles are actual signed degrees. Bars and +/- buttons also work. The camera's
status/messages are wrapped underneath its image, without covering the picture.

The new speed ceiling starts at **600 deg/s**, and **Tempo 2** requests half the
configured movement times. Holds keep their configured durations. Your existing
angles and individual phase times are preserved. Tempo and the speed ceiling are
editable in the panel; actual motion remains capped by current firmware speed and
acceleration. Too-short timing requests are stretched to the minimum feasible ramp
and logged as `timing_limited`, rather than increasing acceleration or refusing the
entire flip. This is a configured motion estimate, not verified motor capability.
No automatic acceleration increases were added.

Slide now does a **1-degree, one-cycle wiggle**: slide target + 1, target - 1,
then back to the slide target, before its hold and upward launch. Amplitude,
cycle count, and wiggle move time are editable. Set amplitude or cycles to 0 to
disable. All extrema must fit joint limits; invalid settings are rejected before
motion. The wiggle may help release a pancake but must be tuned physically.

# Direct flip controls (updated)

**Z** saves physical neutral. **F / Space / L** executes your configured sequence:
slide down → hold → launch upward → move up to catch → settle → neutral.
**T** opens angle and time sliders. Times are milliseconds; angle sliders show
an offset from the configured minimum, while the main window shows signed targets.
Changes save automatically while idle. **S** tests slide/hold only; **N** returns.

Timed flips and slide tests do not require clicked image targets, intrinsics,
markers or pancake detection. The camera continues recording. The operator must
check the pancake is positioned before pressing flip. **V** selects the separate
calibrated camera-guided mode, which retains its vision readiness checks.

On the first movement request, the app automatically lowers excessive firmware
acceleration to your configured ceiling, while Idle, and verifies readback before
moving. It NEVER raises acceleration or changes firmware speed/current settings.
This $123 change persists on the controller. No need to switch to gSender.
The 1000 deg/s² ceiling is still provisional, not a measured motor capability.
Controller alarms, stop inputs and impossible movement times still stop the trial.

Restart the app to load this update. Existing angles, times, axis calibration and
speed defaults were preserved. The panel edits slide angle/time/hold, launch
angle/time, catch angle/time/hold and return time. Camera-guided calibration below
is needed only for V, not for ordinary flips.

Independent application: existing demo, motor programs and training code are
unchanged. No learning model is required. This is an experimental trial recorder
and controller, not a claim of demonstrated autonomous catching.

## Intended movement

1. **Slide:** tilt down slowly, hold at least 180 ms, then wait until the pancake
   center stays inside a clicked edge target for 120 ms. If it never gets there,
   stop the camera-guided trial before launch. Timed mode uses the configured hold only. The target must be INSIDE the usable pan surface,
   with enough room for the entire pancake; do not click the literal rim.
2. **Launch:** execute one bounded upward acceleration/braking motion.
3. **Catch:** in vision mode, fit planar ballistic flight from at least four
   consecutive airborne observations and choose a reachable stationary pan pose
   on the arm's arc. Send ONE catch motion, hold through predicted landing, settle.
   No accumulating chase commands. If data are missing/stale or interception is
   impossible, execute the configured timed catch and label it **fallback**.
4. **Return:** return to neutral. Inspect/reset pancake and press F or L again.
   Rate every trial manually. Seeing a pancake in the pan does not prove a flip.

"Position vertically" means raise the pan center along the arm's circular arc;
it does not mean rotate the pan to 90°. Pan height and tilt are coupled. It cannot
translate straight up independently, level the pan independently, or correct
sideways drift. There may be NO reachable catch for a particular launch.

## Install and launch

Use an isolated environment so the existing detector/demo dependencies stay intact:

```sh
cd /Users/andrewdai/Programming/Awesome-Flipping-Robot
uv venv --python 3.13 .venv-workshop
uv pip install --python .venv-workshop/bin/python -r robot/flip_workshop/requirements.txt
.venv-workshop/bin/python robot/flip_workshop/app.py --source 0
```

Watch mode opens a camera and records observations/video; it never opens a serial
port. Close demo.py/the earlier camera window before opening the same camera.
Use `--source oak` for OAK-1. Recorded files are watch-only, never motor inputs.

To enable motor keys explicitly, close gSender and run:

```sh
.venv-workshop/bin/python robot/flip_workshop/app.py --source 0 --live --port /dev/cu.usbmodemYOUR_PORT
```

The app lowers excessive firmware acceleration as described above; it does not change speed defaults in other programs, current,
encoder wiring, or hold-enable settings. A scale remains **0.049984558**. It
requires the existing SLB A hold setting and a working supported mechanism.

## Controls

- Left-click pancake: calibrate HSV color using the repository's detector.
- Double-click the pancake's intended resting center: mark ready region (blue).
- Right-click desired pancake center near the leading edge: mark slide target
  (magenta). Image targets remain fixed because camera and pan are rigidly linked.
- **Z:** save motor zero at the known physical neutral pose, no movement.
- **S:** slide test only; leaves arm at slide angle. **N:** return to neutral.
- **F / Space / L:** complete timed flip and fixed catch; no image targets or depth calibration needed.
- **C:** test configured catch position/hold only. N returns afterward.
- **V:** complete camera-guided flip; calibration and fresh world observations required.
- **M:** save neutral camera mounting pose from table marker. Do this only at physical zero.
- **R:** reload config when idle. Files/edits are frozen for each executing trial.
- **0:** miss; **1:** catch without confirmed flip; **2:** verified opposite side down.
- **X:** request jog cancellation; **Q/Escape:** cancel and close.

Software stop depends on serial responsiveness; keep the physical stop accessible.
No automatic unlock/retry follows a controller fault. A software-reported angle
is commanded motion, not encoder verification. Support/re-reference after a fault.
The app permits repeat trials by keypress; it deliberately does not run an
unattended loop that could toss an absent or unseated pancake.

## Parameters and acceleration

All workshop variables live in `config.json`; edit and press R:

| Phase | Angle | Move | Hold |
|---|---:|---:|---:|
| Slide | -8° | .40 s | .18 s minimum + camera positioning gate |
| Launch | +20° | .36 s | 0 |
| Timed catch fallback | +30° | .25 s | .30 s |
| Return | 0° | .50 s | 0 |

These are commissioning examples, not measured motor limits or a successful toss.
New-app speed cap is 180°/s and acceleration ceiling is 1000°/s²; existing app
speeds are untouched. The controller's actual acceleration must be AT OR BELOW
this ceiling; the app lowers it and verifies readback before motion. The program never silently increases it to
satisfy a requested duration. Lower feed alone does not lower acceleration.

With current axis scale: `$123 = physical_deg_s2 * 0.049984558`.
For the provisional 1000°/s² ceiling: `$123=49.984558`. The app applies this downward change automatically at Idle. Resolve any
driver/E-stop fault and verify axis calibration first. Lower it further if the loaded
motor cannot track; lengthen move times accordingly. This is not proof that the
motor can deliver its rated holding torque at launch speed. Alarm 10 denotes an
E-stop assertion; inspect driver faults/wiring rather than assuming acceleration.

Tune ONE stage at a time: edge position/slide dwell; launch angle/time; catch
angle/time. Use cold trials and video. Increasing slide tilt may help sticking,
but cannot guarantee controlled sliding. Shortening launch time raises required
peak speed; it does not change firmware acceleration. Every phase is rest-to-rest
with acceleration/braking included; host/serial delays add time. Camera detection
is not performed inside the microcontroller motion loop.

## Camera-guided interception: what must be measured

A top-down pixel offset is NOT a 3D airborne location. A flipping pancake's
apparent width is also not a reliable depth gauge. This implementation uses a
**planar-flight assumption**, a known reference marker fixed to the table, and
camera calibration. Sideways drift violates the model and is not observable
reliably from this single-view constraint. Do not claim arbitrary 3D tracking.

1. Calibrate camera intrinsics using a measured chessboard. Counts are INNER
   corners (default 9x6). Capture >=15 varied positions/tilts with SPACE; C fits.
   Do not change focus, crop or camera mode afterward. The pipeline uses the
   detector's max-640-pixel image width throughout; dimensions must match.

   ```sh
   .venv-workshop/bin/python robot/flip_workshop/calibrate_camera.py --source 0 --square-mm 20
   ```

2. Generate a table marker:

   ```sh
   .venv-workshop/bin/python robot/flip_workshop/app.py --make-marker /tmp/flip-marker.png
   ```

   Print the BLACK marker square at exactly 80 mm; retain the white border. Lay
   it flat on the table, fully visible throughout motion. Marker center is world
   origin; X points right on the printed marker, Y points toward its top, Z points
   upward off the paper. Align X with the arm's neutral outward direction. The
   joint axis runs parallel to Y. Positive arm angle raises the pan along +Z.

3. Measure `geometry.joint_world_m = [x,y,z]` from that marker center to the joint
   pivot, and `pan_center_from_joint_m = [x,y,z]` from pivot to the pan's INNER
   FLOOR center at neutral. Enter metres. Neutral pan floor must be horizontal.
   Your pan outer diameter is **304.8 mm (12 inches)** and measured camera-to-pan
   center distance is **355.6 mm (14 inches)**. Measure the usable inner FLOOR
   radius: it is deliberately unset rather than treating the rim as catch area.
   Pancake radius defaults to 50 mm; measure and adjust it too.
   Marker size and all camera/pan mounts must be rigid and unchanged.

4. Mount the camera with a SIDEWAYS offset from the swing plane, aimed to keep
   marker and flying pancake visible. At least 30 mm is required by the
   reconstruction, but larger offsets/angles generally improve conditioning.
   A camera sitting exactly in the swing plane gives a degenerate ray-plane
   intersection, even if it clearly sees the pancake. Pure top-down centerline
   placement cannot provide this algorithm's airborne depth.

5. At physical neutral save motor reference with Z, then camera mounting pose
   with M. This creates `mount.json`. Verify reconstructed marker camera pose,
   known test-point coordinates and angles at several stationary arm positions
   against measurements; confirm rigid-rotation consistency and detector accuracy. Camera-to-pan distance must
   agree with the measured 355.6 mm within 30 mm; two lengths alone do not define
   the full mount orientation or pivot position.
   Inspect `events.jsonl` world_m/angle_deg/projection_error values in watch mode.
   Only then set `calibration.verified` true and press R. M invalidates this flag.

6. For a USB webcam, measure exposure-to-host latency (e.g. camera view of a
   timestamped flashing LED/display compared to its host event) and enter
   `tracking.webcam_latency_s`. Without this, vision catching is blocked. A
   single constant is approximate; short exposures and stable buffering matter.
   OAK uses its synchronized frame timestamps to measure age instead. Motion blur,
   rolling shutter and pose jitter remain limits. No timing claim follows from FPS.

The fixed marker estimates camera pose in each SAME image as the pancake, avoiding
an unsynchronized commanded-angle transform. ArUco pose ambiguity, poor reprojection,
non-rigid motion, near-parallel rays and missing marker all invalidate world points.
The original color/ellipse detector can lose a pancake when it turns edge-on; this
is treated as missing data, not a hallucinated position. Thin/folding pancakes and
perspective-centroid bias can still produce incorrect detections.

After launch, the algorithm fits gravity-constrained ballistic motion from at
least 4 consecutive points spanning >=60 ms, rejects residuals over 12 mm, and
searches catch angles 20–35°. It checks descending intersection with the pan plane,
whole-pancake footprint clearance, motor travel time and a 60 ms dispatch margin.
It holds at that target until predicted impact plus settling. It does NOT predict
flip orientation, deformation, adhesion, or collision while the pan is moving.
A reachable mathematical intercept is therefore not proof of a successful catch.
Measured camera + detection + command + motor response must fit the flight window;
otherwise use timed mode or change mechanics/camera rather than pretend feedback works.

## Evidence, repeatability and tests

Each session writes `data/logs/flip_workshop/<timestamp>/events.jsonl` and
`camera.avi`. Every processed frame has its receive/exposure time, video index,
detection, world reconstruction/error and phase. Trial events snapshot all settings,
limits, target commands, predictions, fallback reason and your rating. The AVI is
encoded at nominal 30 FPS; use JSONL times for analysis, not playback frame rate.
Repeat exactly the same pancake setup and change one parameter between groups.

Offline checks (no motor, camera, physics engine or training):

```sh
.venv-workshop/bin/python -B -m unittest discover -s robot/flip_workshop -p 'test_*.py' -v
```

References: [OpenCV calibration](https://docs.opencv.org/4.8.0/d9/d0c/group__calib3d.html),
[Luxonis timestamps/latency](https://docs.luxonis.com/software-v3/depthai/tutorials/optimizing),
[Sienci controller manual](https://resources.sienci.com/view/slb-manual/?print=print).

Compare repeated settings and explicit ratings:

```sh
.venv-workshop/bin/python robot/flip_workshop/report.py
```

The report groups exact settings and mode, distinguishes commanded vision catches
from operator-confirmed flips, and shows unrated/incomplete attempts separately.
