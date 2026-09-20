# Motor control for the pancake robot

Start with [the current A/X guide](DUAL_AXIS.md). A is the shoulder and X is the wrist.

From this folder, run `.venv/bin/python control_112.py --dual-axis --port /dev/cu.usbmodem2083337842301 --diagnose` for a read-only hardware diagnostic.

The following older A-only notes are retained for reference; the A/X guide takes precedence.

# 57HYB112: rotate and hold on an SLB A axis

This is a separate manual controller; it does not modify the CAD, simulation, training code or existing robot configuration files. No motor was connected or operated during development.

The assumed connection is **laptop USB → SLB external A-axis STEP/DIR/ENABLE → 57HYB112 integrated driver**, with the motor's separate power connection and its internal encoder intact. The SLB's four-pin X/Y/Z winding-power outputs are not compatible with the motor's eight-pin signal connector.

## What “hold” means

The integrated driver keeps the motor at the last commanded step position using its internal encoder while enabled. The Python program does not estimate torque or run a feedback loop on the laptop. It supports:

- `to ANGLE`: rotate to an angle relative to a session zero.
- `by ANGLE`: rotate by a specified amount from the current controller position.
- `hold`: decelerate and cancel the current move, then remain energized at the stopping position.
- `speed VALUE`: set the requested speed for subsequent moves in degrees/second, with no application cap.
- `settings`: read the firmware's rate and acceleration values without changing them.
- Reaching a target normally also leaves the motor enabled, without needing a separate hold command.

`hold` uses grblHAL **jog cancel (`0x85`)**, so the interrupted target is discarded. It does not send cycle-start, reset, motor-disable or alarm-unlock commands. A stop takes finite time/distance. Controller position reports are commanded step positions, not direct readings of the motor's internal encoder. Alarm/power loss can remove holding torque, and holding strength is limited by the actual driver/motor/load.

For the **A shoulder + X wrist**, use the new [two-axis guide](DUAL_AXIS.md) and launch this script with `--dual-axis`. The A-only instructions below remain available.

## Set up

1. Support the arm mechanically. First verify direction and angular calibration with the arm/pan load removed or supported.
2. In gSender, verify that independent A-axis operation already works. Enable A as a rotational axis (`$376`, A bit). For Sienci's standard closed-loop A wiring, its guide calls for **A enable-pin inversion** (`$4`, A bit on); verify your actual wiring. This script does not change enable polarity, disable limits or unlock/home the board.
3. Verify `$103` is **steps per degree**, matched to the driver DIP switches and any gearing. For a 1.8° motor at 1/16 microstepping and no reduction, the example is `200 × 16 / 360 = 8.888889 steps/degree`; do not use that example unless it matches your motor setup. The Vortex kit's calibration includes its own transmission and is not a direct-drive robot setting.
4. Use metric reporting (`$13=0`) and a positive step-idle delay (`$1`, normally 254 on SLB). The script rejects zero idle delay because it can defeat persistent enable on some firmware.
5. Disconnect gSender. Only one application should control the SLB serial connection.
6. Install Python 3.10+ and the single dependency in a dedicated environment:

```sh
cd /Users/andrewdai/Programming/awesome-flipping-robot/robot/motor_control
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python control_112.py --list-ports
```

Use the exact port from that list. Example only:

```sh
.venv/bin/python control_112.py --port /dev/cu.usbmodemYOUR_PORT --configure
```

`--configure` makes one persistent controller change when necessary:

- Adds the A bit to `$37` (`existing_value | 8`) so A stays enabled at idle, preserving all other axis bits.

It **does not modify `$113` (speed), `$123` (acceleration), or `$103` (angle calibration)**. There is no application speed or acceleration ceiling. Positive finite speed requests are sent as requested; the controller applies its own configured limits and may run slower than requested.

It prints the old/new settings when changed. Save those values if you plan to restore them later. Enabling a closed-loop driver may make it correct a position error; support the arm before using configuration. `$1`, `$4`, axis calibration, homing and hardware limits are not modified.

On subsequent runs omit `--configure`. The script requires the hold bit to already be configured. It never restores settings to a de-energized state on exit. USB resets, firmware faults and power loss can still remove holding; do not rely on a laptop connection as a mechanical support.

**If you ran the previous version with `--configure`, it may already have changed `$123` to 30.** This version does not guess or restore the prior value. Check `settings`, then use gSender to restore your chosen acceleration if needed. The old `--max-speed` and `--max-acceleration` arguments have been removed.

## Operate

Place the arm at the intended supported reference position, then enter:

```text
zero
speed 60
to 10
status
hold
by -5 90
status
hold
to 0
hold
quit
```

The `112>` terminal stays running until `quit`; a move returns to the prompt as soon as the controller accepts it. You can enter `hold`, `status`, `settings` or `speed` while it moves. `speed` affects subsequent commands; it does not change a move already running. A speed supplied on `to` or `by` overrides the session speed for that move only.

Issue the next move only after the previous one finishes, or after `hold`. Commands do not queue multiple movements. `zero` labels the current position as 0° in Python only; it does not move the arm, home the controller or write G92/work offsets. `to 0` physically returns to that session reference.

Defaults are ±360° from session zero and a **5°/s initial requested speed, not a cap**. Use `--speed VALUE` at startup or `speed VALUE` in the terminal to choose another default. Acceleration comes entirely from the existing firmware `$123` setting. The existing angle bounds remain configurable with `--min-angle` and `--max-angle`; choose a collision-free range for the actual arm. Firmware speed, acceleration and soft/hard limits still apply.

For example, use a ±15° starting range:

```sh
.venv/bin/python control_112.py --port /dev/cu.usbmodemYOUR_PORT --min-angle -15 --max-angle 15
```

Ctrl+C requests hold and returns to the prompt. `quit` first confirms Idle, then disconnects while leaving the A hold setting enabled. If communication or stopping cannot be confirmed, the program reports that explicitly. It makes no automatic attempt to retry movement, clear an alarm or resume after a reset.

The SLB continues executing an already accepted bounded jog if the laptop disappears before sending cancel. Use the physical stop and a passive rest/catch suitable for a gravity-loaded arm. This manual tool is not a fail-safe robot supervisor or a torque-control mode.

## Python API

`Motor112.move_to(degrees, speed)`, `move_by(degrees, speed)` and `hold()` are available for later integration. Construct `SerialLink`, call `Motor112.initialize()`, then explicitly call `zero()` at the known reference. One caller must own the connection; the API is not thread-safe.

## Verification

If opening the port succeeds but the program times out, completely disconnect/quit gSender and retry. The controller now asserts **DTR** when opening native USB and allows 350 ms for the connection to settle. Upstream STM32 grblHAL treats DTR as the host-connected indication; the earlier DTR-low setting could leave this program without a usable connection.

For a read-only connection check, use your actual port:

```sh
.venv/bin/python control_112.py --port /dev/cu.usbmodemYOUR_PORT --diagnose --debug
```

This prints transmitted requests and received replies, reads firmware/settings/status and exits. It does not configure holding, move, reset, unlock, or send jog cancel. Do not combine it with `--configure`. Regular interactive operation also accepts `--debug`. Timeouts identify the request and last reply; a startup failure no longer claims that a jog was cancelled when this run never sent a move.

Run the protocol/command tests without pyserial or hardware:

```sh
python3 -m unittest discover -s robot/motor_control -p 'test_*.py' -v
```

Run that command from the main project directory. Tests use a fake controller; they do not validate actual shaft motion, torque, holding, wiring or physical stopping distance.

## References

- [Sienci SLB manual: A-axis wiring and motor holding](https://resources.sienci.com/view/slb-manual/?print=print).
- [Sienci closed-loop A-axis setup](https://resources.sienci.com/view/vx-closed-loop-motor/): A `$4` and `$37` enable settings; do not copy its geared Vortex angle calibration into this direct-drive arm.
- [SLB settings](https://resources.sienci.com/view/slb-firmware-flashing/): `$103`, `$113`, `$123`, `$376`.
- [grblHAL jog-cancel command](https://github.com/grblHAL/Plugin_keypad): `0x85`.
- [grblHAL stepper enable behaviour](https://github.com/grblHAL/core/blob/master/stepper.c): idle enable mask and zero-delay behaviour.
- [grblHAL STM32 USB connection detection](https://github.com/grblHAL/STM32F4xx/blob/master/Src/usb_serial.c): connection requires asserted DTR and a line-state settling interval.


### A-axis correction for measured 45° → approximately 135° travel

The command-line program now defaults to `--a-units-per-degree 0.333333333333`.
X conversion is unchanged. The former ±45° bounds were only limits, not a scaling factor.
The old A conversion assumed one controller unit equalled one physical degree; the
observed travel indicates about three physical degrees per controller unit. This
software correction divides A travel and feed by three, and converts status back
to estimated physical degrees. It does not change firmware settings or measure the shaft.

Launch using your previous command (remove any old ±45 angle-limit overrides if
full-turn limits are wanted). Enter `zero`, then test `a by 5 2` in dual-axis mode
(or `by 5 2` in A-only mode). Measure travel before attempting a full turn.
`a to 360` means one revolution from session zero; `a to 0` returns in reverse.
The ±360° bounds do not establish mechanical or cable clearance.

Because “nearly 135” is approximate, refine with a new measured trial: if the new
5° command travels 5.4°, enter `calibrate a 5 5.4` (A-only: `calibrate 5 5.4`), then
`zero`. Calibration prints a replacement `--a-units-per-degree` argument for the
next launch. Do not apply the original 45/135 correction again to an already
corrected session.

To establish the hardware cause, compare `$103` with the external driver's actual
pulses per revolution and gearing: correct steps per output degree = pulses per
motor revolution × motor revolutions per output revolution / 360. A mismatch in
these settings can cause this factor; the observation alone cannot distinguish
incorrect `$103` from driver microstep or transmission assumptions. If firmware
calibration is fixed to true physical degrees, launch with `--a-units-per-degree 1`
to avoid correcting twice.
