# Establish a launch before long training

Run from `Awesome-Flipping-Robot/training/baseten`:

```sh
baseten train push --config config_probe.py
baseten train job logs --job-id PROBE_JOB_ID --tail
```

Replace PROBE_JOB_ID with the returned ID. This executes on the cloud H100, not your laptop. The job runs regression tests, the CPU/GPU parity gate, then a finite trajectory search. It does not run PPO or send motor commands.

## What is searched

- 512 seven-parameter trajectory actions: midpoint, all 128 corners of the allowed action box, and seeded random interior candidates.
- Every candidate sees the same nine cases: 95/100/105 mm diameter at the centre and two opposite ±10 mm diagonal positions.
- Up to 126 simultaneous GPU worlds (14 candidates × nine cases, within the configured 128-world allocation).
- Each motion still obeys existing commanded angle, speed and acceleration limits, speed-dependent torque caps and runtime fault checks.
- The best eight candidates receive CPU reference replays with motor diagnostics. Candidates with a fault-free GPU launch are also checked on 16 separately seeded pancake size/location cases, including positions outside the search diagonals.

Selection data is not the final unbiased test set. No limits, masses, friction, dimensions, reward or physical randomization were changed. A launch is not necessarily a flip. Search failure means no qualifying trajectory was found in this finite search/family, not proof that all possible two-joint mechanisms fail.

## Read the results

Download using the repository's checkpoint-download instructions. Under `probe_TIMESTAMP/`:

| File | Purpose |
|---|---|
| `launch_report.json` | Whether a confirmed baseline exists, best candidate summaries, faults and elapsed time. |
| `trials.jsonl` | GPU launch height, airtime, rotation, tracking error and faults by candidate/case. |
| `baseline_replay.json` | Original midpoint trajectory, CPU reference history and motor diagnostics. |
| `candidate_NNNN_replay.json` | Best-candidate CPU replay with requested/applied torque, speed and saturation metrics. |
| `candidate_NNNN_confirmation.json` | CPU results on separate selection cases for promising candidates. |
| `baseline_action.json` | Saved qualified action, only written if confirmation passes. |
| `BLOCKED.json` | No qualified launch / invalid initial setup; no PPO was started. |
| `COMPLETE.json` | Search finished; check `ready_for_launch_training`, not just file existence. |

Motor diagnostic arrays list shoulder first, wrist second. `saturation_fraction` is the fraction of simulated steps where requested PD torque exceeded the modeled available torque. `peak_requested_torque_nm` can exceed the limit because it describes controller demand; `peak_applied_torque_nm` remains clipped. These are simulation diagnostics, not measurements of motor current or physical output torque.

A candidate qualifies with fault-free airborne motion in at least 8/16 CPU confirmation cases. Flips and catches are reported separately; this gate establishes a starting point for launch learning only.

## What happens next

If confirmed, submit `baseten train push --config config.py`. Full training repeats the parity/search gates for its own settings rather than trusting a stale report. It starts PPO exploration near the confirmed action (initial action standard deviation 0.2). Baseline evaluation compares the learned policy against that confirmed action. Smoke remains a short installation test and does not require a launch.

If confirmation fails, full training stops before constructing PPO and retains diagnostic files. Inspect whether candidate faults, torque saturation or inadequate height dominate. Change dynamics only after physical measurements; do not increase simulated torque to manufacture success.

For manual visualization after downloading a qualified action:

```sh
mjpython sim/view_sim.py --action-file /absolute/path/to/baseline_action.json
```

Run from the repository root in your viewer environment. The viewer checks the saved physical/action contract and sends no hardware commands. No local simulation is run automatically by this update.

## Confirming actual motor behavior

The unchanged model uses 3.0 / 1.2356379 N·m zero-speed references, 90°/s maximum speeds, 180°/s² acceleration and a provisional torque-speed rolloff. To validate those assumptions, measure actual joint angle versus time under the representative cold load, including acceleration, stopping and tracking. Controller commanded position alone is not proof the shaft followed it. Update fixed model values from that evidence and repeat the search. Cloud success alone does not qualify hot-pan operation or guarantee physical flipping.
