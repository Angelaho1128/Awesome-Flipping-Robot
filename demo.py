"""
demo.py

Opens the vision window (OpenCV pancake detection) and the MuJoCo simulation
window at the same time, so you can show both side by side.

RUN (from the repo root, with your venv active):
    python3 demo.py                 # webcam window + simulation window
    python3 demo.py --source 1      # choose the camera for the vision window
    python3 demo.py --no-sim        # vision window only
    python3 demo.py --no-vision     # simulation window only

WHY TWO SEPARATE PROCESSES?
    On a Mac, each of these windows needs to own the main thread of its
    program, so they can't share one Python process. This launcher starts both
    and keeps them running side by side. It also starts the simulation with
    `mjpython` on macOS, because MuJoCo's interactive viewer (launch_passive,
    used by sim/view_sim.py) only opens when run that way. `mjpython` is
    installed together with the `mujoco` package.

HOW TO STOP
    Close both windows (press Q in the vision window), or press Ctrl+C in
    this terminal to stop everything.
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VISION_SCRIPT = ROOT / "vision" / "pancake_detection.py"
SIM_SCRIPT = ROOT / "sim" / "view_sim.py"


def find_mjpython():
    """mjpython lives next to python inside the venv (installed by `pip install mujoco`)."""
    beside_python = Path(sys.executable).parent / "mjpython"
    if beside_python.exists():
        return str(beside_python)
    return shutil.which("mjpython")


def build_jobs(args):
    """Work out which programs to start. Returns a list of (name, command)."""
    jobs = []

    if not args.no_vision:
        if not VISION_SCRIPT.exists():
            print(f"[demo] Can't find {VISION_SCRIPT} - skipping the vision window.")
        else:
            command = [sys.executable, str(VISION_SCRIPT)]
            if args.source:
                command += ["--source", args.source]
            jobs.append(("vision", command))

    if not args.no_sim:
        if not SIM_SCRIPT.exists():
            print(f"[demo] Can't find {SIM_SCRIPT} - skipping the simulation window.")
        elif sys.platform == "darwin":
            mjpython = find_mjpython()
            if mjpython is None:
                print("[demo] macOS needs 'mjpython' to open the MuJoCo viewer, but it "
                      "wasn't found.\n"
                      "       Install MuJoCo in this environment:  "
                      "python3 -m pip install mujoco\n"
                      "       Skipping the simulation window.")
            else:
                jobs.append(("simulation", [mjpython, str(SIM_SCRIPT)]))
        else:
            jobs.append(("simulation", [sys.executable, str(SIM_SCRIPT)]))

    return jobs


def stop_all(processes):
    """Politely stop anything still running, then force it if needed."""
    for process in processes.values():
        if process.poll() is None:
            process.terminate()
    deadline = time.time() + 3
    for process in processes.values():
        try:
            process.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            process.kill()


def main():
    parser = argparse.ArgumentParser(description="Run the vision and simulation windows together")
    parser.add_argument("--source", default=None,
                        help="camera for the vision window: a number like 0 or 1, "
                             "or a video file (default: auto-detect)")
    parser.add_argument("--no-vision", action="store_true", help="don't open the vision window")
    parser.add_argument("--no-sim", action="store_true", help="don't open the simulation window")
    args = parser.parse_args()

    jobs = build_jobs(args)
    if not jobs:
        print("[demo] Nothing to run.")
        return 1

    processes = {}
    try:
        for name, command in jobs:
            print(f"[demo] Starting {name}: {' '.join(command)}")
            # Run from the repo root so relative paths (sim/assets/arm.xml) work
            processes[name] = subprocess.Popen(command, cwd=ROOT)
            time.sleep(0.5)  # stagger the two windows opening

        print("[demo] Windows are opening. Drag them side by side. "
              "Press Ctrl+C here to stop everything.")

        while processes:
            for name, process in list(processes.items()):
                code = process.poll()
                if code is None:
                    continue
                del processes[name]
                if code == 0:
                    print(f"[demo] The {name} window was closed.")
                else:
                    print(f"[demo] The {name} program stopped with an error "
                          f"(exit code {code}). Scroll up for its message.")
            time.sleep(0.3)

    except KeyboardInterrupt:
        print("\n[demo] Stopping...")
    finally:
        stop_all(processes)

    return 0


if __name__ == "__main__":
    sys.exit(main())