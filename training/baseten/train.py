import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from toss.domain import load_settings
from toss.training import train,evaluate,write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("smoke","train","evaluate","probe"),default="smoke")
    parser.add_argument("--settings",default="settings.yaml")
    parser.add_argument("--profile",default=None)
    parser.add_argument("--output",default=None)
    parser.add_argument("--policy",default=None)
    parser.add_argument("--resume",default=None)
    parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    args=parser.parse_args()
    cfg=load_settings(args.settings)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    root=Path(args.output or os.environ.get("BT_CHECKPOINT_DIR","checkpoints"))
    out=root/f"{args.mode}_{stamp}"
    out.mkdir(parents=True,exist_ok=False)
    resume=args.resume
    if not resume and os.environ.get("PANCAKE_RESUME_DIR"):
        candidates=list(Path(os.environ["PANCAKE_RESUME_DIR"]).rglob("policy.zip"))
        if len(candidates)!=1:
            raise ValueError("Expected exactly one resumed policy.zip; select a single named checkpoint")
        resume=str(candidates[0])
    if args.mode in ("smoke","probe"):
        # Cloud-only runtime checks: called by run.sh AFTER the job is provisioned.
        result=subprocess.run([sys.executable,"-m","pytest","-q","tests"],text=True,capture_output=True)
        (out/"tests.txt").write_text(result.stdout+result.stderr)
        print(result.stdout,flush=True)
        if result.returncode:
            print(result.stderr,flush=True)
            raise SystemExit(result.returncode)
    if args.mode=="probe":
        from toss.launch_probe import run_probe
        report=run_probe(cfg,out,profile=args.profile)
        write_json(out/'COMPLETE.json',{'mode':'probe','search_completed':True,
            'ready_for_launch_training':report['ready_for_launch_training'],'hardware_ready':False})
        return
    if args.mode=="evaluate":
        if not args.policy: raise ValueError("--policy required for evaluate")
        policy=Path(args.policy)
    else:
        policy=train(cfg,out,profile=args.profile,smoke=args.mode=="smoke",resume=resume,device=args.device)
        summary=json.loads((out/"summary.json").read_text())
        if summary["stop_reason"] in ("termination_signal","stop_file"):
            write_json(out/"STOPPED.json",{"policy":str(policy),"reason":summary["stop_reason"],
                                          "physical_success_demonstrated":False})
            return
    count=4 if args.mode=="smoke" else cfg["training"]["evaluation_cases"]
    for profile in cfg["motor_profiles"]:
        evaluate(cfg,policy,out/f"evaluation_{profile}.json",profile,count=count)
    write_json(out/"COMPLETE.json",{"policy":str(policy),"mode":args.mode,
                                    "physical_success_demonstrated":False})
    print(f"Artifacts saved under {out}",flush=True)


if __name__=="__main__":main()
