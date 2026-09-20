#!/usr/bin/env python3
"""Group recorded trials by exact settings; use only explicit operator ratings."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sessions',nargs='*',type=Path)
    args=parser.parse_args()
    folders=args.sessions or sorted((Path(__file__).resolve().parents[2]/'data'/'logs'/'flip_workshop').glob('*'))
    trials={}
    for folder in folders:
        path=folder if folder.is_file() else folder/'events.jsonl'
        if not path.exists():continue
        for line in path.read_text().splitlines():
            try:e=json.loads(line)
            except json.JSONDecodeError:continue # a final line may be incomplete during a live session
            key=e.get('trial')
            if e['event']=='trial_start' and e['mode']!='slide':
                trials[key]={'config':e['config'],'mode':e['mode'],'complete':False,'rating':None,'catch_mode':None}
            elif key in trials:
                if e['event']=='trial_complete':trials[key].update(complete=True,catch_mode=e['catch_mode'])
                if e['event']=='operator_rating':trials[key]['rating']=e['rating']
    groups={}
    for trial in trials.values():
        digest=hashlib.sha256(json.dumps(trial['config'],sort_keys=True).encode()).hexdigest()[:8]
        key=(digest,trial['mode'])
        group=groups.setdefault(key,{'attempts':0,'complete':0,'rated':0,'flips':0,'vision':0})
        group['attempts']+=1;group['complete']+=trial['complete']
        group['rated']+=trial['rating'] is not None;group['flips']+=trial['rating']=='2'
        group['vision']+=trial['catch_mode']=='vision'
    print('config    mode    attempts complete rated confirmed_flips vision_catches_commanded')
    for (digest,mode),v in groups.items():
        print(digest,mode,*v.values())
    print('Ratings are operator observations; camera-commanded catches are not verified successes. Unrated trials are not successes.')

if __name__=='__main__':main()
