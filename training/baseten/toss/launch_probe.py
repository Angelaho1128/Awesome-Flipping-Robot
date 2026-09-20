"""Bounded cloud GPU search; no motor commands, no changes to hardware limits."""
from itertools import product
from pathlib import Path
from collections import Counter
import copy,json,time
import numpy as np
from .domain import contract_hash,decode_action
from .environment import TossEnv
from .training import write_json


def candidate_actions(count,seed):
    if count<129:raise ValueError('At least 129 candidates cover midpoint and all action corners')
    corners=np.array(list(product((-1.,1.),repeat=7)),np.float32)
    rng=np.random.default_rng(seed)
    return np.concatenate([np.zeros((1,7),np.float32),corners,
                           rng.uniform(-1,1,(count-129,7)).astype(np.float32)])


def case_grid(cfg):
    lo,hi=cfg['domain']['pancake_diameter_mm'];r=cfg['domain']['initial_offset_mm']
    return [{'pancake_diameter_mm':float(d),'initial_x_mm':float(x),'initial_y_mm':float(y)}
            for d in (lo,(lo+hi)/2,hi) for x,y in ((0,0),(-r,-r),(r,r))]


def valid_launch(info):
    return bool(info.get('airborne') and not info.get('unsafe') and not info.get('rejected')
                and not info.get('invalid_action'))


def replay(cfg,profile,action,case=None,seed=0,record=False):
    env=TossEnv(cfg,profile,filter_infeasible=False,record=record)
    try:
        env.reset(seed=seed,options={'case':case} if case is not None else None)
        _,reward,_,_,info=env.step(action)
        return {'reward':reward,**info,'history':env.history if record else []}
    finally:env.close()


def run_probe(cfg,output,profile=None,parity_checked=False):
    from .warp_backend import WarpBatch
    from .gpu_checks import check_backend
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    profile=profile or cfg['training']['profile']
    p=cfg['launch_probe'];started=time.monotonic()
    if not 1<=p['minimum_valid_launches']<=p['validation_cases']:
        raise ValueError('Invalid launch confirmation threshold')
    if p['worlds']<9 or p['confirm_candidates']<1:
        raise ValueError('Search requires at least nine worlds and one confirmation candidate')
    write_json(output/'probe_config.json',cfg)
    if not parity_checked:check_backend(cfg,output,profile)
    actions=candidate_actions(p['candidates'],p['seed'])
    cases=case_grid(cfg)
    # Each action gets exactly the same nine starting conditions.
    base=[]
    centre={'pancake_diameter_mm':sum(cfg['domain']['pancake_diameter_mm'])/2,
            'initial_x_mm':0.,'initial_y_mm':0.}
    write_json(output/'baseline_replay.json',replay(cfg,profile,np.zeros(7),centre,p['seed'],True))
    for i,case in enumerate(cases):
        env=TossEnv(cfg,profile,filter_infeasible=False)
        env.reset(seed=p['seed']+i,options={'case':case})
        if env.blocked:
            write_json(output/'BLOCKED.json',{'reason':'initial_state','case':case,'issues':env.blocked})
            raise RuntimeError('Launch search cannot establish its initial states; see BLOCKED.json')
        base.append(env)
    per_batch=max(1,p['worlds']//len(base))
    gpu=WarpBatch(cfg,[copy.copy(e) for _ in range(per_batch) for e in base])
    summaries=[];all_faults=Counter()
    with (output/'trials.jsonl').open('w') as stream:
        for start in range(0,len(actions),per_batch):
            block=actions[start:start+per_batch]
            padded=np.vstack([block,np.repeat(block[-1:],per_batch-len(block),axis=0)])
            rewards,infos=gpu.run(np.repeat(padded,len(base),axis=0))
            for j in range(len(block)):
                rows=infos[j*len(base):(j+1)*len(base)]
                for k,info in enumerate(rows):
                    compact={key:info.get(key) for key in ('airborne','success','unsafe','reason','invalid_action',
                        'peak_height_m','airtime_s','flight_rotation_rad','max_tracking_error_deg','landing_offset_m','duration_s')}
                    stream.write(json.dumps({'candidate':start+j,'case_id':k,**compact})+'\n')
                    if info.get('reason'):all_faults.update([str(info['reason'])])
                    if info.get('invalid_action'):all_faults.update([info['invalid_action']])
                summaries.append({'candidate':start+j,'action':actions[start+j].tolist(),
                    'parameters':decode_action(actions[start+j]),'valid_launches':sum(valid_launch(i) for i in rows),
                    'successes':sum(bool(i.get('success')) for i in rows),
                    'peak_height_m':max(float(i.get('peak_height_m',0.)) for i in rows),
                    'faults':sum(bool(i.get('unsafe') or i.get('invalid_action')) for i in rows)})
            stream.flush()
            print(json.dumps({'event':'launch_search','candidates_completed':len(summaries),
                'candidates_total':len(actions),'safe_launches':sum(s['valid_launches'] for s in summaries),
                'successful_flips':sum(s['successes'] for s in summaries),
                'elapsed_seconds':time.monotonic()-started}),flush=True)
    ranked=sorted(summaries,key=lambda s:(s['successes'],s['valid_launches'],s['peak_height_m'],-s['faults']),reverse=True)
    # Save motor diagnostics for the best candidates even when none launches.
    centre={'pancake_diameter_mm':sum(cfg['domain']['pancake_diameter_mm'])/2,
            'initial_x_mm':0.,'initial_y_mm':0.}
    confirmations=[]
    chosen=None
    for candidate in ranked[:p['confirm_candidates']]:
        trace=replay(cfg,profile,candidate['action'],centre,p['seed'],True)
        write_json(output/f"candidate_{candidate['candidate']:04d}_replay.json",trace)
        if not candidate['valid_launches']:continue
        # Same separate random validation cases for every candidate. These are
        # selection data, not the final unbiased test report.
        rows=[replay(cfg,profile,candidate['action'],seed=p['validation_seed']+i)
              for i in range(p['validation_cases'])]
        launches=sum(valid_launch(i) for i in rows)
        result={'candidate':candidate['candidate'],'valid_launches':launches,
                'cases':len(rows),'successes':sum(bool(i['success']) for i in rows)}
        confirmations.append(result)
        write_json(output/f"candidate_{candidate['candidate']:04d}_confirmation.json",{'summary':result,'trials':rows})
        if launches>=p['minimum_valid_launches'] and chosen is None:
            chosen={**candidate,'confirmation':result}
    report={'contract_hash':contract_hash(cfg),'simulated_only':True,'hardware_ready':False,
        'ready_for_launch_training':chosen is not None,'selected':chosen,'confirmations':confirmations,
        'candidates':len(actions),'cases_per_candidate':len(cases),'faults':dict(all_faults),
        'elapsed_seconds':time.monotonic()-started,'top_candidates':ranked[:p['confirm_candidates']],
        'interpretation':'A failed finite search means no qualifying motion was found in this trajectory family; '
                         'it does not prove all possible two-joint motions are impossible.'}
    write_json(output/'launch_report.json',report)
    write_json(output/'candidate_actions.json',summaries)
    if chosen:
        write_json(output/'baseline_action.json',{'action':chosen['action'],'parameters':chosen['parameters'],
            'contract_hash':contract_hash(cfg),'simulated_only':True,'hardware_ready':False})
    else:
        write_json(output/'BLOCKED.json',{'reason':'no_confirmed_launch','training_started':False,
            'next':'Inspect candidate replay motor diagnostics; measure actual response before changing limits.'})
    for e in base:e.close()
    print(json.dumps({'event':'launch_search_finished','ready_for_launch_training':chosen is not None,
                      'report':str(output/'launch_report.json')}),flush=True)
    return report
