"""Configuration rejection happens before the learner sees an observation.

Vectorized static screening uses the best allowed capacity to locate the small
eligible geometric region. Motor capacity is then conditioned within its bounds.
This is a biased training distribution; it is not full-domain coverage.
"""
from collections import Counter, deque
import numpy as np
from .domain import (sample_case, condition_motor_capacity, feasibility_issues,
                     optimistic_holding_screen)


class FeasibilityError(RuntimeError):
    pass


class FeasibleCaseSampler:
    def __init__(self, cfg, profile, seed):
        self.cfg, self.profile = cfg, profile
        self.rng = np.random.default_rng(seed)
        self.pending = deque()
        self.counts = Counter()
        screen = optimistic_holding_screen(cfg, profile)
        if screen['conclusive'] and not screen['feasible']:
            raise FeasibilityError(f'Entire domain fails static holding screen: {screen}')

    def sample(self):
        if not self.pending:
            self._refill()
        return self.pending.popleft()

    def _refill(self):
        cfg, rng = self.cfg, self.rng
        p, domain, motors = cfg['physics'], cfg['domain'], cfg['motor_profiles'][self.profile]
        q1, q2 = np.radians(cfg['control']['home_deg'])
        budget = cfg['training'].get('feasibility_candidate_budget', 2_000_000)
        tested = 0
        while tested < budget:
            n = min(8192, budget-tested)
            draws = {k:rng.uniform(*v, size=n) for k,v in domain.items() if isinstance(v,list)}
            L = draws['elbow_wrist_mm']/1000
            r = draws['wrist_pan_center_mm']/1000
            pan = draws['pan_mass_kg'] + draws['pancake_mass_g']/1000
            link = p['link_fixed_mass_kg'] + p['link_linear_density_kg_m']*L
            wrist = 9.81*pan*r*np.cos(q1+q2)
            shoulder = 9.81*(p['wrist_motor_mass_kg']+draws['mount_mass_kg']+pan+link/2)*L*np.cos(q1)+wrist
            caps = np.array([motors[j+'_torque_nm'][1] for j in ('shoulder','wrist')])
            caps *= domain['torque_multiplier'][1]*p['initial_holding_torque_fraction']
            okay = (abs(shoulder)<=caps[0]) & (abs(wrist)<=caps[1])
            tested += n
            self.counts['geometric_candidates'] += n
            self.counts['static_rejections'] += int((~okay).sum())
            for i in np.flatnonzero(okay)[:256]:
                case = sample_case(cfg,rng,self.profile,overrides={k:v[i] for k,v in draws.items()})
                condition_motor_capacity(case,cfg,self.profile,rng)
                issues = feasibility_issues(case,cfg)
                if issues:
                    self.counts.update(issues)
                else:
                    self.pending.append(case)
                    self.counts['static_eligible'] += 1
            if self.pending:
                return
        raise FeasibilityError(f'No eligible configuration in {tested} candidates. '
                               f'Change measured mechanics, not rewards. Counts: {dict(self.counts)}')


def coverage_report(cfg, profile, count=10000, seed=8000000):
    """Pure arithmetic, independent unconditioned draw; no model fitting."""
    rng = np.random.default_rng(seed)
    rejected = Counter()
    for _ in range(count):
        rejected.update(feasibility_issues(sample_case(cfg,rng,profile,noisy=False),cfg))
    return {'draws':count,'reasons':dict(rejected), 'seed':seed,
            'distribution':'independent original bounds, without motor conditioning',
            'optimistic_screen':optimistic_holding_screen(cfg,profile)}
