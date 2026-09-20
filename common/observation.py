"""Convert calibrated pan-relative measurements to the trained 23-value context.

The colour detector returns pixels. This adapter deliberately accepts millimetres
only; camera/pan calibration must be performed before calling it.
"""
import numpy as np
from common.config import settings,nominal_case
from toss.domain import observe,feasibility_issues

def from_measurements(diameter_mm,x_mm,y_mm):
    cfg=settings()
    diameter_mm,x_mm,y_mm=map(float,(diameter_mm,x_mm,y_mm))
    if not np.isfinite([diameter_mm,x_mm,y_mm]).all():
        raise ValueError('Nonfinite camera measurements')
    lo,hi=cfg['domain']['pancake_diameter_mm']
    limit=cfg['domain']['initial_offset_mm']
    if not lo<=diameter_mm<=hi or abs(x_mm)>limit or abs(y_mm)>limit:
        raise ValueError('Measurement outside trained diameter/location bounds')
    case=nominal_case()
    case.update(pancake_diameter_mm=diameter_mm,initial_x_mm=x_mm,initial_y_mm=y_mm,
                observed_initial_x_mm=x_mm,observed_initial_y_mm=y_mm)
    issues=feasibility_issues(case,cfg)
    if issues:raise ValueError('Invalid initial conditions: '+', '.join(issues))
    return observe(case),case
