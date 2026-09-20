"""Offline command logic tests. No serial connection, physics or training."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from control_112 import ControllerError
from flip_test import validate, prepare, toss, preflight, corrected_scale

BASE=json.loads(Path(__file__).with_name('flip_test.json').read_text())

class FakeArm:
    def __init__(self):
        self.pos={'A':0.,'X':0.};self.moves=[];self.a_scale=.25;self.x_scale=.02;self.fail=False
    def move_axis(self,axis,target,speed):
        self.moves.append((axis,target,speed))
        if self.fail:raise ControllerError('lost acknowledgement')
        self.pos[axis]=target
    def wait_until_idle(self,timeout):return object()
    def angles(self,status):return self.pos.copy()
    def _idle(self):return object()
    def firmware_settings(self):return {100:800,103:20,110:36,113:1800,120:10,123:90,140:2000,210:50}

class FlipTests(unittest.TestCase):
    def test_ninefold_overtravel_divides_scale_by_nine(self):
        self.assertAlmostEqual(corrected_scale(.3,5,45),.3/9)
        self.assertAlmostEqual(corrected_scale(.3,-5,-45),.3/9)
        for args in ((.3,0,45),(.3,5,-45),(0,5,45),(.3,5,float('nan'))):
            with self.assertRaises(ValueError):corrected_scale(*args)

    def test_execute_requires_calibration(self):
        validate(BASE,.25)
        with self.assertRaises(ValueError):validate(BASE,1,True)
    def test_wrist_cap_cannot_be_bypassed_by_scaling(self):
        c=copy.deepcopy(BASE);c['wrist_prepare_speed_deg_s']=31
        with self.assertRaises(ValueError):validate(c,.25)
    def test_nonfinite_and_bounds_rejected(self):
        for key,value in [('shoulder_launch_speed_deg_s',float('nan')),('shoulder_launch_deg',31),('a_direction',0)]:
            c=copy.deepcopy(BASE);c[key]=value
            with self.assertRaises(ValueError):validate(c)
        c=copy.deepcopy(BASE);c['pan_pitch_limits_deg']=[-10,10]
        with self.assertRaises(ValueError):validate(c)
    @patch('flip_test.time.sleep')
    def test_wrist_fixed_during_shoulder_launch_and_return(self,sleep):
        arm=FakeArm();log=lambda *a,**kw:None
        prepare(arm,BASE,1,log);toss(arm,BASE,1,log)
        self.assertEqual(arm.moves,[('X',-5,15),('A',-10,15),('A',20,60),('A',0,45),('X',0,10)])
    @patch('flip_test.time.sleep')
    def test_direction_scaling_and_speed_scale(self,sleep):
        c=copy.deepcopy(BASE);c['a_direction']=-1
        arm=FakeArm();log=lambda *a,**kw:None
        prepare(arm,c,.25,log);toss(arm,c,.25,log)
        self.assertEqual(arm.moves[2],('A',-20,15))
    def test_failure_never_runs_catch_or_recovery(self):
        arm=FakeArm();arm.pos={'A':-10,'X':-5};arm.fail=True
        with self.assertRaises(ControllerError):toss(arm,BASE,1,lambda *a,**kw:None)
        self.assertEqual(len(arm.moves),1)
    def test_changed_prepared_position_prevents_launch(self):
        arm=FakeArm()
        with self.assertRaises(ControllerError):toss(arm,BASE,1,lambda *a,**kw:None)
        self.assertEqual(arm.moves,[])
    def test_controller_rate_ceiling_checked(self):
        arm=FakeArm()
        preflight(arm,BASE,1)
        arm.a_scale=2
        with self.assertRaises(ControllerError):preflight(arm,BASE,1)
    def test_position_mismatch_stops_sequence(self):
        arm=FakeArm();arm.angles=lambda s:{'A':-10,'X':-5}
        with self.assertRaises(ControllerError):toss(arm,BASE,1,lambda *a,**kw:None)
        self.assertEqual(len(arm.moves),1)

if __name__=='__main__':unittest.main()
