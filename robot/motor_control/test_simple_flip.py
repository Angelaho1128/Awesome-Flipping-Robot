"""Offline fake-controller tests: no hardware, simulation or training."""
import unittest
from unittest.mock import patch
from control_112 import ControllerError,Status
from simple_flip import (load_config,toss_plan,parse,execute,run,ramp_time,
                         perform_toss,apply_acceleration,check_acceleration)

class Link:
    def __init__(self,m):self.m=m;self.commands=[];self.stops=[];self.fail_at=None
    def status(self):return Status('Idle',self.m.pos*self.m.a_scale,'MPos')
    def command(self,line):
        self.commands.append(line)
        if len(self.commands)==self.fail_at:raise ControllerError('ack lost')
        if line.startswith('$123='):self.m.settings[123]=float(line.split('=')[1])
        else:self.m.pos+=float(line.split(' A')[1].split()[0])/self.m.a_scale
    def realtime(self,value):self.stops.append(value)
class Motor:
    def __init__(self):
        self.a_scale=.049984558;self.settings={113:600,123:1000*self.a_scale}
        self.pos=0.;self.reference=True;self.link=Link(self)
    def zero(self):self.pos=0.;self.reference=True
    def angle(self,s):return s.a/self.a_scale
    def _idle(self):return self.link.status()
    def firmware_settings(self):return self.settings.copy()

class Tests(unittest.TestCase):
    def test_intended_four_phases_and_timings(self):
        m=Motor();cfg=load_config();phases=toss_plan(m,cfg)
        self.assertEqual([p['angle_deg'] for p in phases],[-3,20,30,0])
        self.assertEqual([p['name'] for p in phases],['position','launch','catch','return'])
        self.assertGreater(phases[0]['hold_s'],0)
        for p in phases:
            self.assertAlmostEqual(ramp_time(p['distance'],p['speed'],1000),p['move_s'])
    def test_no_implicit_firmware_changes(self):
        m=Motor();m.settings[123]=1000
        with self.assertRaises(ValueError):toss_plan(m,load_config())
        self.assertEqual(m.link.commands,[])
    def test_explicit_acceleration_only_lowers_and_reads_back(self):
        m=Motor();m.settings[123]=1000
        apply_acceleration(m,load_config())
        self.assertEqual(m.link.commands,['$123=49.984558'])
        check_acceleration(m,load_config())
        m.link.commands.clear();m.settings[123]=25
        apply_acceleration(m,load_config());self.assertEqual(m.link.commands,[])
    def test_reject_failed_acceleration_readback(self):
        m=Motor();m.settings[123]=1000
        with patch.object(m.link,'command'):
            with self.assertRaises(ValueError):apply_acceleration(m,load_config())
    def test_impossible_timing_preflight(self):
        m=Motor();cfg=load_config();cfg['launch']['move_s']=.01
        with self.assertRaises(ValueError):toss_plan(m,cfg)
        self.assertEqual(m.link.commands,[])
    def test_speed_limit_preflight(self):
        m=Motor();m.settings[113]=1
        with self.assertRaises(ValueError):toss_plan(m,load_config())
        self.assertEqual(m.link.commands,[])
    def test_fault_prevents_catch_and_return(self):
        m=Motor();m.link.fail_at=2
        with patch('simple_flip.monitored_hold'):
            with self.assertRaises(ControllerError):perform_toss(m,toss_plan(m,load_config()))
        self.assertEqual(len(m.link.commands),2)
        self.assertIsNone(m.reference);self.assertIn(b'\x85',m.link.stops)
    def test_bottom_hold_before_launch_and_four_moves(self):
        m=Motor();events=[]
        def hold(motor,seconds,target):events.append((len(motor.link.commands),seconds,target))
        with patch('simple_flip.monitored_hold',side_effect=hold):
            perform_toss(m,toss_plan(m,load_config()))
        self.assertEqual(events[0],(1,.18,-3))
        self.assertEqual(events[2],(3,.15,30))
        self.assertAlmostEqual(m.pos,0)
        for c in m.link.commands:self.assertRegex(c,r'^\$J=G21 G91 A-?[0-9.]+ F[0-9.]+$')
    def test_hold_alarm_stops_remaining_moves(self):
        m=Motor()
        with patch('simple_flip.monitored_hold',side_effect=ControllerError('ALARM:10')):
            with self.assertRaises(ControllerError):perform_toss(m,toss_plan(m,load_config()))
        self.assertEqual(len(m.link.commands),1);self.assertIsNone(m.reference)
    def test_repeated_target_consumes_time(self):
        m=Motor();cfg=load_config();cfg['launch']['angle_deg']=-3;cfg['catch']['move_s']=.5
        with patch('simple_flip.monitored_hold') as hold:
            perform_toss(m,toss_plan(m,cfg))
        self.assertEqual(len(m.link.commands),3)
        self.assertEqual(hold.call_args_list[1].args[1],.32)
    def test_no_toss_without_zero(self):
        m=Motor()
        with patch('builtins.input',side_effect=['toss',EOFError]),patch('builtins.print'):
            with self.assertRaises(EOFError):run(m)
        self.assertEqual(m.link.commands,[])
    def test_parser_full_line_validated(self):
        self.assertEqual(parse('zero;toss -3 20 30 0'),[('zero',None),('toss',[-3,20,30,0])])
        for line in ('move 3;toss nan 2 3 0','toss 1 2','x 1','toss 20'):
            with self.assertRaises(ValueError):parse(line)
    def test_existing_move_defaults_preserved(self):
        import simple_flip as s
        self.assertEqual(s.MOVE_SPEED_DEG_S,60)
        self.assertEqual(s.A_UNITS_PER_DEGREE,.049984558)
    def test_execute_full_preflight(self):
        m=Motor()
        with self.assertRaises(ValueError):execute(m,[20,0],[100,-1])
        self.assertEqual(m.link.commands,[])

if __name__=='__main__':unittest.main()
