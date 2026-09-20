"""Offline geometry/protocol tests; no camera, motor, physics engine or training."""
import copy,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'robot'/'motor_control'));sys.path.insert(0,str(HERE))
from core import rotation,ray_plane,pan_pose,speed_for_time,move_time,fit_flight,Flight,intercept
from hardware import validate,Worker,Shared


def config():
    cfg=json.loads((HERE/'config.json').read_text())
    cfg['slide'].update(angle_deg=-8,move_s=.4,min_hold_s=.18,wiggle_deg=0,wiggle_cycles=0,wiggle_move_s=.08)
    cfg['launch'].update(angle_deg=20,move_s=.36)
    cfg['catch'].update(fallback_angle_deg=30,fallback_move_s=.25,settle_s=.3)
    cfg['limits'].update(speed_deg_s=180,acceleration_deg_s2=1000,angle_min_deg=-35,angle_max_deg=45)
    cfg['six_part']=False;cfg['four_part']=False;cfg['start_angle_deg']=0;cfg['tempo']=1;cfg['queued_launch_catch']=False;cfg['three_part']=False;validate(cfg)
    cfg['geometry']['pan_inner_radius_m']=.14
    cfg['geometry']['joint_world_m']=[0,0,.3]
    cfg['geometry']['pan_center_from_joint_m']=[.24,0,0]
    return cfg


class Tests(unittest.TestCase):
    def test_moving_camera_ray_reconstruction(self):
        K=np.array([[500,0,320],[0,500,180],[0,0,1.]])
        camera0=np.array([.2,-.3,.5]);target0=np.array([.2,0,.3])
        forward=target0-camera0;forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
        down=np.cross(forward,right);Rwc=np.column_stack([right,down,forward])
        point=np.array([.27,0,.45]);joint=np.array([0,0,.3])
        for q in (-15,0,30):
            camera=joint+rotation(q)@(camera0-joint);Rcw=(rotation(q)@Rwc).T;t=-Rcw@camera
            image=K@(Rcw@point+t);pixel=image[:2]/image[2]
            np.testing.assert_allclose(ray_plane(pixel,K,Rcw,t,0),point,atol=1e-10)
    def test_coplanar_camera_rejected(self):
        with self.assertRaises(ValueError):ray_plane([0,0],np.eye(3),np.eye(3),np.array([0.,0.,-1.]),0)
    def test_ramp_solver(self):
        for d,T in [(8,.4),(28,.36),(10,.25),(35,.5)]:
            v=speed_for_time(d,T,180,1000)
            self.assertAlmostEqual(move_time(d,v,1000),T)
        with self.assertRaises(ValueError):speed_for_time(90,.1,180,1000)
    def test_ballistic_fit_and_bad_data(self):
        cfg=config()['tracking'];flight=Flight(1,np.array([.3,0,.5]),np.array([.1,0,1.]),0)
        samples=[(t,flight.at(t)) for t in np.linspace(1,1.15,6)]
        result=fit_flight(samples,cfg)
        np.testing.assert_allclose(result.at(1.3),flight.at(1.3),atol=1e-10)
        with self.assertRaises(ValueError):fit_flight(samples[:2],cfg)
        with self.assertRaises(ValueError):fit_flight([samples[0]]*5,cfg)
        bad=copy.deepcopy(samples);bad[2]=(bad[2][0],bad[2][1]+[1,0,0])
        with self.assertRaises(ValueError):fit_flight(bad,cfg)
    def test_reachable_intercept_and_unreachable(self):
        cfg=config();center,normal=pan_pose(20,cfg['geometry'])
        hit=center+normal*cfg['geometry']['pancake_half_thickness_m']
        # Analytic falling point reaches a stationary pan after .2 seconds.
        flight=Flight(0,hit+np.array([0,0,.1962]),np.zeros(3),0)
        result=intercept(flight,0,20,cfg,180,1000)
        self.assertIsNotNone(result);self.assertAlmostEqual(result['angle_deg'],20)
        self.assertAlmostEqual(result['impact_time'],.2)
        flight.position[0]+=2
        self.assertIsNone(intercept(flight,0,20,cfg,180,1000))
    def test_insufficient_actuator_time_rejected(self):
        cfg=config();cfg['catch']['command_margin_s']=1
        f=Flight(0,np.array([.3,0,.6]),np.zeros(3),0)
        self.assertIsNone(intercept(f,0,20,cfg,180,1000))
    def test_configuration_bounds(self):
        cfg=config();cfg['launch']['angle_deg']=100
        with self.assertRaises(ValueError):validate(cfg)
        cfg=config();cfg['catch']['angle_step_deg']=0
        with self.assertRaises(ValueError):validate(cfg)
        cfg=config();cfg['tracking']['webcam_latency_s']=-.1
        with self.assertRaises(ValueError):validate(cfg)
    def test_defaults_preflight(self):
        worker=object.__new__(Worker)
        worker.preflight(config(),180,1000)
    def test_acceleration_lowered_once_and_verified(self):
        from unittest.mock import Mock
        class Motor:
            a_scale=.049984558
            def __init__(self):
                self.values={113:8000,123:1000};self.link=Mock()
                self.link.command.side_effect=lambda cmd,**kw:self.values.update({123:float(cmd.split('=')[1])})
            def _idle(self):return None
            def firmware_settings(self):return self.values.copy()
        w=object.__new__(Worker);w.motor=Motor();w.shared=Shared();w.log=Mock();w.check_stop=Mock()
        _,a=w.limits(config());self.assertAlmostEqual(a,1000)
        w.motor.link.command.assert_called_once_with('$123=49.984558',timeout=.5)
        w.limits(config());self.assertEqual(w.motor.link.command.call_count,1)
        self.assertEqual(w.motor.values[113],8000)
    def test_failed_acceleration_readback_blocks_motion(self):
        from unittest.mock import Mock
        from control_112 import ControllerError
        w=object.__new__(Worker);w.motor=Mock();w.motor.a_scale=.049984558
        w.motor.firmware_settings.return_value={113:8000,123:1000};w.check_stop=Mock()
        with self.assertRaises(ControllerError):w.limits(config())
    def test_no_trial_without_zero(self):
        worker=object.__new__(Worker);worker.shared=Shared()
        with self.assertRaises(ValueError):worker.trial('vision',config())
    def test_slide_only_needs_no_camera_targets(self):
        from unittest.mock import Mock
        w=object.__new__(Worker);w.shared=Shared();w.shared.reference=True
        w.log=Mock();w.motor=Mock();w.motor.angle.return_value=0
        with patch.object(w,'limits',return_value=(180,1000)),patch.object(w,'move') as move,patch.object(w,'hold'),patch.object(w,'gate') as gate:
            w.trial('slide',config())
        gate.assert_not_called();self.assertEqual(move.call_count,1)
        self.assertEqual(move.call_args.args[0],-8)
    def test_timed_trial_phase_order_and_unknown_success(self):
        class Log:
            def __init__(self):self.events=[]
            def event(self,name,**kw):self.events.append((name,kw))
        class Motor:
            def _idle(self):return None
            def angle(self,s):return 0
        worker=object.__new__(Worker);worker.shared=Shared();worker.shared.reference=True
        worker.log=Log();worker.motor=Motor()
        with patch.object(worker,'limits',return_value=(180,1000)),patch.object(worker,'move') as move,patch.object(worker,'hold'),patch.object(worker,'gate') as gate:
            worker.trial('timed',config())
        gate.assert_not_called()
        self.assertEqual([call.args[0] for call in move.call_args_list],[-8,20,30,0])
        self.assertEqual(worker.log.events[-1][0],'trial_complete')
        self.assertIsNone(worker.log.events[-1][1]['success'])
        self.assertEqual(worker.log.events[-1][1]['catch_mode'],'timed')
    def test_vision_missing_calibration_cannot_launch(self):
        class Motor:
            def _idle(self):return None
            def angle(self,s):return 0
        worker=object.__new__(Worker);worker.shared=Shared();worker.shared.reference=True;worker.motor=Motor()
        with patch.object(worker,'limits',return_value=(180,1000)),patch.object(worker,'move') as move:
            with self.assertRaises(ValueError):worker.trial('vision',config())
        move.assert_not_called()
    def test_launch_fault_does_not_attempt_catch_or_return(self):
        from control_112 import ControllerError
        class Log:
            def event(self,*a,**kw):pass
        class Motor:
            def _idle(self):return None
            def angle(self,s):return 0
        worker=object.__new__(Worker);worker.shared=Shared();worker.shared.reference=True
        worker.log=Log();worker.motor=Motor()
        with patch.object(worker,'limits',return_value=(180,1000)),patch.object(worker,'move',side_effect=[None,ControllerError('ALARM:10')]) as move,patch.object(worker,'hold'),patch.object(worker,'gate'):
            with self.assertRaises(ControllerError):worker.trial('timed',config())
        self.assertEqual([call.args[0] for call in move.call_args_list],[-8,20])
    def test_wiggle_bounded_and_returns_to_slide(self):
        from hardware import wiggle_targets
        cfg=config();cfg['slide'].update(wiggle_deg=1,wiggle_cycles=2)
        self.assertEqual(wiggle_targets(validate(cfg)),[-7,-9,-8,-7,-9,-8])
        cfg['slide']['angle_deg']=-35
        with self.assertRaises(ValueError):validate(cfg)
    def test_faster_requests_still_obey_existing_acceleration(self):
        from hardware import effective_time
        cfg=config();cfg['tempo']=10
        duration=effective_time(30,.5,cfg,600,1000)
        self.assertGreaterEqual(duration,move_time(30,600,1000))
        self.assertLessEqual(speed_for_time(30,duration,600,1000),600)
    def test_signed_numeric_entry(self):
        import tuning
        tuning._values=[0];tuning._bounds=[(-35,45)];tuning._editing=0;tuning._text='0';tuning._replace=True
        for key in '-12.5':self.assertTrue(tuning.handle_key(ord(key)))
        tuning.handle_key(13)
        self.assertEqual(tuning._values[0],-12.5);self.assertIsNone(tuning._editing)
    def test_wiggle_executes_before_launch(self):
        from unittest.mock import Mock
        w=object.__new__(Worker);w.shared=Shared();w.shared.reference=True
        w.log=Mock();w.motor=Mock();w.motor.angle.return_value=0
        cfg=config();cfg['slide'].update(wiggle_deg=1,wiggle_cycles=1)
        with patch.object(w,'limits',return_value=(180,1000)),patch.object(w,'move') as move,patch.object(w,'hold'):
            w.trial('timed',cfg)
        self.assertEqual([c.args[0] for c in move.call_args_list],[-8,-7,-9,-8,20,30,0])
    def queued_worker(self,fail=False):
        from unittest.mock import Mock
        from types import SimpleNamespace
        from control_112 import ControllerError
        w=object.__new__(Worker);w.shared=Shared();w.shared.reference=True;w.log=Mock();w.check_stop=Mock()
        w.motor=Mock();w.motor.a_scale=.049984558;w.motor.reference=True
        events=[];position=[-8.]
        def status():
            events.append('status');return SimpleNamespace(state='Idle',a=position[0])
        def command(text,**kwargs):
            events.append(text)
            if fail:raise ControllerError('ALARM:10')
            position[0]+=float(text.split(' A')[1].split()[0])/w.motor.a_scale
        w.motor.angle.side_effect=lambda s:s.a;w.status=status;w.motor.link.command.side_effect=command
        return w,events
    def test_launch_and_catch_queued_without_intermediate_idle(self):
        w,events=self.queued_worker();w.launch_catch(config(),180,1000)
        self.assertEqual(events[0],'status');self.assertEqual(events[-1],'status')
        self.assertEqual(len(events),4)
        self.assertTrue(all(e.startswith('$J=G21 G91 A') for e in events[1:-1]))
        for cmd in events[1:-1]:
            feed=float(cmd.split(' F')[1]);self.assertLessEqual(feed,180*.049984558*60)
    def test_queue_alarm_cancels_and_does_not_send_catch(self):
        from control_112 import ControllerError
        w,events=self.queued_worker(fail=True)
        with self.assertRaises(ControllerError):w.launch_catch(config(),180,1000)
        self.assertEqual(w.motor.link.command.call_count,1)
        w.motor.link.realtime.assert_called_once_with(b'\x85')
        self.assertFalse(w.shared.reference);self.assertIsNone(w.motor.reference)
    def test_queue_all_targets_validated_before_motion(self):
        w,events=self.queued_worker();cfg=config();cfg['catch']['fallback_angle_deg']=500
        with self.assertRaises(ValueError):w.launch_catch(cfg,180,1000)
        w.motor.link.command.assert_not_called()
    def test_timed_trial_uses_queue_and_then_settles_returns(self):
        from unittest.mock import Mock
        w=object.__new__(Worker);w.shared=Shared();w.shared.reference=True;w.log=Mock()
        w.motor=Mock();w.motor.angle.return_value=0;cfg=config();cfg['queued_launch_catch']=True
        with patch.object(w,'limits',return_value=(180,1000)),patch.object(w,'move') as move,patch.object(w,'hold') as hold,patch.object(w,'launch_catch') as queued:
            w.trial('timed',cfg)
        queued.assert_called_once()
        self.assertEqual([c.args[0] for c in move.call_args_list],[-8,0])
        self.assertEqual([c.args[0] for c in hold.call_args_list],[.18,.3])
    def test_exact_three_parts_no_intermediate_waits(self):
        cfg=config();cfg['limits']['angle_min_deg']=-75
        cfg['slide']['angle_deg']=-75;cfg['launch']['angle_deg']=25;cfg['catch']['fallback_angle_deg']=-75
        w,events=self.queued_worker();w.motor.angle.side_effect=lambda s:s.a+8
        w.three_part(cfg,180,1000)
        self.assertEqual(len(events),4);self.assertEqual(events[-1],'status')
        deltas=[float(e.split(' A')[1].split()[0])/w.motor.a_scale for e in events[:3]]
        np.testing.assert_allclose(deltas,[-75,100,-100],atol=1e-7)
        for cmd in events[:3]:self.assertAlmostEqual(float(cmd.split(' F')[1]),180*w.motor.a_scale*60,places=7)
        self.assertIn('-75',w.shared.message)
    def test_three_part_fault_does_not_send_remaining_legs(self):
        from control_112 import ControllerError
        cfg=config();cfg['limits']['angle_min_deg']=-75
        cfg['slide']['angle_deg']=-75;cfg['launch']['angle_deg']=25;cfg['catch']['fallback_angle_deg']=-75
        w,events=self.queued_worker(fail=True)
        with self.assertRaises(ControllerError):w.three_part(cfg,180,1000)
        self.assertEqual(w.motor.link.command.call_count,1)
        w.motor.link.realtime.assert_called_once_with(b'\x85')
    def test_three_part_trial_bypasses_wiggle_holds_and_return(self):
        from unittest.mock import Mock
        w=object.__new__(Worker);w.shared=Shared();w.shared.reference=True
        w.motor=Mock();w.motor.angle.return_value=0;cfg=config();cfg['three_part']=True
        with patch.object(w,'limits',return_value=(180,1000)),patch.object(w,'three_part') as three,patch.object(w,'move') as move,patch.object(w,'hold') as hold:
            w.trial('timed',cfg)
        three.assert_called_once();move.assert_not_called();hold.assert_not_called()
    def test_new_start_sequence_deltas(self):
        cfg=config();cfg['start_angle_deg']=25;cfg['limits']['angle_min_deg']=-80
        cfg['slide']['angle_deg']=-80;cfg['launch']['angle_deg']=25;cfg['catch']['fallback_angle_deg']=0
        w,events=self.queued_worker();w.motor.angle.side_effect=lambda s:s.a+33
        w.three_part(cfg,180,1000)
        deltas=[float(e.split(' A')[1].split()[0])/w.motor.a_scale for e in events[:3]]
        np.testing.assert_allclose(deltas,[-105,105,-25],atol=1e-7)
    def test_three_part_rejects_wrong_start(self):
        from unittest.mock import Mock
        w=object.__new__(Worker);w.shared=Shared();w.shared.reference=True
        w.motor=Mock();w.motor.angle.return_value=0
        cfg=config();cfg['three_part']=True;cfg['start_angle_deg']=25
        with patch.object(w,'limits',return_value=(180,1000)),patch.object(w,'three_part') as three:
            with self.assertRaises(ValueError):w.trial('timed',cfg)
        three.assert_not_called()
    def test_four_part_exact_sequence_and_final_zero(self):
        cfg=config();cfg['four_part']=True;cfg['limits']['angle_min_deg']=-85
        cfg['slide']['angle_deg']=-85;cfg['launch']['angle_deg']=30;cfg['catch']['fallback_angle_deg']=0
        w,events=self.queued_worker();w.motor.angle.side_effect=lambda s:s.a+8
        w.three_part(cfg,180,1000)
        self.assertEqual(len(events),5);self.assertEqual(events[-1],'status')
        deltas=[float(e.split(' A')[1].split()[0])/w.motor.a_scale for e in events[:4]]
        np.testing.assert_allclose(deltas,[30,-115,115,-30],atol=1e-7)
        self.assertIn('Finished at 0',w.shared.message)
    def test_six_move_exact_sequence(self):
        cfg=config();cfg['six_part']=True;cfg['limits']['angle_min_deg']=-80
        cfg['slide']['angle_deg']=-80;cfg['launch']['angle_deg']=35;cfg['catch']['fallback_angle_deg']=-10
        w,events=self.queued_worker();w.motor.angle.side_effect=lambda s:s.a+8
        w.three_part(cfg,180,1000)
        self.assertEqual(len(events),7);self.assertEqual(events[-1],'status')
        deltas=[float(e.split(' A')[1].split()[0])/w.motor.a_scale for e in events[:6]]
        np.testing.assert_allclose(deltas,[-10,45,-115,115,-45,10],atol=1e-7)
    def test_arrow_key_codes(self):
        from keys import decode_key
        for code in (63232,2490368,65362):self.assertEqual(decode_key(code),'nudge_up')
        for code in (63233,2621440,65364):self.assertEqual(decode_key(code),'nudge_down')
        self.assertIsNone(decode_key(-1));self.assertIsNone(decode_key(ord('f')))
    def test_nudge_does_not_redefine_existing_zero_and_ignores_tempo(self):
        from unittest.mock import Mock
        w=object.__new__(Worker);w.motor=Mock();w.shared=Shared();w.motor.angle.return_value=3
        cfg=config();cfg['tempo']=10
        with patch.object(w,'limits',return_value=(180,1000)),patch.object(w,'move') as move:
            w.nudge(-1,cfg)
        self.assertEqual(move.call_args.args[0],2)
        self.assertTrue(move.call_args.kwargs['exact'])
        self.assertGreaterEqual(move.call_args.args[1],move_time(1,10,1000))
        w.motor.zero.assert_not_called()
    def test_detector_reused_on_synthetic_image(self):
        import cv2
        from tracking import PancakeDetector
        img=np.zeros((360,640,3),np.uint8)
        color=cv2.cvtColor(np.uint8([[[25,180,200]]]),cv2.COLOR_HSV2BGR)[0,0].tolist()
        cv2.circle(img,(300,180),40,color,-1)
        settings=config()['vision']['hsv'].copy();settings.update({'H low':18,'H high':34,'S low':100,'S high':255,'V low':130,'V high':255})
        detection,_,_=PancakeDetector().detect(img,settings)
        self.assertIsNotNone(detection);self.assertAlmostEqual(detection['x'],300,delta=1)

if __name__=='__main__':unittest.main()
