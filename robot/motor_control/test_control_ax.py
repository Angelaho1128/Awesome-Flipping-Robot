"""Fake-controller checks only: no serial hardware or training access."""
import unittest
from unittest.mock import patch
from control_112 import Status,ControllerError,parse_status
from control_ax import ArmAX,interactive
from test_control_112 import FakeLink


class AXLink(FakeLink):
    def __init__(self):
        super().__init__()
        self.x=20.
        self.settings.update({37:9,100:800,110:4000,120:750,140:2800,150:32,210:35,338:7,
                              142:1800,212:50})
    def status(self):return Status(self.state,self.position,"MPos",z=200.,x=self.x)


class AXTests(unittest.TestCase):
    def ready(self,scale=.1):
        link=AXLink();arm=ArmAX(link,scale,a_limits=(-45,45),x_limits=(-45,45));arm.initialize();arm.zero()
        return link,arm

    def test_status_preserves_both_axes(self):
        self.assertEqual(parse_status("<Idle|MPos:12.5,0,200,-9>"),Status("Idle",-9,"MPos",z=200,x=12.5))

    def test_x_distance_and_feed_are_converted_together(self):
        link,arm=self.ready()
        arm.move_axis("x",10,speed=5)
        self.assertEqual(link.commands[-1],"$J=G21 G91 X1.000000 F30.000")
        link.x=21;link.state="Idle"
        arm.move_axis("A",15,speed=3)
        self.assertEqual(link.commands[-1],"$J=G21 G91 A15.000000 F180.000")

    def test_relative_x_and_hold_cancel(self):
        link,arm=self.ready();link.x=20.5
        arm.move_axis("X",-2,relative=True)
        self.assertEqual(link.commands[-1],"$J=G21 G91 X-0.200000 F30.000")
        arm.hold();self.assertEqual(arm.pending,{})
        self.assertEqual(link.realtime_bytes,[b"\x85"])

    def test_configuration_preserves_other_axes_and_motor_limits(self):
        link=AXLink();link.settings[37]=6
        before=link.settings.copy()
        arm=ArmAX(link,1);arm.initialize(True,60)
        self.assertEqual(link.settings[37],15)
        self.assertEqual(link.settings[210],60)
        for k in before:
            if k not in (37,210):self.assertEqual(link.settings[k],before[k])
        self.assertLess(link.commands.index("$210=60"),link.commands.index("$37=15"))

    def test_missing_x_hold_requires_explicit_configuration(self):
        link=AXLink();link.settings[37]=8
        with self.assertRaises(ControllerError):ArmAX(link,1).initialize()
        self.assertEqual(link.commands,["$I","$$"])

    def test_invalid_current_and_wrong_driver_rejected_before_writes(self):
        for key,value in ((140,0),(210,0),(338,6),(100,0)):
            link=AXLink();link.settings[key]=value
            with self.assertRaises(ControllerError):ArmAX(link,1).initialize(True)
            self.assertEqual(link.commands,["$I","$$"])

    def test_moving_blocks_but_idle_short_of_target_completes(self):
        link,arm=self.ready();arm.move_axis("X",10)
        with self.assertRaises(ControllerError):arm.move_axis("A",5)
        link.state="Idle";link.x=20.7
        arm.move_axis("X",10)
        self.assertEqual(link.commands[-1],"$J=G21 G91 X0.300000 F30.000")
        arm.hold();arm.move_axis("A",5)

    def test_limits_and_conversion_are_required(self):
        for scale in (0,-1,float("nan")):
            with self.assertRaises(ValueError):ArmAX(AXLink(),scale)
        link,arm=self.ready()
        for axis,angle in (("X",46),("A",-46),("Z",1)):
            with self.assertRaises(ValueError):arm.move_axis(axis,angle)
        self.assertFalse(any(x.startswith("$J=") for x in link.commands))

    def test_uncertain_move_is_never_retried(self):
        link,arm=self.ready();original=link.command
        def command(line):
            if line.startswith("$J="):
                link.commands.append(line);raise ControllerError("acknowledgement lost")
            return original(line)
        link.command=command
        with self.assertRaises(ControllerError):arm.move_axis("X",5)
        self.assertIsNone(arm.reference)
        self.assertEqual(link.realtime_bytes,[b"\x85"])
        self.assertEqual(sum(x.startswith("$J=") for x in link.commands),1)

    def test_terminal_uses_x_and_rejects_old_z_command(self):
        link,arm=self.ready()
        with patch("builtins.input",side_effect=["z to 10","x to 10 3","hold","a by 5","quit"]),patch("builtins.print"):
            interactive(arm)
        self.assertEqual([x for x in link.commands if x.startswith("$J=")],
                         ["$J=G21 G91 X1.000000 F18.000","$J=G21 G91 A5.000000 F300.000"])





class CalibrationTests(unittest.TestCase):
    def test_a_correction_converts_distance_speed_and_readback(self):
        link=AXLink();arm=ArmAX(link,.1);arm.initialize();arm.zero()
        before=list(link.commands)
        self.assertAlmostEqual(arm.calibrate_axis("a",45,135),1/3)
        self.assertEqual(link.commands,before)
        self.assertEqual(arm.x_scale,.1)
        self.assertIsNone(arm.reference)
        arm.zero();arm.move_axis("A",45,speed=3)
        self.assertEqual(link.commands[-1],"$J=G21 G91 A15.000000 F60.000")
        link.position+=15;link.state="Idle"
        self.assertAlmostEqual(arm.angles(link.status())["A"],45)
        arm.move_axis("X",10,speed=3)
        self.assertEqual(link.commands[-1],"$J=G21 G91 X1.000000 F18.000")

    def test_full_turn_bounds_and_saved_scale(self):
        link=AXLink();arm=ArmAX(link,.1,a_units_per_degree=1/3)
        arm.initialize();arm.zero()
        with self.assertRaises(ValueError):arm.move_axis("A",361)
        arm.move_axis("A",360)
        self.assertEqual(link.commands[-1],"$J=G21 G91 A120.000000 F100.000")
        arm.hold();arm.move_axis("X",360)
        self.assertEqual(link.commands[-1],"$J=G21 G91 X36.000000 F30.000")

    def test_bad_calibration_does_not_change_scale(self):
        link=AXLink();arm=ArmAX(link,.1)
        for commanded,measured in ((0,135),(45,0),(45,-135),(45,float("nan"))):
            with self.assertRaises(ValueError):arm.calibrate_axis("A",commanded,measured)
        link.state="Jog"
        with self.assertRaises(ControllerError):arm.calibrate_axis("A",45,135)
        self.assertEqual(arm.a_scale,1)
        self.assertEqual(link.commands,[])

    def test_a_only_scale_and_full_turn(self):
        from control_112 import Motor112
        link=FakeLink();motor=Motor112(link)
        motor.initialize();motor.zero();motor.calibrate(45,135);motor.zero()
        motor.move_to(360,3)
        self.assertEqual(link.commands[-1],"$J=G21 G91 A120.000000 F60.000")
        link.position+=120;link.state="Idle"
        self.assertAlmostEqual(motor.angle(link.status()),360)
        motor.zero()



class ModuloTests(unittest.TestCase):
    def test_shortest_path_and_multiple_turns(self):
        link=AXLink();arm=ArmAX(link,.1,a_units_per_degree=1/3,modulo=True,enforce_limits=False)
        arm.initialize();arm.zero()
        arm.move_axis("A",350)
        self.assertEqual(link.commands[-1],"$J=G21 G91 A-3.333333 F100.000")
        link.position-=10/3;link.state="Idle"
        arm.move_axis("A",10)
        self.assertEqual(link.commands[-1],"$J=G21 G91 A6.666667 F100.000")
        link.position+=20/3;link.state="Idle"
        arm.move_axis("A",720,relative=True)
        self.assertEqual(link.commands[-1],"$J=G21 G91 A240.000000 F100.000")

    def test_360_is_zero_and_shorthand(self):
        link=AXLink();arm=ArmAX(link,.1,modulo=True,enforce_limits=False)
        arm.initialize();arm.zero();before=list(link.commands)
        arm.move_axis("A",360)
        self.assertEqual(before,link.commands)
        with patch("builtins.input",side_effect=["a 5 2","hold","quit"]),patch("builtins.print"):
            interactive(arm)
        self.assertIn("$J=G21 G91 A5.000000 F120.000",link.commands)



class MotionTests(unittest.TestCase):
    def test_physical_ramps_both_axes_and_current_unchanged(self):
        link=AXLink();arm=ArmAX(link,1/45,a_units_per_degree=.45)
        arm.initialize();old=link.settings.copy()
        arm.configure_motion(True)
        self.assertAlmostEqual(link.settings[120],5/45)
        self.assertAlmostEqual(link.settings[123],2.25)
        self.assertEqual(link.settings[110],old[110])
        self.assertEqual(link.settings[113],old[113])
        for key in (140,150,210,100,103):self.assertEqual(link.settings[key],old[key])
        arm.zero();arm.move_axis("X",5)
        self.assertTrue(link.commands[-1].endswith("F6.667"))
        arm.hold()
        arm.move_to_angle("A",5,3)
        self.assertTrue(link.commands[-1].endswith("F81.000"))

    def test_configuration_required_and_idle_completion_api(self):
        link=AXLink();arm=ArmAX(link,.1);arm.initialize()
        with self.assertRaises(ControllerError):arm.configure_motion()
        self.assertFalse(any(c.startswith("$120=") for c in link.commands))
        arm.configure_motion(True);arm.zero();arm.move_to_angle("X",5,2)
        link.state="Idle"
        self.assertEqual(arm.wait_until_idle().state,"Idle")
        self.assertEqual(arm.pending,{})



class SimpleStartupTests(unittest.TestCase):
    def test_default_startup_preserves_motion_settings(self):
        from control_ax import main
        link=AXLink();before=link.settings.copy()
        link.close=lambda:None
        with patch("control_ax.SerialLink",return_value=link),patch("control_ax.interactive"),patch("builtins.print"):
            result=main(["--port","FAKE","--a-units-per-degree","1","--x-units-per-degree",".1"])
        self.assertEqual(result,0)
        self.assertEqual(link.settings,before)
        self.assertEqual(link.commands,["$I","$$"])

    def test_configure_alone_does_not_rewrite_motion(self):
        from control_ax import main
        link=AXLink();before=link.settings.copy();link.close=lambda:None
        with patch("control_ax.SerialLink",return_value=link),patch("control_ax.interactive"),patch("builtins.print"):
            result=main(["--port","FAKE","--a-units-per-degree","1","--x-units-per-degree",".1","--configure"])
        self.assertEqual(result,0)
        self.assertEqual(link.settings,before)
        self.assertFalse(any(c.startswith(("$120=","$123=","$110=","$113=")) for c in link.commands))

if __name__=="__main__":unittest.main()
