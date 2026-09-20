"""Protocol-only tests: no serial port, motor, physics or training execution."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from control_112 import Motor112, SerialLink, Status, ControllerError, parse_status, parse_settings, interactive, main


class USBSerialStub:
    """STM32-style connection gate and deliberately fragmented USB replies."""
    def __init__(self):
        self.dtr=False
        self.rts=False
        self.rx=bytearray()
        self.writes=[]
        self.reply=b"[VER:1.1f:grblHAL]\nok\n"

    def open(self):
        self.dtr_at_open=self.dtr

    def close(self):
        pass

    @property
    def in_waiting(self):
        return len(self.rx)

    def write(self,data):
        self.writes.append(data)
        if self.dtr and data == b"$I\n":
            self.rx.extend(self.reply)
        return len(data)

    def read(self,count):
        result=bytes(self.rx[:min(count,3)])
        del self.rx[:len(result)]
        return result


class ConnectionTests(unittest.TestCase):
    def make_link(self,device):
        with patch.dict("sys.modules",{"serial":SimpleNamespace(Serial=lambda **kwargs:device)}), \
             patch("control_112.time.sleep"):
            return SerialLink("test-port")

    def test_native_usb_dtr_and_fragmented_reply(self):
        device=USBSerialStub();link=self.make_link(device)
        self.assertTrue(device.dtr_at_open)
        self.assertEqual(link.command("$I",timeout=.1),["[VER:1.1f:grblHAL]"])
        self.assertEqual(device.writes,[b"$I\n"])

    def test_timeout_identifies_request_and_absent_reply(self):
        device=USBSerialStub();device.reply=b"";link=self.make_link(device)
        with self.assertRaisesRegex(ControllerError,r"\$I.*none"):
            link.command("$I",timeout=.01)
        self.assertFalse(link.motion_command_attempted)

    def test_connection_failure_does_not_claim_jog_cancel(self):
        with patch("sys.argv",["control_112.py","--port","test-port"]), \
             patch("control_112.SerialLink",side_effect=OSError("port busy")), \
             patch("builtins.print") as output:
            self.assertEqual(main(),1)
        messages="\n".join(str(call.args[0]) for call in output.call_args_list)
        self.assertIn("ERROR: port busy",messages)
        self.assertNotIn("Jog cancel",messages)

    def test_diagnostic_never_writes_configuration_or_cancels(self):
        link=FakeLink();link.close=lambda:None
        with patch("sys.argv",["control_112.py","--port","test-port","--diagnose"]), \
             patch("control_112.SerialLink",return_value=link),patch("builtins.print"):
            self.assertEqual(main(),0)
        self.assertEqual(link.commands,["$I","$$"])
        self.assertEqual(link.realtime_bytes,[])


class FakeLink:
    def __init__(self):
        self.settings = {1:254, 4:8, 13:0, 37:8, 103:8.888889, 113:8000, 123:30, 376:1}
        self.position = 100.0
        self.state = "Idle"
        self.commands = []
        self.realtime_bytes = []
        self.fault = None
        self.ready = False

    def command(self, line):
        self.commands.append(line)
        if line == "$I":
            return ["[VER:1.1f:grblHAL]", "[AXS:4:XYZA]"]
        if line == "$$":
            return [f"${k}={v}" for k,v in self.settings.items()]
        if line.startswith("$J="):
            self.state = "Jog"
            return []
        key,value=line[1:].split("=")
        self.settings[int(key)] = float(value)
        return []

    def status(self):
        return Status(self.state, self.position, "MPos")

    def realtime(self, data):
        self.realtime_bytes.append(data)
        if data == b"\x85":
            self.state = "Idle"


class MotorTests(unittest.TestCase):
    def ready(self):
        link=FakeLink(); motor=Motor112(link,minimum=-45,maximum=45); motor.initialize(); motor.zero()
        return link,motor

    def test_read_only_initialization_and_zero(self):
        link,motor=self.ready()
        self.assertEqual(link.commands,["$I","$$"])
        self.assertEqual(motor.reference.a,100)

    def test_network_settings_do_not_block_startup_or_configuration(self):
        for configure in (False, True):
            with self.subTest(configure=configure):
                link=FakeLink()
                link.settings.update({300:"pancake-controller",302:"192.168.5.1",
                                      303:"192.168.5.1",304:"255.255.255.0",307:""})
                if configure:
                    link.settings[37]=3
                motor=Motor112(link)
                motor.initialize(configure=configure)
                self.assertEqual(motor.settings[302],"192.168.5.1")
                self.assertEqual(motor.firmware_settings()[307],"")
                self.assertEqual(link.commands,["$I","$$","$37=11","$$","$$"]
                                 if configure else ["$I","$$","$$"])
                self.assertEqual(link.realtime_bytes,[])

    def test_invalid_motor_setting_rejected_before_configuration(self):
        for value in ("192.168.5.1", "8oops", "", "nan", "inf", "1e999"):
            with self.subTest(value=value):
                link=FakeLink();link.settings[103]=value;link.settings[37]=0
                with self.assertRaisesRegex(ControllerError,r"Invalid numeric motor setting \$103"):
                    Motor112(link).initialize(configure=True)
                self.assertEqual(link.commands,["$I","$$"])

    def test_absolute_target_converts_from_latest_position(self):
        link,motor=self.ready(); link.position=105
        motor.move_to(20,3)
        self.assertEqual(link.commands[-1],"$J=G21 G91 A15.000000 F180.000")

    def test_relative_move(self):
        link,motor=self.ready(); link.position=110
        motor.move_by(-5,2)
        self.assertEqual(link.commands[-1],"$J=G21 G91 A-5.000000 F120.000")

    def test_hold_discards_old_target_and_preserves_reference(self):
        link,motor=self.ready(); motor.move_to(30)
        link.position=108
        self.assertEqual(motor.hold().a,108)
        self.assertIsNone(motor.pending_target)
        motor.move_to(10)
        self.assertEqual(link.commands[-1],"$J=G21 G91 A2.000000 F300.000")
        self.assertEqual(link.realtime_bytes,[b"\x85"])

    def test_no_queued_moves(self):
        link,motor=self.ready(); motor.move_to(30)
        with self.assertRaises(ControllerError): motor.move_by(5)
        self.assertEqual(sum(x.startswith("$J=") for x in link.commands),1)

    def test_idle_completes_execution_even_short_of_target(self):
        link,motor=self.ready(); motor.move_to(30); link.position=127;link.state="Idle"
        motor.move_to(10)
        self.assertEqual(link.commands[-1],"$J=G21 G91 A-17.000000 F300.000")

    def test_finished_move_allows_next_move(self):
        link,motor=self.ready(); motor.move_to(20); link.position=120; link.state="Idle"
        motor.move_to(0)
        self.assertEqual(link.commands[-1],"$J=G21 G91 A-20.000000 F300.000")

    def test_nonfinite_limits_and_speed_rejected(self):
        link,motor=self.ready()
        for target,speed in [(46,5),(-46,5),(float("nan"),5),(1,0),(1,-1),(1,float("inf"))]:
            with self.assertRaises(ValueError): motor.move_to(target,speed)
        self.assertFalse(any(x.startswith("$J=") for x in link.commands))

    def test_configure_preserves_speed_acceleration_and_other_axes(self):
        link=FakeLink();link.settings[37]=3;link.settings[123]=1000
        motor=Motor112(link);changes=motor.initialize(configure=True)
        self.assertEqual(link.settings[37],11)
        self.assertEqual(link.settings[123],1000)
        self.assertEqual(link.settings[113],8000)
        self.assertEqual(link.settings[1],254)
        self.assertEqual(link.settings[4],8)
        self.assertEqual(len(changes),1)
        self.assertFalse(any(x.startswith(("$113=", "$123=")) for x in link.commands))

    def test_no_app_speed_cap_even_above_firmware_rate(self):
        link,motor=self.ready()
        motor.move_to(10,1000)
        self.assertEqual(link.commands[-1],"$J=G21 G91 A10.000000 F60000.000")
        self.assertEqual(link.settings[113],8000)

    def test_existing_high_acceleration_is_accepted_without_change(self):
        link=FakeLink();link.settings[123]=10000
        self.assertEqual(Motor112(link).initialize(),[])
        self.assertEqual(link.commands,["$I","$$"])

    def test_live_speed_setting_and_per_move_override(self):
        link,motor=self.ready();motor.set_speed(60)
        motor.move_to(10)
        self.assertTrue(link.commands[-1].endswith("F3600.000"))
        motor.hold();motor.move_to(10,120)
        self.assertTrue(link.commands[-1].endswith("F7200.000"))
        self.assertEqual(motor.default_speed,60)

    def test_terminal_remains_open_and_accepts_hold_during_motion(self):
        link=FakeLink();motor=Motor112(link);motor.initialize()
        entries=["zero","speed 60","to 20","status","settings","hold","by -5 90","quit"]
        with patch("builtins.input",side_effect=entries) as prompts, patch("builtins.print"):
            interactive(motor)
        self.assertEqual(prompts.call_count,len(entries))
        self.assertEqual(link.realtime_bytes,[b"\x85",b"\x85"])
        self.assertEqual([x for x in link.commands if x.startswith("$J=")],
                         ["$J=G21 G91 A20.000000 F3600.000","$J=G21 G91 A-5.000000 F5400.000"])

    def test_lower_acceleration_is_not_increased(self):
        link=FakeLink();link.settings[123]=5
        self.assertEqual(Motor112(link).initialize(configure=True),[])

    def test_missing_hold_mode_rejected_without_configure(self):
        link=FakeLink();link.settings[37]=0
        with self.assertRaises(ControllerError): Motor112(link).initialize()
        self.assertNotIn("$37=8",link.commands)

    def test_degrees_and_idle_delay_required(self):
        for setting,value in ((376,0),(1,0),(13,1)):
            link=FakeLink();link.settings[setting]=value
            with self.assertRaises(ControllerError): Motor112(link).initialize(configure=True)
            self.assertFalse(any(x.startswith("$J=") for x in link.commands))

    def test_alarm_is_not_hold_success(self):
        link,motor=self.ready()
        link.realtime=lambda data: None
        link.state="Alarm"
        with self.assertRaises(ControllerError):motor.hold()

    def test_coordinate_mode_change_rejected(self):
        link,motor=self.ready()
        link.status=lambda:Status("Idle",0,"WPos")
        with self.assertRaises(ControllerError):motor.move_to(10)

    def test_failed_move_is_not_retried_and_cancels(self):
        link,motor=self.ready()
        previous=link.command
        def command(line):
            if line.startswith("$J="):
                link.commands.append(line)
                raise ControllerError("ack timeout")
            return previous(line)
        link.command=command
        with self.assertRaises(ControllerError):motor.move_to(10)
        self.assertIsNone(motor.reference)
        self.assertEqual(link.realtime_bytes,[b"\x85"])
        self.assertEqual(sum(x.startswith("$J=") for x in link.commands),1)

    def test_status_and_settings_parsing(self):
        self.assertEqual(parse_status("<Jog|MPos:0,0,0,-12.5|FS:0,0>"),Status("Jog",-12.5,"MPos"))
        self.assertEqual(parse_status("<Idle|WPos:0,0,0,9>"),Status("Idle",9,"WPos"))
        self.assertEqual(parse_settings(["$37=8 (Steppers deenergize)","$103=8.888889"]),{37:8,103:8.888889})
        with self.assertRaises(ControllerError):parse_status("<Idle|MPos:0,0,0>")


if __name__ == "__main__":
    unittest.main()
