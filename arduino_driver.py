"""
actuators.py
Classes to control the motors and servos. These classes
are wrapped in a mixer class before being used in the drive loop.
"""

import time
import logging
from contextlib import contextmanager

import serial
import re

logger = logging.getLogger(__name__)

DEFAULT_ANGLE_PWM = 1400
DEFAULT_THROTTLE_PWM = 1400

ports = ['/dev/ttyUSB0', '/dev/ttyUSB1']


def connect_serial():
    for port in ports:
        print("connecting to port {}".format(port))
        try:
            ser = serial.Serial(port, 115200, timeout=0.01)
            print('connected')
            return ser
        except serial.serialutil.SerialException as e:
            if e.errno == 2:
                continue
            raise
    raise FileNotFoundError("Unable to open ports {}".format(ports))


state_re = re.compile(r"ret: s=(\d+) t=(\d+)")


class Ard:
    def __init__(self):
        self.connection = connect_serial()
        self.latest = ""

    def send(self, str_param):
        out = str_param + "\r\n"
        self.connection.write(out.encode('latin'))

    def read_latest_state(self):
        line = self.latest + self.connection.read(1000).decode('latin')
        split = line.rsplit('\n', 1)
        if len(split) == 1:
            self.latest = split[0]
            return None, None
        line, self.latest = split

        steering, throttle = None, None
        for command in line.splitlines():
            match = state_re.match(command)
            if match:
                steering = int(match.group(1))
                throttle = int(match.group(2))
        return steering, throttle

    def close(self):
        self.connection.close()


class ArduinoDriver:
    def __init__(self, **params):
        logger.info("Setting up arduino driver")
        self.ard = Ard()

    def run(self, mode, angle, throttle):
        if mode == 'user':
            self.ard.send('g 1')
            self.ard.send('r')
            user_angle, user_throttle = self.ard.read_latest_state()
            return user_angle, user_throttle
        else:
            self.ard.send('g 2 {} {}'.format(self.scale_angle(angle), self.scale_throttle(throttle)))
            return None, None

    def shutdown(self):
        logger.info("Steering shutdown")
        self.ard.send('g 1')
        self.ard.close()

    def scale_angle(self, angle):
        return DEFAULT_ANGLE_PWM + 500 * angle

    def scale_throttle(self, throttle):
        return DEFAULT_THROTTLE_PWM + 50 * throttle
