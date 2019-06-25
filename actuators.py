"""
actuators.py
Classes to control the motors and servos. These classes
are wrapped in a mixer class before being used in the drive loop.
"""

import time
import logging
from gpiozero import PhaseEnableMotor

logger = logging.getLogger(__name__)


           
class MotorDriver(object):
    def __init__(self, change_threshold=0.02):
        self.m1 = PhaseEnableMotor(5, 12)
        self.m2 = PhaseEnableMotor(6, 13)
        self.n_users = 2
        self.speed = 0
        self.angle = 0
        self.change_threshold = change_threshold

    def set_angle(self, angle):
        self.drive(self.speed, angle)

    def set_speed(self, speed):
        self.drive(speed, self.angle)

    def drive(self, speed, angle):
        if abs(speed - self.speed) < self.change_threshold and abs(angle - self.angle) < self.change_threshold:
            return
        self.speed = speed
        self.angle = angle
        v1, v2 = calc_speed_for_each_motor(speed, angle)
        print("driving speed {:5.3f}, angle {:5.3f}, v1 {:5.3f}, v2 {:5.3f}".format(self.speed, self.angle, v1, v2))
        self.m1.value = max(min(v1, 1.0), -1.0)
        self.m2.value = max(min(v2, 1.0), -1.0)

    def release_one(self):
        if self.n_users <= 0:
            return
        self.n_users -= 1
        if self.n_users == 0:
            GPIO.cleanup()
            
def calc_speed_for_each_motor(speed, angle):
    max_speed = speed * 2
    v1 = max_speed / (2.0-abs(angle))
    v2 = max_speed - v1
    if angle < 0:
        v1, v2 = v2, v1
    return v1, v2

class PWMSteering:
    def __init__(self, motor_driver, **params):
        logger.info("Steering params %s", params)
        self.motor_driver = motor_driver

    def run(self, angle):
        logger.info("Steering angle %s", angle)
        self.motor_driver.set_angle(angle)

    def shutdown(self):
        logger.info("Steering shutdown") 
        self.run(0)
        self.motor_driver.release_one()

class PWMThrottle:
    def __init__(self, motor_driver, **params):
        logger.info("Throttle params %s", params)

        self.motor_driver = motor_driver
    
    def run(self, throttle):
        logger.info("Throttle throttle %s", throttle)
        self.motor_driver.set_speed(throttle)

    def shutdown(self):
        logger.info("Throttle shutdown") 
        self.run(0) 
        self.motor_driver.release_one()

