#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car and train a model for it.

Usage:
    manage.py (drive) [--model=<model>] [--js] [--chaos]
    manage.py (drive_vis) [--model=<model>] [--js] [--chaos]
    manage.py (train) [--tub=<tub1,tub2,..tubn>]  (--model=<model>) [--base_model=<base_model>] [--no_cache]

Options:
    -h --help        Show this screen.
    --tub TUBPATHS   List of paths to tubs. Comma separated. Use quotes to use wildcards. ie "~/tubs/*"
    --chaos          Add periodic random steering when manually driving
"""
import logging
import os
from docopt import docopt
import donkeycar as dk

from donkeycar.parts.camera import PiCamera
from donkeycar.parts.web_controller import LocalWebController
from donkeycar.parts.clock import Timestamp
from donkeycar.parts.datastore import TubGroup, TubWriter
from donkeycar.parts.keras import KerasLinear
from donkeycar.parts.transform import Lambda

from actuators import PWMSteering, PWMThrottle, MotorDriver
from arduino_driver import ArduinoDriver

from web_controller import LocalWebControllerVis
from top_view_transform import TopViewTransform


def drive(cfg, model_path=None, use_chaos=False):

    """
    Construct a working robotic vehicle from many parts.
    Each part runs as a job in the Vehicle loop, calling either
    it's run or run_threaded method depending on the constructor flag `threaded`.
    All parts are updated one after another at the framerate given in
    cfg.DRIVE_LOOP_HZ assuming each part finishes processing in a timely manner.
    Parts may have named outputs and inputs. The framework handles passing named outputs
    to parts requesting the same named input.
    """

    V = dk.vehicle.Vehicle()

    clock = Timestamp()
    V.add(clock, outputs=['timestamp'])

    cam = PiCamera(resolution=cfg.CAMERA_RESOLUTION)
    V.add(cam, outputs=['cam/image_array'], threaded=True)

    ctr = LocalWebController(use_chaos=use_chaos)
    V.add(ctr,
          inputs=['cam/image_array'],
          outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
          threaded=True)

    # See if we should even run the pilot module.
    # This is only needed because the part run_condition only accepts boolean
    def pilot_condition(mode):
        if mode == 'user':
            return False
        else:
            return True

    pilot_condition_part = Lambda(pilot_condition)
    V.add(pilot_condition_part,
          inputs=['user/mode'],
          outputs=['run_pilot'])

    # Run the pilot if the mode is not user.
    kl = KerasLinear()
    if model_path:
        kl.load(model_path)

    V.add(kl,
          inputs=['cam/image_array'],
          outputs=['pilot/angle', 'pilot/throttle'],
          run_condition='run_pilot')

    driver = ArduinoDriver() 

    V.add(driver, inputs=['user/mode', 'pilot/angle', 'pilot/throttle'], outputs=['user/angle', 'user/throttle'])

    # add tub to save data
    inputs = ['cam/image_array', 'user/angle', 'user/throttle', 'user/mode', 'timestamp']
    types = ['image_array', 'float', 'float',  'str', 'str']

    # multiple tubs
    # th = TubHandler(path=cfg.DATA_PATH)
    # tub = th.new_tub_writer(inputs=inputs, types=types)

    # single tub
    tub = TubWriter(path=cfg.TUB_PATH, inputs=inputs, types=types)
    V.add(tub, inputs=inputs, run_condition='recording')

    # run the vehicle
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ,
            max_loop_count=cfg.MAX_LOOPS)

    


############################################################################


def drive_vis(cfg, model_path=None, use_chaos=False):
    V = dk.vehicle.Vehicle()

    clock = Timestamp()
    V.add(clock, outputs=['timestamp'])

    cam = PiCamera(resolution=cfg.CAMERA_RESOLUTION)
    V.add(cam, outputs=['cam/image_array'], threaded=True)
    
    # Run the pilot if the mode is not user.
    kl = KerasLinear()
    if model_path:
        kl.load(model_path)

    top_view_transform = TopViewTransform(cfg.CAMERA_RESOLUTION)

    V.add(Lambda(top_view_transform.wrap),
          inputs=['cam/image_array'],
          outputs=['cam/image_array_proj'])

    V.add(kl,
          inputs=['cam/image_array_proj'],
          outputs=['pilot/angle', 'pilot/throttle'])

    ctr = LocalWebControllerVis(use_chaos=use_chaos)
    V.add(ctr,
          inputs=['cam/image_array_proj', 'pilot/angle', 'pilot/throttle'],
          outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
          threaded=True)

    # See if we should even run the pilot module.
    # This is only needed because the part run_condition only accepts boolean
    def pilot_condition(mode):
        if mode == 'user':
            return False
        else:
            return True

    # Choose what inputs should change the car.
    def drive_mode(mode,
                   user_angle, user_throttle,
                   pilot_angle, pilot_throttle):
        if mode == 'user':
            return user_angle, user_throttle

        elif mode == 'local_angle':
            return pilot_angle, user_throttle

        else:
            return pilot_angle, pilot_throttle

    drive_mode_part = Lambda(drive_mode)
    V.add(drive_mode_part,
          inputs=['user/mode', 'user/angle', 'user/throttle',
                  'pilot/angle', 'pilot/throttle'],
          outputs=['angle', 'throttle'])

    #steering_controller = None #PCA9685(cfg.STEERING_CHANNEL)
    #steering = PWMSteering(controller=steering_controller,
    #                       left_pulse=cfg.STEERING_LEFT_PWM,
    #                       right_pulse=cfg.STEERING_RIGHT_PWM) 

    #throttle_controller = None #PCA9685(cfg.THROTTLE_CHANNEL)
    #throttle = PWMThrottle(controller=throttle_controller,
    #                       max_pulse=cfg.THROTTLE_FORWARD_PWM,
    #                       zero_pulse=cfg.THROTTLE_STOPPED_PWM,
    #                       min_pulse=cfg.THROTTLE_REVERSE_PWM)


    md = MotorDriver()

    steering = PWMSteering(motor_driver=md)
    throttle = PWMThrottle(motor_driver=md)

    V.add(steering, inputs=['angle'])
    V.add(throttle, inputs=['throttle'])

    # add tub to save data
    inputs = ['cam/image_array', 'user/angle', 'user/throttle', 'user/mode', 'timestamp']
    types = ['image_array', 'float', 'float',  'str', 'str']

    # multiple tubs
    # th = TubHandler(path=cfg.DATA_PATH)
    # tub = th.new_tub_writer(inputs=inputs, types=types)

    # single tub
    tub = TubWriter(path=cfg.TUB_PATH, inputs=inputs, types=types)
    V.add(tub, inputs=inputs, run_condition='recording')

    # run the vehicle
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ,
            max_loop_count=cfg.MAX_LOOPS)



if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN,
        format="%(asctime)s|%(name)-20.20s|%(levelname)-5.5s|%(message)s")

    args = docopt(__doc__)
    cfg = dk.load_config()
    
    print(args)
    
    if args['drive']:
        drive(cfg, model_path=args['--model'], use_chaos=args['--chaos'])
    elif args['drive_vis']:
        drive_vis(cfg, model_path=args['--model'], use_chaos=args['--chaos'])
    elif args['train']:
        tub = args['--tub']
        new_model_path = args['--model']
        base_model_path = args['--base_model']
        cache = not args['--no_cache']
        train(cfg, tub, new_model_path, base_model_path)





