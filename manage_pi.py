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
import os
import logging

from docopt import docopt
import donkeycar as dk

from donkeycar.parts.camera import PiCamera
from donkeycar.parts.transform import Lambda
from donkeycar.parts.keras import KerasLinear
#from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle 
from donkeycar.parts.datastore import TubGroup, TubWriter
from donkeycar.parts.web_controller import LocalWebController
from donkeycar.parts.clock import Timestamp
from donkeycar.parts.datastore import TubGroup, TubWriter
from donkeycar.parts.keras import KerasLinear
from donkeycar.parts.transform import Lambda

from actuators import PWMSteering, PWMThrottle, MotorDriver


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

    
######################################################3


import random


import os
import time

import tornado
import tornado.ioloop
import tornado.web
import tornado.gen
import numpy as np

from donkeycar import util


class LocalWebControllerVis(tornado.web.Application):
    port = 8887

    def __init__(self, use_chaos=False):
        """
        Create and publish variables needed on many of
        the web handlers.
        """
        print('Starting Donkey Server...')

        this_dir = os.path.dirname(os.path.realpath(__file__))
        self.static_file_path = os.path.join(this_dir, 'templates', 'static')

        self.angle = 0.0
        self.throttle = 0.0
        self.mode = 'user'
        self.recording = False
        self.ip_address = util.web.get_ip_address()
        self.access_url = 'http://{}:{}'.format(self.ip_address, self.port)

        self.chaos_on = False
        self.chaos_counter = 0
        self.chaos_frequency = 1000  # frames
        self.chaos_duration = 10

        if use_chaos:
            self.run_threaded = self.run_chaos
        else:
            self.run_threaded = self._run_threaded

        handlers = [
            (r"/", tornado.web.RedirectHandler, dict(url="/drive")),
            (r"/drive", DriveAPI),
            (r"/video", VideoAPI),
            (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": self.static_file_path}),
        ]

        settings = {'debug': True}
        self.img_arr = np.zeros([100,100])
        self.pilot_angle = None
        self.pilot_throttle = None
        super().__init__(handlers, **settings)

    def run_chaos(self, img_arr=None):
        """
        Run function where steering is made random to add corrective
        """
        self.img_arr = img_arr
        if self.chaos_counter == self.chaos_frequency:
            self.chaos_on = True
            random_steering = random.random()
        elif self.chaos_counter == self.chaos_duration:
            self.chaos_on = False

        if self.chaos_on:
            return random_steering, self.throttle, self.mode, False
        else:
            return self.angle, self.throttle, self.mode, self.recording

    def say_hello(self):
        """
        Print friendly message to user
        """
        print("You can now go to {} to drive your car.".format(self.access_url))

    def update(self):
        """ Start the tornado web server. """
        self.port = int(self.port)
        self.listen(self.port)
        instance = tornado.ioloop.IOLoop.instance()
        instance.add_callback(self.say_hello)
        instance.start()

    def _run_threaded(self, img_arr=None, pilot_angle=None, pilot_throttle=None):
        self.img_arr = img_arr
        # here we save what model generates to show 
        self.pilot_angle = pilot_angle
        self.pilot_throttle = pilot_throttle
        # 
        
        return self.angle, self.throttle, self.mode, self.recording

    def run(self, img_arr=None, pilot_angle=None, pilot_throttle=None):
        return self.run_threaded(img_arr, pilot_angle, pilot_throttle)


class DriveAPI(tornado.web.RequestHandler):
    def get(self):
        data = {}
        self.render("templates/vehicle.html", **data)

    def post(self):
        """
        Receive post requests as user changes the angle
        and throttle of the vehicle on a the index webpage
        """
        data = tornado.escape.json_decode(self.request.body)
        self.application.angle = data['angle']
        self.application.throttle = data['throttle']
        self.application.mode = data['drive_mode']
        self.application.recording = data['recording']

        
from PIL import Image, ImageDraw
import math

def plot_arrow(pil_img, length, angle, color, width):
    angle = math.pi / 2 * angle
    w, h = pil_img.size
    scaling = h * 0.45
    
    dx = scaling * length * np.sin(angle)
    dy = scaling * length * np.cos(angle)
    draw = ImageDraw.Draw(pil_img)

    draw.line(
        (
            w / 2,
            h / 2,
            w / 2 + dx, 
            h / 2 - dy, 
        ),
        fill=color, width=width)
    del draw
    
def numpy_with_arrow_to_binary(img_arr, pilot_throttle, pilot_angle):
    im_pil = util.img.arr_to_img(img_arr)
    if (pilot_angle is not None and pilot_throttle is not None):
        plot_arrow(im_pil, pilot_throttle, pilot_angle, (255, 0, 0), 1)
    img = util.img.img_to_binary(im_pil)
    return img


class VideoAPI(tornado.web.RequestHandler):
    """
    Serves a MJPEG of the images posted from the vehicle.
    """

    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def get(self):
        ioloop = tornado.ioloop.IOLoop.current()
        self.set_header("Content-type", "multipart/x-mixed-replace;boundary=--boundarydonotcross")

        self.served_image_timestamp = time.time()
        my_boundary = "--boundarydonotcross\n"
        while True:
            interval = .1
            if self.served_image_timestamp + interval < time.time():
                img = numpy_with_arrow_to_binary(
                    self.application.img_arr,
                    self.application.pilot_throttle,
                    self.application.pilot_angle 
                )
                
                self.write(my_boundary)
                self.write("Content-type: image/jpeg\r\n")
                self.write("Content-length: %s\r\n\r\n" % len(img))
                self.write(img)
                print("VideoAPI ", self.application.pilot_throttle)
                self.served_image_timestamp = time.time()
                yield tornado.gen.Task(self.flush)
            else:
                yield tornado.gen.Task(ioloop.add_timeout, ioloop.time() + interval)
    
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

    V.add(kl,
          inputs=['cam/image_array'],
          outputs=['pilot/angle', 'pilot/throttle'])

    ctr = LocalWebControllerVis(use_chaos=use_chaos)
    V.add(ctr,
          inputs=['cam/image_array', 'pilot/angle', 'pilot/throttle'],
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


def train(cfg, tub_names, new_model_path, base_model_path=None):
    """
    use the specified data in tub_names to train an artifical neural network
    saves the output trained model as model_name
    """
    X_keys = ['cam/image_array']
    y_keys = ['user/angle', 'user/throttle']

    new_model_path = os.path.expanduser(new_model_path)

    kl = KerasLinear()
    if base_model_path is not None:
        base_model_path = os.path.expanduser(base_model_path)
        kl.load(base_model_path)

    print('tub_names', tub_names)
    if not tub_names:
        tub_names = os.path.join(cfg.DATA_PATH, '*')
    tubgroup = TubGroup(tub_names)
    train_gen, val_gen = tubgroup.get_train_val_gen(X_keys, y_keys,
                                                    batch_size=cfg.BATCH_SIZE,
                                                    train_frac=cfg.TRAIN_TEST_SPLIT)

    total_records = len(tubgroup.df)
    total_train = int(total_records * cfg.TRAIN_TEST_SPLIT)
    total_val = total_records - total_train
    print('train: %d, validation: %d' % (total_train, total_val))
    steps_per_epoch = total_train // cfg.BATCH_SIZE
    print('steps_per_epoch', steps_per_epoch)

    kl.train(train_gen,
             val_gen,
             saved_model_path=new_model_path,
             steps=steps_per_epoch,
             train_split=cfg.TRAIN_TEST_SPLIT)


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





