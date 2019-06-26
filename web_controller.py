import time

import math
import os
import random

import numpy as np
import tornado
import tornado.gen
import tornado.web
from PIL import ImageDraw

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