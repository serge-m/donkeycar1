import cv2
import numpy as np


def create_transform(shape):
    h, w, *_ = shape
    xy_points_src = np.float32([[0, h], [w, h], [0.25 * w, 0.33 * h], [0.75 * w, 0.33 * h]])
    xy_points_dst = np.float32([[0.25 * w, h], [0.75 * w, h], [0.25 * w, 0.33 * h], [0.75 * w, 0.33 * h]])
    M = cv2.getPerspectiveTransform(xy_points_src, xy_points_dst)  # The transformation matrix
    return M, xy_points_src, xy_points_dst


class TopViewTransform:
    def __init__(self, shape, in_field_name='cam/image_array', out_field_name='cam/image_array'):
        self.transform = create_transform(shape)
        self.in_field_name = in_field_name
        self.out_field_name = out_field_name

    def __call__(self, record):
        img = record[self.in_field_name]
        warped_img = self.wrap(img)
        return {**record, self.out_field_name: warped_img}

    def wrap(self, img):
        M, _, _ = self.transform
        warped_img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))  # Image warping
        return warped_img
