import math

""" box_center format = [x, y, width/2, height/2]"""
""" box_range format = [left, bottom, width, height] """


def crop_im(im, box_range):
    """ Return a cropped image using box_range format """
    cropped_im = im[box_range[1]:box_range[1] + box_range[3],
                  box_range[0]:box_range[0] + box_range[2]]
    return cropped_im


def box_range_to_box_center(box_range):
    """ Takes a box_range format data and convert to box_center """
    box_width = box_range[2] / 2
    box_height = box_range[3] / 2
    box_center_x = box_range[0] + math.floor(box_width)
    box_center_y = box_range[1] + math.floor(box_height)
    return [box_center_x, box_center_y, box_width, box_height]


def box_center_to_box_range(box_center):
    """ Takes box_center and covert to box_range format """
    box_width = int(box_center[2] * 2)
    box_height = int(box_center[3] * 2)
    bottom = box_center[0] - math.floor(box_center[2])
    left = box_center[1] - math.floor(box_center[3])
    return [int(bottom), int(left), box_width, box_height]


def inf_box_range(box_range, x_inf, y_inf):
    box_center = box_range_to_box_center(box_range)
    box_center[2] *= x_inf
    box_center[3] *= y_inf
    box_range = box_center_to_box_range(box_center)
    return box_range
