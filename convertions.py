import binascii
import numpy as np
import math


def string2binary(words):

    text_array = bin(int(binascii.hexlify(words), 16))
    return text_array[2:]


def binary2string(string):

    bits = int(string, 2)
    text = binascii.unhexlify('%x' % bits)
    return text


def binary2array(string):

    return np.array(map(int, string))


def list2string(data):

    return ''.join(map(str, data))


def dist(x, y, ax):
    """Calculating distance between two arrays along a given axis
    """
    d = np.sqrt(np.sum((x-y)**2, axis=ax))
    return d


def distance(posi, *arg):

    if arg is not None and len(arg) != 0:
        for i in arg:
            for j in i:
                x, y = j[0], j[1]
                print('located at:', x, y)
                print('to the real position: ', dist(posi, np.array([x, y]), 0))
                print('')


def rotate(origin, point, angle):

    angle = math.radians(angle)
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy
