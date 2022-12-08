import collections


def init():
    global POSITIONS
    POSITIONS = collections.defaultdict(dict)

    global LOC
    LOC = []

    global ETOA
    ETOA = collections.defaultdict(dict)

    global ONSET
    ONSET = []

    global ORIENTATION
    ORIENTATION = []
