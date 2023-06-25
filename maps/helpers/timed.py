#!/usr/bin/env python3
"""Copied from microsoft/mamba"""

import time
from contextlib import contextmanager

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

@contextmanager
def timed(msg):
    print(colorize(msg, color='magenta'), end='', flush=True)
    tstart = time.perf_counter()
    yield
    print(colorize(" in %.3f seconds" % (time.perf_counter() - tstart), color='magenta'))
