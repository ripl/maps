#!/usr/bin/env python3
import colorlog
from logging import INFO

def patch_logger_info(_logger):
    """This allows us to write logger.info('foo', 'bar') just like print function,
    whereas the original logger.info cannot take tuple of string (i.e., logger.info('string') only).
    """
    original_logger_info = _logger.info
    def __info(*args, **kwargs):
        original_logger_info(' '.join([str(arg) for arg in args]), **kwargs)
    return __info

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(levelname)s:%(name)s: %(message)s'))

logger = colorlog.getLogger('maps')
logger.propagate = False  # https://stackoverflow.com/a/19561320/19913466
logger.addHandler(handler)

logger.info = patch_logger_info(logger)

logger.setLevel(INFO)
