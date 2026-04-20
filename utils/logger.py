# Copyright (C) 2024 MediaTek Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""mtk_llm_sdk logging utilities."""

import enum
import logging
import os
import sys
import threading
import traceback
from datetime import datetime


class ColorBook(enum.Enum):
    """Define the color book."""

    RED = '\33[91m'
    GREEN = '\33[92m'
    YELLOW = '\33[93m'
    BLUE = '\33[94m'
    WHITE = '\33[0m'


class LevelBook(enum.Enum):
    """Define the level book."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


# Logger name, also prefix to each log line.
_LOGGER_NAME = 'mtk_llm_sdk'  # Logger name, also the log prefix

# Log level preference, default is INFO
_LOG_LEVEL = LevelBook[os.getenv('MTK_LLM_SDK_LOG_LEVEL', 'INFO')].value
_LOG_LEVEL = _LOG_LEVEL if __debug__ else max(_LOG_LEVEL, LevelBook.ERROR.value)

# Enable to log streaming or file
# At least one of streaming and file should be enabled
strtobool = lambda string: string.lower() not in ('false', 'f', '0')  # noqa: E731
_LOG_STREAM = strtobool(os.getenv('MTK_LLM_SDK_LOG_EN_STREAM', 'True'))
_LOG_FILE = strtobool(os.getenv('MTK_LLM_SDK_LOG_EN_FILE', 'True'))
_LOG_FORMAT = '%(asctime)s [%(name)s:%(levelname)s] %(filename)s:%(lineno)d: %(message)s'
assert _LOG_STREAM is True or _LOG_FILE is True, 'None of LOG_STREAM and LOG_FILE enabled'

# Global logger
_logger = None
_lock = threading.Lock()  # Lock for thread safety logger creation


def _find_caller(stack_info=False, stacklevel=1):  # pylint: disable=unused-argument
    """Track back system caller stack to find caller file, function and line-no.

    Returns:
        4-element tuple, which represent caller's filename, line-no, code name, stack info.
            Reference to `logging` for full spec.

    Raises:
        Exceptions raised during tracing caller stack.
    """
    try:
        # logger_f = execute frame of this file, this function.
        logger_f = sys._getframe(3)  # noqa: SLF001

        # Track back until caller frame is found
        # This is needed because we may redirect function call within this logger.
        caller_f = logger_f
        while caller_f.f_code.co_filename == logger_f.f_code.co_filename:
            caller_f = caller_f.f_back

        # Stack info if required
        sinfo = None
        if stack_info:
            sinfo = '\n'.join(traceback.format_stack())

        # Return tuple information according to `findcaller` of logging module.
        return (caller_f.f_code.co_filename, caller_f.f_lineno, caller_f.f_code.co_name, sinfo)

    except:
        print('Unexpected runtime caller stack.')
        raise


def _get_logger():
    """Get logger, or create one if logger is not exist.

    Intended to create mtk_llm_sdk logger instance, or return created logger directly. The logger is
    created by `logging`, python built-in logging utilities.
    """
    global _logger

    # Return logger if already exist
    if _logger:
        return _logger

    with _lock:
        # Return logger if already exist, in lock section because it may created by other thread.
        if _logger:
            return _logger

        # Create logger with name 'mtk_llm_sdk'
        logger = logging.getLogger(_LOGGER_NAME)
        logger.setLevel(_LOG_LEVEL)
        logger.findCaller = _find_caller
        logger.propagate = False

        # Log format, e.g. [mtk_llm_sdk:INFO] install.py:7: It is log message
        formatter = logging.Formatter(_LOG_FORMAT)
        if _LOG_STREAM:
            # Setup stream handler
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        if _LOG_FILE:
            # Setup file handler
            out_dir = os.getenv('MTK_LLM_SDK_LOG_DIR', 'logs')
            os.makedirs(out_dir, exist_ok=True)
            fh = logging.FileHandler(
                os.path.join(
                    out_dir,
                    f'{datetime.now().strftime("%Y%m%d_%H%M%S")}{"_" + os.getenv("MTK_LLM_SDK_SCRIPT", "")}.log',
                )
            )
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        _logger = logger
        return _logger


def _color_string(msg, color=None):
    """Color the given string.

    Args:
        msg: String type. The message to format with colour.
        color: String type. Identifier to the color book. Default to None.

    Returns:
        Colored string if color is given, original string if color is None.

    Raises:
        KeyError: Invalid color identifier.
    """
    if color is None:
        return msg
    return ColorBook[color.upper()].value + msg + ColorBook.WHITE.value


def set_level(level):
    """Interface to change logging level.

    Args:
        level: The logging level, must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL.

    Raises:
        ValueError: The level is not one of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    """
    # if level not in LevelBook:
    #     raise ValueError('Unknown level')
    _get_logger().setLevel(level)


def set_logfile_path(file_path):
    """Set the log-file path for the file handler.

    Remove the original file handler, then set a new one.

    Args:
        file_path: A string or `pathlib.Path` object. The path of the log file.
    """
    global _logger  # pylint: disable=global-variable-not-assigned

    if not _logger:
        _get_logger()

    with _lock:
        # Remove the existed `FileHandler`.
        for handler in _logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()
                handler.close()
                _logger.removeHandler(handler)

        # Set the new file handler.
        fh = logging.FileHandler(file_path)
        fh.setFormatter(logging.Formatter(_LOG_FORMAT))
        _logger.addHandler(fh)


def log(level, msg, *args, color=None, **kwargs):
    """Log message in given log level."""
    msg = _color_string(msg, color)
    _get_logger().log(level, msg, *args, **kwargs)


def debug(msg, *args, color=None, **kwargs):
    """Log message in DEBUG level."""
    msg = _color_string(msg, color)
    _get_logger().debug(msg, *args, **kwargs)


def info(msg, *args, color=None, **kwargs):
    """Log message in INFO level."""
    msg = _color_string(msg, color)
    _get_logger().info(msg, *args, **kwargs)


def warning(msg, *args, color=None, **kwargs):
    """Log message in WARNING level."""
    msg = _color_string(msg, color)
    _get_logger().warning(msg, *args, **kwargs)


def error(msg, *args, color=None, **kwargs):
    """Log message in ERROR level."""
    msg = _color_string(msg, color)
    err = kwargs.pop('err', RuntimeError)
    _get_logger().error(msg, *args, **kwargs)
    raise err


def fatal(msg, *args, color=None, **kwargs):
    """Log message in FATAL(=CRITICAL) level."""
    msg = _color_string(msg, color)
    _get_logger().fatal(msg, *args, **kwargs)


def critical(msg, *args, color=None, **kwargs):
    """Log message in CRITICAL level."""
    msg = _color_string(msg, color)
    _get_logger().critical(msg, *args, **kwargs)


def exception(msg, *args, color=None, **kwargs):
    """Log message with specific exception message."""
    msg = _color_string(msg, color)
    _get_logger().exception(msg, *args, **kwargs)


def vlog(msg, *args, color=None, **kwargs):
    """Log message with INFO level, and extra level by verbose."""
    msg = _color_string(msg, color)
    _get_logger().info(msg, *args, **kwargs)


def vlog_if(condition, msg, *args, color=None, **kwargs):
    """Conditional log message with INFO level, and extra level by verbose."""
    if condition:
        msg = _color_string(msg, color)
        _get_logger().info(msg, *args, **kwargs)


def check(condition, msg, *args, color='yellow', **kwargs):
    """Check the condition, log and raise if the condition is not meet."""
    if not condition:
        msg = _color_string('Check failed: ' + msg, color)
        _get_logger().error(msg, *args, **kwargs)
        raise ValueError('Condition check failed.')


def dcheck(condition, msg, *args, color='yellow', **kwargs):
    """In debug version, check the condition, log and raise if the condition is not meet."""
    if __debug__ and not condition:
        msg = _color_string('Check failed: ' + msg, color)
        _get_logger().error(msg, *args, **kwargs)
        raise ValueError('Condition check failed.')


def check_type(value, expected_type, *args, msg=None, color='yellow', **kwargs):
    """Check the value type, log and raise if the value type is not expected.

    Args:
        value: The value to be checked.
        expected_type: The expected type of the value. It could be a type or tuple of types.
        args: Additional positional arguments.
        msg: The specific logging message. If not specified, use the default logging message.
            Defaults to None.
        color: The color of the message. Defaults to yellow.
        kwargs: Additional keyword arguments.

    Raises:
        TypeError: The type of value is not expected.
        TypeError: The expected_type is not a type. Raised by `isinstance`.
    """
    if not isinstance(value, expected_type):
        msg = f'Check failed: {msg}' if msg else f'Expected type {expected_type}, but received {type(value)}.'
        _get_logger().error(_color_string(msg, color), *args, **kwargs)
        raise TypeError('The type is not expected.')


def dcheck_type(value, expected_type, *args, msg=None, color='yellow', **kwargs):
    """In debug version, check the value type, log and raise if the value type is not expected.

    Args:
        value: The value to be checked.
        expected_type: The expected type of the value. It could be a type or tuple of types.
        args: Additional positional arguments.
        msg: The specific logging message. If not specified, use the default logging message.
                Defaults to None.
        color: The color of the message. Defaults to yellow.
        kwargs: Additional keyword arguments.

    Raises:
        TypeError: The type of value is not expected.
        TypeError: The expected_type is not a type. Raised by `isinstance`.
    """
    if __debug__ and not isinstance(value, expected_type):
        msg = f'Check failed: {msg}' if msg else f'Expected type {expected_type}, but received {type(value)}.'
        _get_logger().error(_color_string(msg, color), *args, **kwargs)
        raise TypeError('The type is not expected.')


def unreachable(*args, msg='', color='yellow', **kwargs):
    """Mark unreachable code, and raise runtime exception if program executed.

    Raises:
        RuntimeError: the unreachable code is executed.
    """
    if __debug__:
        msg = _color_string(f'Unreachable error {msg}', color)
        _get_logger().error(msg, *args, **kwargs)
        raise RuntimeError('Execute unreachable code.')


def remove_empty_logs():
    """Removes all empty log files."""
    if not os.path.exists('logs'):
        return

    for f in os.listdir('logs'):
        logfile = os.path.join('logs', f)
        if os.path.exists(logfile) and os.path.getsize(logfile) == 0:
            os.remove(logfile)
