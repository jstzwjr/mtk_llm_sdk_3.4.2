# Copyright (C) 2024 MediaTek Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# ==================================================================================================
"""Monitor for peak RAM and VRAM usage during script execution."""

import ctypes
import inspect
import os
import subprocess
import time
from functools import wraps
from multiprocessing import Array, Event, Process, Value
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event as EventClass
from typing import Any, Callable, Optional, TypeVar, cast

from . import logger

try:
    import psutil

    psutil_present = True
except ModuleNotFoundError:
    psutil_present = False

try:
    import nvidia_smi  # nvidia-ml-py3

    nvidia_smi_present = True
except ModuleNotFoundError:
    nvidia_smi_present = False


class MemoryVRAMScope:
    """Context manager for monitoring RAM and VRAM usage.

    Use as context `with MemoryVRAMScope(name, gpu_ids):`.

    Args:
        name (str): The name of the scope.
        gpu_ids (list): List of GPU IDs to monitor.

    Attributes:
        _name (str): The name of the scope.
        _max_ram_process (Synchronized): The maximum RAM used by the process.
        _max_ram (Synchronized): The maximum RAM used.
        _max_vram (Synchronized): The maximum VRAM used.
        _total_ram (Synchronized): The total RAM used.
        _total_vram (Synchronized): The total VRAM used.
        _gpu_ids (Synchronized): The GPU IDs to monitor.
        _stop_event (Event): Event to signal the monitor process to stop.
        _monitor_process (Optional[Process]): The monitor process.
        _start_time (float): The start time of the scope.
    """

    def __init__(self, name: str, gpu_ids: list):
        """Initializes the MemoryVRAMScope context manager.

        Args:
            name (str): The name of the scope.
            gpu_ids (list): List of GPU IDs to monitor.

        Attributes:
            _name (str): The name of the scope.
            _max_ram_process (Synchronized): The maximum RAM used by the process.
            _max_ram (Synchronized): The maximum RAM used.
            _max_vram (Synchronized): The maximum VRAM used.
            _total_ram (Synchronized): The total RAM used.
            _total_vram (Synchronized): The total VRAM used.
            _gpu_ids (Synchronized): The GPU IDs to monitor.
            _stop_event (Event): Event to signal the monitor process to stop.
            _monitor_process (Optional[Process]): The monitor process.
            _start_time (float): The start time of the scope.
        """
        self._name = name
        self._max_ram_process: Synchronized = Value('L', 0)
        self._max_ram: Synchronized = Value('L', 0)
        self._max_vram: Synchronized = Value('L', 0)
        self._total_ram: Synchronized = Value('L', 0)
        self._total_vram: Synchronized = Value('L', 0)
        self._gpu_ids: Synchronized = Array(ctypes.c_int, gpu_ids)

        self._stop_event = Event()
        self._monitor_process: Optional[Process] = None
        self._start_time = None

    def __enter__(self):
        """Starts the monitor process when entering the context."""
        self._start_time = time.time()
        pid = os.getpid()
        self._monitor_process = Process(
            target=_ram_vram_monitor,
            args=(
                pid,
                self._stop_event,
                self._max_ram_process,
                self._max_ram,
                self._max_vram,
                self._total_ram,
                self._total_vram,
                self._gpu_ids,
            ),
        )
        self._monitor_process.start()

    def __exit__(self, exception_type, exception_value, exception_traceback):
        """Stops the monitor process and prints the usage statistics when exiting the context."""
        self._stop_event.set()  # Signal the monitor process to stop
        assert self._monitor_process is not None
        self._monitor_process.join()

        elapsed_time = time.time() - self._start_time

        max_ram_pid = bytes2human(int(self._max_ram_process.value))
        tot_ram = bytes2human(int(self._total_ram.value))
        max_vram = bytes2human(int(self._max_vram.value))
        tot_vram = bytes2human(int(self._total_vram.value))

        if exception_value is None:
            logger.info(
                (f'RAM-Peak = {max_ram_pid}/{tot_ram} ' if psutil_present else 'install psutil to monitor peak RAM ')
                + '/'
                + (
                    f' VRAM-Peak = {max_vram}/{tot_vram} '
                    if nvidia_smi_present
                    else ' install nvidia-ml-py3 to monitor peak VRAM '
                )
                + '/'
                + (f' Elapsed time: {format_time(elapsed_time)}')
            )
        logger.remove_empty_logs()


class MemoryScope:
    """Context manager for monitoring RAM usage.

    Use as context `with MemoryScope(name):`.

    Args:
        name (str): The name of the scope.

    Attributes:
        _name (str): The name of the scope.
        _max_ram_process (Synchronized): The maximum RAM used by the process.
        _max_ram (Synchronized): The maximum RAM used.
        _total_ram (Synchronized): The total RAM used.
        _stop_event (Event): Event to signal the monitor process to stop.
        _monitor_process (Optional[Process]): The monitor process.
        _start_time (float): The start time of the scope.
    """

    def __init__(self, name: str):
        """Initializes the MemoryScope context manager.

        Args:
            name (str): The name of the scope.
        """
        self._name = name
        self._max_ram_process: Synchronized = Value('L', 0)
        self._max_ram: Synchronized = Value('L', 0)
        self._total_ram: Synchronized = Value('L', 0)

        self._stop_event = Event()
        self._monitor_process: Optional[Process] = None
        self._start_time = None

    def __enter__(self):
        """Starts the monitor process when entering the context."""
        self._start_time = time.time()
        pid = os.getpid()
        self._monitor_process = Process(
            target=_ram_monitor,
            args=(pid, self._stop_event, self._max_ram_process, self._max_ram, self._total_ram),
        )
        self._monitor_process.start()

    def __exit__(self, exception_type, exception_value, exception_traceback):
        """Stops the monitor process and prints the usage statistics when exiting the context."""
        self._stop_event.set()  # Signal the monitor process to stop
        assert self._monitor_process is not None
        self._monitor_process.join()

        elapsed_time = time.time() - self._start_time

        max_ram_pid = bytes2human(int(self._max_ram_process.value))
        tot_ram = bytes2human(int(self._total_ram.value))

        if exception_value is None:
            logger.info(
                (f'RAM-Peak = {max_ram_pid}/{tot_ram} ' if psutil_present else 'install psutil to monitor peak RAM ')
                + f'/ Elapsed time: {format_time(elapsed_time)}'
            )
        logger.remove_empty_logs()


def _ram_monitor(
    pid: int,
    stop_event: EventClass,
    max_ram_pid: Synchronized,
    max_ram: Synchronized,
    total_ram: Synchronized,
) -> None:
    """Monitor Memory consumption in parallel to a given process (RAM and VRAM).

    Args:
        pid: ID of the Process to monitor
        stop_event: Shared event triggered when the monitoring has to stop
        max_ram_pid: Peak number of bytes used by the profiled process in the RAM
        max_ram: Peak number of bytes used in the RAM
        max_vram: Peak number of bytes used in the Video RAM (GPU)
        total_ram: Total number of bytes available in RAM (used+free)
        total_vram: Total number of bytes available in VRAM (used+free).
    """
    max_ram_pid_value = 0
    max_ram_value = 0

    if psutil_present:
        total_ram.value = get_total_ram_available_bytes()

        while not stop_event.is_set():
            max_ram_pid_value = max(max_ram_pid_value, get_pid_ram_used_bytes(pid))
            max_ram_value = max(max_ram_value, get_total_ram_used_bytes())
            time.sleep(1)  # Arbitrary time interval (in s)

    max_ram_pid.value = max_ram_pid_value
    max_ram.value = max_ram_value


def _ram_vram_monitor(
    pid: int,
    stop_event: EventClass,
    max_ram_pid: Synchronized,
    max_ram: Synchronized,
    max_vram: Synchronized,
    total_ram: Synchronized,
    total_vram: Synchronized,
    gpu_ids: Synchronized,
) -> None:
    """Monitor Memory consumption in parallel to a given process (RAM and VRAM).

    Args:
        pid: ID of the Process to monitor
        stop_event: Shared event triggered when the monitoring has to stop
        max_ram_pid: Peak number of bytes used by the profiled process in the RAM
        max_ram: Peak number of bytes used in the RAM
        max_vram: Peak number of bytes used in the Video RAM (GPU)
        total_ram: Total number of bytes available in RAM (used+free)
        total_vram: Total number of bytes available in VRAM (used+free).
        gpu_ids: GPU ID
    """
    max_ram_pid_value = 0
    max_ram_value = 0
    max_vram_value = 0

    if nvidia_smi_present:
        nvidia_smi.nvmlInit()

    gpu_handles = []
    if nvidia_smi_present:
        for gpu_id in gpu_ids:
            gpu_handles.append(nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_id))
        total_vram.value = int(sum(nvidia_smi.nvmlDeviceGetMemoryInfo(handle).total for handle in gpu_handles))
    if psutil_present:
        total_ram.value = get_total_ram_available_bytes()

    while not stop_event.is_set():
        if psutil_present:
            max_ram_pid_value = max(max_ram_pid_value, get_pid_ram_used_bytes(pid))
            max_ram_value = max(max_ram_value, get_total_ram_used_bytes())
        gpu_infos = []
        if nvidia_smi_present:
            for i in range(len(gpu_ids)):
                gpu_infos.append(nvidia_smi.nvmlDeviceGetMemoryInfo(gpu_handles[i]))
            max_vram_value = max(max_vram_value, int(sum(gpu_info.used for gpu_info in gpu_infos)))
        time.sleep(1)  # Arbitrary time interval (in s)

    max_ram_pid.value = max_ram_pid_value
    max_ram.value = max_ram_value
    max_vram.value = max_vram_value
    if nvidia_smi_present:
        nvidia_smi.nvmlShutdown()


def format_time(seconds):
    """Formats a time duration given in seconds into a human-readable string.

    Args:
        seconds (float): The time duration in seconds.

    Returns:
        str: The formatted time string.
    """
    if seconds < 60:
        return f'{int(seconds)}s'
    if seconds < 3600:
        mins = int(seconds // 60)
        sec = int(seconds % 60)
        return f'{mins}:{0 if sec < 10 else ""}{sec}'
    hr = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    sec = int((seconds % 3600) % 60)
    return f'{hr}:{0 if mins < 10 else ""}{mins}:{0 if sec < 10 else ""}{sec}'


def get_function_name(func: Callable) -> str:
    """Get file and function name."""
    module_name = func.__module__.split('.')[-1]
    if module_name == '__main__':
        module_name = os.path.splitext(os.path.basename(inspect.getfile(func)))[0]
    return f'{module_name}::{func.__name__}'


FuncT = TypeVar('FuncT', bound=Callable[..., Any])


def get_total_ram_available_bytes() -> int:
    """Return the total number of bytes available on the platform."""
    return int(psutil.virtual_memory().total)


def get_total_ram_used_bytes() -> int:
    """Return the number of bytes currently used on the platform."""
    return int(psutil.virtual_memory().used)


def get_pid_ram_used_bytes(pid: int) -> int:
    """Return the number of bytes currently used by a given process.

    (Use os.getpid() to get current process ID).
    """
    return int(psutil.Process(pid).memory_info().rss)


def bytes2human(number: int, decimal_unit: bool = True) -> str:
    """Convert number of bytes in a human readable string.

    >>> bytes2human(10000, True)
    '10.00 KB'
    >>> bytes2human(10000, False)
    '9.77 KiB'
    Args:
        number (int): Number of bytes
        decimal_unit (bool): If specified, use 1 kB (kilobyte)=10^3 bytes.
            Otherwise, use 1 KiB (kibibyte)=1024 bytes
    Returns:
        str: Bytes converted in readable string.
    """
    symbols = ['K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']
    symbol_values = [
        (symbol, 1000 ** (i + 1) if decimal_unit else (1 << (i + 1) * 10)) for i, symbol in enumerate(symbols)
    ]

    for symbol, value in reversed(symbol_values):
        if number >= value:
            suffix = 'B' if decimal_unit else 'iB'
            return f'{float(number) / value:.2f}{symbol}{suffix}'

    return f'{number} B'


def memory_peak_profile(func: FuncT) -> FuncT:
    """Memory peak Profiling decorator (Both RAM and VRAM)."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            subprocess.check_output('nvidia-smi')
            cpu_only = False
        except Exception:
            cpu_only = True

        if (
            not cpu_only
            and 'CUDA_VISIBLE_DEVICES' in os.environ
            and os.environ.get('CUDA_VISIBLE_DEVICES') in ['', '-1']
        ):
            cpu_only = True

        if cpu_only:
            with MemoryScope(get_function_name(func)):
                retval = func(*args, **kwargs)
        else:
            if nvidia_smi_present:
                nvidia_smi.nvmlInit()
                if 'CUDA_VISIBLE_DEVICES' in os.environ:
                    total_num_gpus = len(os.environ.get('CUDA_VISIBLE_DEVICES').split(','))
                else:
                    total_num_gpus = nvidia_smi.nvmlDeviceGetCount()
                gpu_ids = list(range(total_num_gpus))
            else:
                gpu_ids = [-1]
            with MemoryVRAMScope(get_function_name(func), gpu_ids):
                retval = func(*args, **kwargs)
        return retval

    return cast(FuncT, wrapper)
