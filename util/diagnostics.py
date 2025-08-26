import faulthandler
import logging
import signal
import threading
import time
from typing import Optional

import psutil

import numpy as np
import torch

from util.constants import EPS

__log = logging.getLogger(__name__)

__HEARTBEAT_STOP_EVENT: threading.Event
__HEARTBEAT_THREAD: threading.Thread

def heartbeat(msg: str = "Heartbeat", delay: int = 15) -> None:
    """
    Starts heartbeat loging
    :param msg: The message to display
    :param delay: delay between heartbeat in seconds
    :return: None
    """
    global  __HEARTBEAT_STOP_EVENT
    global __HEARTBEAT_THREAD

    def _beat(stop: threading.Event) -> None:
        while not stop.set():
            __log.info(f"{msg}")
            time.sleep(delay)

    __HEARTBEAT_STOP_EVENT = threading.Event()
    __HEARTBEAT_THREAD = threading.Thread(target=_beat, args=(__HEARTBEAT_STOP_EVENT,), daemon=True)
    __HEARTBEAT_THREAD.start()

def stop_heartbeat():
    """
    Stops the heartbeat
    :return: None
    """
    __HEARTBEAT_STOP_EVENT.set()
    __HEARTBEAT_THREAD.join(timeout=5)

def log_mem(tag: str) -> None:
    """
    Logs resident set size
    :param tag: a Tag
    :return: None
    """
    if psutil is None:
        return

    rss = psutil.Process().memory_info().rss / (1024 ** 3)
    __log.info(f"[{tag}] rss={rss:.2f} GB")

def set_seed(seed: int = 42) -> None:
    """
    Sets the seed for numpy.random, torch and torch.cuda
    :param seed: Seed
    :return: None
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

def safe_log(x: np.ndarray, dtype = np.float32) -> np.ndarray:
    """
    Safely calculates the log of an array
    :param x: the array
    :param dtype: Data type
    :return: Log of array
    """

    a = np.asarray(x, dtype=dtype)
    return np.log(np.clip(a, EPS, None))

def setup_diagnostics() -> None:
    """
    Sets up faulthandler for SIGTERM and SIGUSR1, starts the hearbeat
    :return: Nothing
    """
    import sys
    faulthandler.enable()
    faulthandler.register(signal.SIGTERM, all_threads=True, chain=False)
    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=True)
    except:
        pass

    heartbeat()
    __log.info(f"Python Version: {sys.version}")

def disable_diagnostics() -> None:
    """
    Stops heartbeat and unregisters SIGTERM and SIGUSR1
    :return:
    """
    stop_heartbeat()
    faulthandler.unregister(signal.SIGTERM)
    try:
        faulthandler.unregister(signal.SIGUSR1)
    except:
        pass





