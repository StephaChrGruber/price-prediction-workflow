import sys
import time
import signal
import threading
import faulthandler

try:
    import psutil
except Exception:  # pragma: no cover - psutil optional
    psutil = None

import numpy as np
import torch

from constants import EPS


def heartbeat(msg: str = "[hb] alive", every: int = 30) -> None:
    """Print periodic heartbeat messages."""
    def _beat() -> None:
        while True:
            print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)
            time.sleep(every)
    threading.Thread(target=_beat, daemon=True).start()


def log_mem(tag: str) -> None:
    """Log the resident set size if psutil is available."""
    if psutil is None:
        return
    rss = psutil.Process().memory_info().rss / (1024 ** 3)
    print(f"[mem] {tag}: rss={rss:.2f} GB", flush=True)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_log(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return np.log(np.clip(a, EPS, None))


def setup_diagnostics() -> None:
    """Enable fault handlers and start heartbeat."""
    import sys

    faulthandler.enable()
    faulthandler.register(signal.SIGTERM, all_threads=True, chain=False)
    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    except Exception:
        pass
    heartbeat()
    print(f"[boot] py={sys.version}")
